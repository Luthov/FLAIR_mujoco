import numpy as np
import time
import sys, signal
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io

import rospy
import actionlib

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from feeding_msgs.srv import GetFeedingParam, GetFeedingParamRequest
from feeding_msgs.srv import GetScoopingPoint, GetScoopingPointRequest
from feeding_msgs.msg import ScoopAction, ScoopGoal
from feeding_msgs.msg import BiteTransferAction, BiteTransferGoal

"""
Feeding Sequence:
    1. Move to reset pose
    2. Move to perception pose
    3. Get scooping points
    4. Get feeding parameters through service server
    5. Move to acq pose
    6. Execute scooping through scooping action server
    7. Move to transfer pose (Might not need coz he has it in his code)
    8. Execute bite transfer through bite transfer action server
    9. Move to reset pose
"""

class FeedingManager():

    def __init__(self):

        self.items = ['mashed potatoes', 'corn', 'minced meat']

        if len(self.items) == 3:
            self.item_portions = [3.0] * len(self.items)
            self.actions_remaining = 9
        else:
            self.item_portions = [3.0] * len(self.items)
            self.actions_remaining = 12

        self.bite_portion = 1.0
        self.bite_history = []
        self.token_history = []

        self.start_feeding = True

        self.simulated_sequence = False
        self.input_interrupts = False

        signal.signal(signal.SIGINT, self.signal_handler)

        # TODO: Need to convert these to cart coordinates
        self.acq_pose = np.radians([0.0, -65.0, -25.0, 0.0, 65.0, -90.0])
        self.transfer_pose = np.radians([0.0, -65.0, -25.0, 0.0, 0.0, -90.0])
        self.perception_pose = np.radians([0.0, -65.0, -25.0, 0.0, 65.0, -90.0])
    
        self.margin_of_error = 10  # Define a margin of error

        self.sub_user_preference = rospy.Subscriber("user_preference", String, self.user_preference_cb)

        # Service clients
        self.get_scooping_points_client = rospy.ServiceProxy('food_perception/get_scooping_point', GetScoopingPoint)
        rospy.loginfo("Waiting for perception server")
        self.get_scooping_points_client.wait_for_service()
        rospy.loginfo("Connected to perception server")

        self.get_feeding_params_client = rospy.ServiceProxy('get_feeding_parameters', GetFeedingParam)
        rospy.loginfo("Waiting for feeding parameters server")
        self.get_feeding_params_client.wait_for_service()
        rospy.loginfo("Connected to feeding parameters server")

        # Action clients
        self.transfer_client = actionlib.SimpleActionClient('start_signal', BiteTransferAction)
        rospy.loginfo("Waiting for bite transfer server")
        self.transfer_client.wait_for_server()
        rospy.loginfo("Bite transfer server started")

        self.execute_scooping_client = actionlib.SimpleActionClient('scooping_action', ScoopAction)
        rospy.loginfo("Waiting for scooping server")
        self.execute_scooping_client.wait_for_server()
        rospy.loginfo("Connected to scooping server")
    
    def setup_arm(self, ip="192.168.1.201", reset=False):
        """
        Set up the xArm robot.
        """
        self.arm = XArmAPI(port=ip, is_radian=True)
        time.sleep(0.1)
        ready = self.arm.motion_enable(enable=True)
        if ready != 0:
            max_retries = 5
            for i in range (1, max_retries+1):
                ready = self.arm.motion_enable(enable=True)
                print(f"Trying to enable motion {i}/{max_retries}: {ready}")
                if ready:
                    break
                time.sleep(0.1)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.1)

        if reset:
            self.reset()

    def disconnect_arm(self, reset=False):
        """
        Disconnect the xArm robot.
        """
        if reset:
            self.arm.reset(wait=True)
        
        self.arm.disconnect()
        print("Disconnected arm...")
        time.sleep(0.1)

    def start_feeding_button(self):

        self.setup_arm()

        self.arm.set_cgpio_analog(1, 5.0)
        print("Frank is ready, please press the button to continue.")
        while True:
            digital_inputs = self.arm.get_cgpio_digital()
            # print('Digital inputs:', digital_inputs)
            c14_state = digital_inputs[1][3]
            # print('C14 state:', c14_state)
            if c14_state == 0:
                print('Start button pressed')
                self.arm.set_cgpio_analog(1, 0.0)
                self.disconnect_arm()
                return True
            
    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        rospy.signal_shutdown("Shutting down...")
        self.reset()
        self.disconnect_arm(reset=True)
        sys.exit(0)

    def say(self, text):
        tts = gTTS(text=text, lang="en", tld="us")  # "com" gives American English accentaudio_buffer = io.BytesIO()
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)

        audio_buffer.seek(0)
        audio = AudioSegment.from_file(audio_buffer, format="mp3")
        play(audio)

    def quat2euler(self, quaternion):
        """
        Converts a quaternion (w,x,y,z) to Euler angles (roll, pitch, yaw) in radians.

        Args:
            quaternion: A list of four floats representing the quaternion [w, x, y, z].

        Returns:
            A list of three floats representing the Euler angles in radians.
        """
        w, x, y, z = quaternion

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, a_min=-1.0, a_max=1.0) # Clamp to prevent singularity
        pitch = np.arcsin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return [roll, pitch, yaw]   # in radians
    
    def move_to_pose(self, pose, wait=True):   
        """
        Move the robot to a given pose
        pose: geometry_msgs/PoseStamped
        """
        # Connect to the robot
        self.setup_arm()

        x_position = pose.pose.position.x
        y_position = pose.pose.position.y
        z_position = pose.pose.position.z
        quat = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
        rotation = R.from_quat(quat)
        euler = rotation.as_euler('xyz', degrees=True)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        self.arm.set_position(x=x_position, y=y_position, z=z_position, roll=roll, pitch=pitch, yaw=yaw, speed=40, mvacc=20, radius=0, wait=wait)
        if wait:
            self.disconnect_arm()
        #TODO: if wait is False, how to disconnect?

    def reset(self):
        self.setup_arm()
        # self.move_to_acq_pose()

        print("Moving to reset pose...")
        self.arm.set_position(x=440, y=0, z=285, roll=3.14159, pitch=-1.5708, yaw=0, speed=60, mvacc=20, radius=0, wait=True)
        self.disconnect_arm()

    def move_to_perception_pose(self):
        self.setup_arm()
        
        # self.move_to_pose(self.perception_pose)
        print("Moving to perception pose...")
        euler_angles = self.quat2euler([-0.1657, 0.5592, -0.6938, 0.4224])
        roll = euler_angles[0]
        pitch = euler_angles[1]
        yaw = euler_angles[2]

        self.arm.set_position(x=391.5737, y=201.6162, z=322.8611, roll=roll, pitch=pitch, yaw=yaw, speed=60, mvacc=20, radius=0, wait=True)
        self.disconnect_arm()

    def move_to_transfer_pose(self):
        self.setup_arm()

        self.arm.set_position(x=576, y=100, z=450, roll=-3.141, pitch=-1.368, yaw=0, speed=40, mvacc=20, radius=0, wait=True)
        print("Moved to start position") 
        self.disconnect_arm()

    def user_preference_cb(self, msg):
        """
        Callback function for user preference
        """
        self.preference_change = True
        print('[Manager Node]: Preference has been changed')

    def execute_scooping(self, scoop_point, bowl_bbox, next_bite, target_amount, get_scooping_point=False):
        """
        Sends a goal to the scooping action server.

        Args:
            scoop_pose (Point): The point for scooping action.
            bowl_bbox (BoundingBox): The bounding box of the bowl.
            next_bite (str): The next bite to scoop.
            target_amount (float): Target amount to scoop.
        """

        # Converting PointStamped() into PoseStamped()
        scoop_pose = PoseStamped()
        scoop_pose.pose.position.x = scoop_point.point.x
        scoop_pose.pose.position.y = scoop_point.point.y
        scoop_pose.pose.position.z = scoop_point.point.z

        # Create a goal
        goal = ScoopGoal()
        goal.scoop_pose = scoop_pose
        goal.bowl_bbox = bowl_bbox
        goal.get_scooping_point = get_scooping_point

        if target_amount < 1.0:
            target_amount = 1.0
        elif target_amount > 5.0:
            target_amount = 5.0

        # goal.target_amount = target_amount * 10
        if next_bite == 'mashed potatoes':
            goal.target_amount = target_amount * 5
        elif next_bite == 'corn':
            goal.target_amount = (target_amount * 10) + 5
        elif next_bite == 'minced meat':
            goal.target_amount = (target_amount * 10) + 5
        else:
            goal.target_amount = (target_amount * 10) + 5

        rospy.loginfo(f"Sending scooping goal....")
        
        # Send the goal and specify feedback and result callbacks
        self.execute_scooping_client.send_goal(goal, feedback_cb=self.feedback_callback)

        # Wait for the result
        self.execute_scooping_client.wait_for_result()
        result = self.execute_scooping_client.get_result()
        rospy.loginfo(f"Result received: success={result.scooping_success}, reward={result.reward}, actual_amount={result.actual_amount}")
        return result
    
    def feedback_callback(self, feedback):
        """
        Feedback callback for the action client.

        Args:
            feedback (ScoopingFeedback): Feedback message from the action server.
        """
        rospy.logdebug(f"[Scooping]: {feedback.message}")

    def execute_bite_transfer(self, distance_to_mouth, exit_angle, transfer_speed):
        rospy.loginfo("Calling bite_transfer action server...")
        goal = BiteTransferGoal()
        # TODO: Set defaults
        goal.distance_to_mouth = distance_to_mouth
        goal.exit_angle = exit_angle
        goal.transfer_speed = transfer_speed
        self.transfer_client.send_goal(goal)
        self.transfer_client.wait_for_result()
        return self.transfer_client.get_result()
    
    def get_feeding_params(self, current_history, food_item_portions):
        rospy.loginfo("Getting feeding params")
        req_feeding_params = GetFeedingParamRequest()
        req_feeding_params.current_history = str(current_history)
        req_feeding_params.food_item_portions = food_item_portions
        resp_feeding_params = self.get_feeding_params_client(req_feeding_params)

        print(f"RAW FEEDING SEQUENCE: {resp_feeding_params.feeding_sequence}")

        feeding_sequence = food_tuples = [(item.food_item, item.bite_size, item.distance_to_mouth, item.exit_angle, item.transfer_speed) for item in resp_feeding_params.feeding_sequence]
        success = resp_feeding_params.success
        rospy.loginfo("=== ARRANGED FEEDING SEQUENCE ===")
        rospy.logwarn(feeding_sequence)

        return feeding_sequence, success

    def get_scooping_points(self):
        rospy.loginfo("Getting scooping points")
        req_scooping_points = GetScoopingPointRequest()
        resp_scooping_points = self.get_scooping_points_client(req_scooping_points)
        scooping_points = resp_scooping_points.scooping_points
        bounding_boxes = resp_scooping_points.bounding_boxes
        return scooping_points, bounding_boxes

    def feed(self):
        
        if self.input_interrupts:
            input("If you haven't already. give a user preference using the mic. Then press ENTER to continue")

        sequence_idx = 0

        if not self.input_interrupts:
            check = 'y'

        while True:

            # print(f"=== ACTIONS REMAINING ===")
            # print(self.actions_remaining)

            ############
            # 1. RESET #
            ############
            # input("Press Enter to continue...")
            # self.reset()

            if self.simulated_sequence:
                feeding_sequence = [('minced meat', 3.0, 7.5, 90.0, 5.0), ('corn', 3.0, 7.5, 90.0, 5.0), ('mashed potatoes', 3.0, 7.5, 90.0, 5.0), ('mashed potatoes', 3.0, 7.5, 90.0, 5.0), ('corn', 3.0, 7.5, 90.0, 5.0), ('minced meat', 3.0, 7.5, 90.0, 5.0), ('corn', 3.0, 7.5, 90.0, 5.0), ('mashed potatoes', 3.0, 7.5, 90.0, 5.0), ('minced meat', 3.0, 7.5, 90.0, 5.0)]

            print("=== BITE HISTORY ===")
            print(self.bite_history)
            print("=== SEQUENCE INDEX ===")
            print(sequence_idx)

            ##############################
            # 2. Move to Perception pose #
            ##############################

            self.say("I am ready to start feeding. Please give your preference and press the button when you are ready.")
            self.start_feeding_button()

            if self.input_interrupts:
                input("Press Enter to move to perception pose...")

            self.say("I am moving to perception pose")
            self.move_to_perception_pose()

            food_portion_rounded = [round(portion) for portion in self.item_portions]

            ##########################
            # 3. Get feeding params #
            ##########################
            if not self.simulated_sequence:
                if self.start_feeding or self.preference_change:
                    
                    self.say("I am getting the feeding parameters")
                    feeding_sequence, feeding_param_success = self.get_feeding_params(
                        self.bite_history, 
                        food_portion_rounded
                    )

                    if not feeding_param_success:
                        print('Failed to get feeding parameters. Please provide a user preference.')
                        continue


                    self.start_feeding = False
                    self.preference_change = False

            print(f"Feeding sequence: {feeding_sequence}")

            next_food = feeding_sequence[sequence_idx]
            next_bite = next_food[0]
            bite_size = next_food[1]
            distance_to_mouth = next_food[2]
            exit_angle = next_food[3]
            transfer_speed = next_food[4]

            if distance_to_mouth < 5.0:
                distance_to_mouth = 5.0
            if distance_to_mouth > 10.0:
                distance_to_mouth = 10.0

            if exit_angle < 80.0:
                exit_angle = 80.0
            if exit_angle > 110.0:
                exit_angle = 110.0

            if transfer_speed < 1.0:
                transfer_speed = 1.0
            if transfer_speed > 10.0:
                transfer_speed = 10.0
            
            ###########################
            # 4a. Get scooping points #
            ###########################
            if self.input_interrupts:
                input("Press ENTER to get scooping points")
            self.say("I am getting the scooping points")
            scooping_points, bounding_boxes = self.get_scooping_points() # Sorted in order of left to right
            print(f"Scooping points: {scooping_points} | Bounding boxes: {bounding_boxes}, | Length: {len(scooping_points)}")

            if len(scooping_points) != 3:
                perception_success = False
            else:
                perception_success = True

            if self.input_interrupts:
                check = input("Was the perception successful? (y/n): ")
            if check != 'y' or not perception_success:
                rospy.logwarn("Getting scooping points failed. Moving to reset pose...")
                self.say("Getting scooping points failed. Moving to reset pose")
                self.reset()
                continue
            # TODO: Handle if in the case get scooping points fail. Can move 3 times until we decide it fails

            for idx in range(len(self.items)):
                print(f'next_bite: {next_bite} | self.items[idx]: {self.items[idx]}')
                if next_bite == self.items[idx]:
                    point_to_be_scooped = scooping_points[idx]
                    bowl_bbox = bounding_boxes[idx]
                    break

            rospy.loginfo(f'Scooping point: {point_to_be_scooped.point} | Bowl index (from left): {idx}')

            ########################
            # 4b. Execute scooping #
            ########################
            if self.input_interrupts:
                input("Press ENTER to execute scooping")
            print("SCOOPING BBOX:", bowl_bbox)
            print("SCOOPING point:", point_to_be_scooped.point.x, point_to_be_scooped.point.y, point_to_be_scooped.point.z)
            self.say(f"I am going to acquire the {next_bite}")
            acquisition_success = self.execute_scooping(point_to_be_scooped, bowl_bbox, next_bite, bite_size)
            if self.input_interrupts:
                check = input("Was the scooping successful? (y/n): ")
            if check.lower() == 'y':
                acquisition_success = True
            else:
                acquisition_success = False

            ############################
            # 5. Execute bite transfer #
            ############################
            if acquisition_success:
                if self.input_interrupts:
                    input("Press ENTER to continue to transfer pose")

                self.say("I am moving to the transfer pose")
                self.move_to_transfer_pose()
                if self.input_interrupts:
                    input("Press ENTER to execute bite transfer")

                self.say("I am going to transfer the bite now")
                transfer_success = self.execute_bite_transfer(distance_to_mouth, exit_angle, transfer_speed)
                print(f"Transfer success: {transfer_success}")

                if self.input_interrupts:
                    check = input("Was the transfer successful? (y/n): ")
                if check.lower() == 'y':
                    transfer_success = True
                else:
                    transfer_success = False
            else:
                rospy.logwarn("Acquisition failed. Moving to reset pose...")
                self.say("Acquisition failed. Moving to reset pose")
                self.reset()
                continue

            for idx in range(len(self.items)):
                if (next_bite == self.items[idx]) and (transfer_success):
                    self.item_portions[idx] -= self.bite_portion
                    break

            if transfer_success:
                # self.actions_remaining -= 1
                sequence_idx += 1
                if sequence_idx == len(feeding_sequence):
                    print("=== FEEDING HAS BEEN COMPLETED ===")
                    break
            else:
                rospy.logwarn("Transfer failed. Moving to reset pose...")
                self.say("Transfer failed. Moving to reset pose")
                self.reset()
                continue

            # Maybe want to publish history so that pref server can sub and update history
            self.bite_history.append(next_food)

            # if (self.actions_remaining == 0) or (next_bite is []):
            #     with open(self.output_directory + f'results.txt', 'a') as f:
            #         f.write(f"=== FINAL HISTORY ===\n{self.bite_history}\n")
            #         f.write(f"=== FINAL TOKEN HISTORY ===\n{self.token_history}\n")
            #         f.write(f"=== USER PREFERENCE ===\n{user_preference}\n")

if __name__ == "__main__":
    rospy.init_node("feeding_manager")
    feeding_manager = FeedingManager()

    # input("Press Enter to move to start feeding...")
    feeding_manager.feed()