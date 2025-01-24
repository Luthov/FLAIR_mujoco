import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI

import rospy
import actionlib

from geometry_msgs.msg import PoseStamped
from feeding_msgs.srv import GetFeedingParam, GetFeedingParamRequest
from feeding_msgs.srv import GetScoopingPoint, GetScoopingPointRequest
from feeding_msgs.msg import ScoopAction, ScoopGoal
from feeding_msgs.msg import BiteTransferAction, BiteTransferActionGoal

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

        self.items = ['chicken', 'rice', 'broccoli']

        if len(self.items) == 3:
            self.item_portions = [2.0] * len(self.items)
            self.actions_remaining = 9
        else:
            self.item_portions = [2.0] * len(self.items)
            self.actions_remaining = 12

        self.bite_portion = 0.6
        self.bite_history = []
        self.token_history = []

        # TODO: Need to convert these to cart coordinates
        self.acq_pose = np.radians([0.0, -65.0, -25.0, 0.0, 65.0, -90.0])
        self.transfer_pose = np.radians([0.0, -65.0, -25.0, 0.0, 0.0, -90.0])
        self.perception_pose = np.radians([0.0, -65.0, -25.0, 0.0, 65.0, -90.0])
    
        # Initialize the xArm API
        self.arm = XArmAPI(port="192.168.1.201", is_radian=True)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.margin_of_error = 10  # Define a margin of error

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

        # TODO: Need to change Action server name and msg when J-Anne ready
        self.execute_scooping_client = actionlib.SimpleActionClient('scooping_action', ScoopAction)
        rospy.loginfo("Waiting for scooping server")
        self.execute_scooping_client.wait_for_server()
        rospy.loginfo("Connected to scooping server")

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

    def reset(self):
        # self.move_to_acq_pose()
        print("Moving to reset pose...")
        self.arm.set_position(x=440, y=0, z=285, roll=3.14159, pitch=-1.5708, yaw=0, speed=40, mvacc=20, radius=0, wait=True)

    def move_to_perception_pose(self):
        # self.move_to_pose(self.perception_pose)
        print("Moving to perception pose...")
        euler_angles = self.quat2euler([-0.1657, 0.5592, -0.6938, 0.4224])
        roll = euler_angles[0]
        pitch = euler_angles[1]
        yaw = euler_angles[2]
        self.arm.set_position(x=391.5737, y=201.6162, z=322.8611, roll=roll, pitch=pitch, yaw=yaw, speed=40, mvacc=20, radius=0, wait=True)

    def move_to_transfer_pose(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)                   # Set to position control mode  
        self.arm.set_state(state=0)
        self.arm.set_position(x=800, y=-86.3, z=457.1, roll=2.852, pitch=-1.297, yaw=0.208, speed=10, radius=0, wait=True)
        print("Moved to start position")

    def execute_scooping(self, scoop_point, bowl_bbox, target_amount, get_scooping_point=False):
        """
        Sends a goal to the scooping action server.

        Args:
            scoop_pose (Point): The point for scooping action.
            bowl_bbox (BoundingBox): The bounding box of the bowl.
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
        goal.target_amount = target_amount
        goal.get_scooping_point = get_scooping_point

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
        rospy.loginfo(f"Feedback received: {feedback.message}")

    # TODO: Need to ask ethan to add distance_to_mouth and other transfer params
    def execute_bite_transfer(self, distance_to_mouth, exit_angle, transfer_speed):
        rospy.loginfo("Calling bite_transfer action server...")
        goal = BiteTransferActionGoal()
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

        next_bite = resp_feeding_params.next_bite
        bite_size = resp_feeding_params.bite_size
        distance_to_mouth = resp_feeding_params.distance_to_mouth
        exit_angle = resp_feeding_params.exit_angle
        transfer_speed = resp_feeding_params.transfer_speed
        user_preference = resp_feeding_params.user_preference
        rospy.loginfo("=== FEEDING PARAMETERS ===")
        rospy.loginfo(f"Next bite: {next_bite}")
        rospy.loginfo(f"Bite size: {bite_size}")
        rospy.loginfo(f"Distance to mouth: {distance_to_mouth}")
        rospy.loginfo(f"Exit angle: {exit_angle}")
        rospy.loginfo(f"Transfer speed: {transfer_speed}")
        rospy.loginfo(f"User preference: {user_preference}")
        # rospy.logwarn(f"HISTORY: {self.bite_history}")

        return next_bite, bite_size, distance_to_mouth, exit_angle, transfer_speed, user_preference

    def get_scooping_points(self):
        rospy.loginfo("Getting scooping points")
        req_scooping_points = GetScoopingPointRequest()
        resp_scooping_points = self.get_scooping_points_client(req_scooping_points)
        scooping_points = resp_scooping_points.scooping_points
        bounding_boxes = resp_scooping_points.bounding_boxes
        return scooping_points, bounding_boxes

    def feed(self):
        
        input("If you haven't already. give a user preference using the mic. Then press ENTER to continue")

        while self.actions_remaining:

            print(f"=== ACTIONS REMAINING ===")
            print(self.actions_remaining)

            input("Press Enter to continue...")
            self.reset()

            input("Press Enter to move to perception pose...")
            self.move_to_perception_pose()

            print("--------------------")
            print("Labels List:", self.items)
            print("Per Food Portions:", self.item_portions)
            print("--------------------\n")

            food_portion_rounded = [round(portion) for portion in self.item_portions]
            
            next_bite, bite_size, distance_to_mouth, exit_angle, transfer_speed, user_preference = self.get_feeding_params(
                self.bite_history, 
                food_portion_rounded)

            if next_bite == '':
                break
            
            input("Press ENTER to get scooping points")
            scooping_points, bounding_boxes = self.get_scooping_points() # Sorted in order of left to right
            print(f"Scooping points: {scooping_points} | Bounding boxes: {bounding_boxes}")
            check = input("Was the perception successful? (y/n): ")
            if check != 'y':
                rospy.logwarn("Getting scooping points failed. Moving to reset pose...")
                self.reset()
                continue
            # Handle if in the case get scooping points fail. Can move 3 times until we decide it fails

            for idx in range(len(self.items)):
                if next_bite == self.items[idx]:
                    point_to_be_scooped = scooping_points[idx]
                    bowl_bbox = bounding_boxes[idx]
                    break

            rospy.loginfo(f'Scooping point: {point_to_be_scooped} | Bowl index: {idx}')

            input("Press ENTER to execute scooping")
            acquisition_success = self.execute_scooping(point_to_be_scooped, bowl_bbox, bite_size)

            check = input("Was the scooping successful? (y/n): ")
            if check == 'y':
                acquisition_success = True
            if acquisition_success:

                input("Press ENTER to continue to transfer pose")
                self.move_to_transfer_pose()
                input("Press ENTER to execute bite transfer")
                transfer_success = self.execute_bite_transfer(distance_to_mouth, exit_angle, transfer_speed)

                check = input("Was the transfer successful? (y/n): ")
                if check == 'y':
                    transfer_success = True
            else:
                rospy.logwarn("Acquisition failed. Moving to reset pose...")
                self.reset()
                continue

            for idx in range(len(self.items)):
                if (next_bite == self.items[idx]) and (transfer_success):
                    self.item_portions[idx] -= self.bite_portion
                    break

            if transfer_success:
                self.actions_remaining -= 1
            else:
                rospy.logwarn("Transfer failed. Moving to reset pose...")
                self.reset()
                continue

            # Maybe want to publish history so that pref server can sub and update history
            self.bite_history.append([next_bite, bite_size, distance_to_mouth, exit_angle, transfer_speed])

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