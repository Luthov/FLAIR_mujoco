import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI

import rospy
import actionlib

from inference_class_mujoco import BiteAcquisitionInference
from geometry_msgs.msg import PoseStamped
from feeding_msgs.srv import GetParam, GetParamRequest
from feeding_msgs.srv import GetScoopingPoints, GetScoopingPointsRequest
from feeding_msgs.msg import ExecuteScoopingAction, ExecuteScoopingGoal, ExecuteScoopingFeedback, ExecuteScoopingResult
from three_ddfa_ros.msg import StartActionAction, StartActionGoal, StartActionFeedback, StartActionResult

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
            self.item_portions = [2.0] * len(self.items[0])
            self.actions_remaining = 9
        else:
            self.item_portions = [2.0] * len(self.items[0])
            self.actions_remaining = 12

        self.bite_portion = 0.6
        self.bite_history = []
        self.token_history = []

        self.acq_pose = np.radians([0.0, -65.0, -25.0, 0.0, 65.0, -90.0])
        self.transfer_pose = np.radians([0.0, -65.0, -25.0, 0.0, 0.0, -90.0])
        self.perception_pose = np.radians([0.0, -65.0, -25.0, 0.0, 65.0, -90.0])

        self.inference_server = BiteAcquisitionInference(mode='motion_primitive')
    
        # Initialize the xArm API
        self.arm = XArmAPI(port="192.168.1.201", is_radian=True)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.margin_of_error = 10  # Define a margin of error

        # Service clients
        self.get_scooping_points_client = rospy.ServiceProxy('get_scooping_points', GetScoopingPoints)
        self.get_scooping_points_client.wait_for_service()

        self.get_feeding_params_client = rospy.ServiceProxy('get_feeding_parameters', GetParam)
        self.get_feeding_params_client.wait_for_service()

        # Action clients
        self.transfer_client = actionlib.SimpleActionClient('start_signal', StartActionAction)
        self.transfer_client.wait_for_server()

        # TODO: Need to change Action server name and msg when J-Anne ready
        self.execute_scooping_client = actionlib.SimpleActionClient('execute_scooping', ExecuteScoopingAction)
        self.execute_scooping_client.wait_for_server()
                         
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
        self.move_to_acq_pose()

    def move_to_perception_pose(self):
        self.move_to_pose(self.perception_pose)

    def move_to_acq_pose(self):
        self.move_to_pose(self.acq_pose)

    def move_to_transfer_pose(self):
        self.move_to_pose(self.transfer_pose)

    def execute_scooping(self, bite_size, scooping_points, bbox):
        rospy.loginfo("Calling scooping action server...")
        goal = StartActionGoal()
        goal.bite_size = bite_size
        goal.scooping_points = scooping_points
        goal.bbox = bbox
        self.scooping_client.send_goal(goal)
        self.scooping_client.wait_for_result()
        return self.scooping_client.get_result()

    def execute_bite_transfer(self):
        rospy.loginfo("Calling bite_transfer action server...")
        goal = StartActionGoal()
        goal.start = True
        self.transfer_client.send_goal(goal)
        self.transfer_client.wait_for_result()
        return self.transfer_client.get_result()
    
    def get_feeding_params(self, current_history, food_item_portions):
        rospy.loginfo("Getting feeding params")
        req_feeding_params = GetParamRequest()
        req_feeding_params.current_history = current_history
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

        return next_bite, bite_size, distance_to_mouth, exit_angle, transfer_speed, user_preference

    def get_scooping_points(self):
        rospy.loginfo("Getting scooping points")
        req_scooping_points = GetScoopingPointsRequest()
        resp_scooping_points = self.scooping_points_client(req_scooping_points)
        return resp_scooping_points

    def feed(self):

        # actions_remaining = 10
        success = True
        
        while self.actions_remaining:

            self.move_to_perception_pose()

            scooping_points, bounding_boxes = self.get_scooping_points() # Sorted in order of left to right

            # Getting feeding parameters
            print(f"=== ACTIONS REMAINING ===")
            print(self.actions_remaining)

            log_path = self.log_file + str(self.log_count)
            self.log_count += 1

            # Hard coded for mujoco
            food_item_labels = [[f"{food} {random.uniform(0.5, 1.0):.2f}" for food in items] for items in self.items]
            item_labels = food_item_labels[0]
            
            clean_item_labels = self.items[0] 

            categories = self.inference_server.categorize_items(item_labels, sim=False) 

            category_list = []
            labels_list = []
            per_food_portions = []

            for i in range(len(categories)):
                if labels_list.count(clean_item_labels[i]) == 0:
                    category_list.append(categories[i])
                    labels_list.append(clean_item_labels[i])
                    per_food_portions.append(self.item_portions[i])
                else:
                    index = labels_list.index(clean_item_labels[i])
                    per_food_portions[index] += self.item_portions[i]

            print("--------------------")
            print("Category List:", category_list)
            print("Labels List:", labels_list)
            print("Per Food Portions:", per_food_portions)
            print("--------------------\n")

            food_portion_rounded = [round(portion) for portion in per_food_portions]
            
            next_bite, bite_size, distance_to_mouth, exit_angle, transfer_speed, user_preference = self.get_feeding_params(self.bite_history, food_portion_rounded)

            if next_bite is []:
                break

            for idx in range(len(self.items)):
                if next_bite == self.items[idx]:
                    point_to_be_scooped = scooping_points[idx]
                    bowl_bbox = bounding_boxes[idx]
                    break

            # TODO: Need to make sure args are correct
            self.move_to_acq_pose()
            acquisition_success = self.execute_scooping(point_to_be_scooped, bowl_bbox, bite_size)

            if acquisition_success:
                self.move_to_transfer_pose()
                transfer_success = self.execute_bite_transfer()
            else:
                rospy.logwarn("Acquisition failed. Moving to perception pose...")
                self.move_to_perception_pose()
                continue

            for idx in range(len(clean_item_labels)):
                if (next_bite == clean_item_labels[idx]) and (transfer_success):
                    self.item_portions[idx] -= self.bite_portion
                    break

            if transfer_success:
                self.actions_remaining -= 1
                self.move_to_perception_pose()
            else:
                rospy.logwarn("Transfer failed. Moving to perception pose...")
                self.move_to_perception_pose()
                continue

            self.bite_history.append([next_bite, bite_size, distance_to_mouth, exit_angle, transfer_speed])

            if (self.actions_remaining == 0) or (next_bite is []):
                with open(self.output_directory + f'results.txt', 'a') as f:
                    f.write(f"=== FINAL HISTORY ===\n{self.bite_history}\n")
                    f.write(f"=== FINAL TOKEN HISTORY ===\n{self.token_history}\n")
                    f.write(f"=== USER PREFERENCE ===\n{user_preference}\n")

if __name__ == "__main__":
    rospy.init_node("feeding_manager")
    feeding_manager = FeedingManager()
    feeding_manager.clear_plate()