import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import math
import os
import time
import os
import pickle

# ros imports
import rospy
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64, Bool

import threading
import utils
import cmath
import yaml
import argparse

from rs_ros import RealSenseROS
from pixel_selector import PixelSelector
# from robot_controller.franka_controller import FrankaRobotController
from robot_controller.kinova_controller import KinovaRobotController
from robot_controller.mujoco_action_controller import MujocoRobotController

PLATE_HEIGHT = 0.2
class SkillLibrary:

    def __init__(self):

        self.robot_controller = MujocoRobotController()

        print("Skill library initialized")
    
    def matrix_to_quaternion(self, matrix):
        # Extract the rotation part from the 4x4 transformation matrix
        rotation_matrix = matrix[:3, :3]
        
        # Create a Rotation object from the rotation matrix
        rotation = Rotation.from_matrix(rotation_matrix)
        
        # Convert to quaternion (qx, qy, qz, qw)
        quaternion = rotation.as_quat()  # This returns (x, y, z, w)
        
        return quaternion

    def move_spoon_to_pose(self, tip_pose, tip_to_wrist = None):

        self.robot_controller.move_to_pose(tip_pose)
    
    def pushing_skill_mujoco(self, keypoints = None):
        """
        keypoints: list of 2 pixel coordinates of the start and end points
        start: [x, y, z], pixel coordinates of the start point,
        end: [x, y, z], pixel coordinates of the end point
        """
        # if keypoints is not None:
        start, end = keypoints
        # else:
        #     clicks = self.pixel_selector.run(color_image, num_clicks=2)
        #     start = clicks[0]
        #     end = clicks[1]
        
        ## Get points in world frame using depth_image and camera_info
        # validity, end_vec_3d = utils.pixel2World(camera_info, end[0], end[1], depth_image)
        # if not validity:
        #     print("Invalid depth detected")
        #     return
        
        # validity, start_vec_3d = utils.pixel2World(camera_info, start[0], start[1], depth_image)
        # if not validity:
        #     print("Invalid depth detected")
        #     return
        
        # Get end 3d point from mujoco. Probs can go from 1 end of the bowl to the next
        start_vec_3d = np.array(start)
        end_vec_3d = np.array(end)

        # Get angle between start and end points
        push_angle = utils.angle_between_points(np.array(start), np.array(end)) 
                
        print("Executing pushing action.")

        grouping_start_pose = np.zeros((4,4))
        # I can't visualise why they want to rotate about the z axis
        grouping_start_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,push_angle], degrees=True).as_matrix()
        grouping_start_pose[:3,3] = start_vec_3d.reshape(1,3)
        grouping_start_pose[3,3] = 1

        # grouping_start_pose_quat = self.matrix_to_quaternion(grouping_start_pose)
        grouping_start_pose_quat = [0.9238795325050545, 0, 0.3826834323625085, 0]

        print("Pushing Depth: ", grouping_start_pose[2,3])
        grouping_start_pose[2,3] = max(PLATE_HEIGHT, grouping_start_pose[2,3])

        grouping_start_pose_msg = PoseStamped()
        grouping_start_pose_msg.pose.position.x = grouping_start_pose[0,3]
        grouping_start_pose_msg.pose.position.y = grouping_start_pose[1,3]
        grouping_start_pose_msg.pose.position.z = grouping_start_pose[2,3]

        grouping_start_pose_msg.pose.orientation.x = grouping_start_pose_quat[0]
        grouping_start_pose_msg.pose.orientation.y = grouping_start_pose_quat[1]
        grouping_start_pose_msg.pose.orientation.z = grouping_start_pose_quat[2]
        grouping_start_pose_msg.pose.orientation.w = grouping_start_pose_quat[3]

        grouping_end_pose = np.zeros((4,4))
        grouping_end_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,push_angle], degrees=True).as_matrix()
        grouping_end_pose[:3,3] = end_vec_3d.reshape(1,3)
        grouping_end_pose[3,3] = 1

        grouping_end_pose[2,3] = max(PLATE_HEIGHT, grouping_start_pose[2,3])

        grouping_end_pose_msg = PoseStamped()
        grouping_end_pose_msg.pose.position.x = grouping_end_pose[0,3]
        grouping_end_pose_msg.pose.position.y = grouping_end_pose[1,3]
        grouping_end_pose_msg.pose.position.z = grouping_end_pose[2,3]

        grouping_end_pose_msg.pose.orientation.x = grouping_start_pose_quat[0]
        grouping_end_pose_msg.pose.orientation.y = grouping_start_pose_quat[1]
        grouping_end_pose_msg.pose.orientation.z = grouping_start_pose_quat[2]
        grouping_end_pose_msg.pose.orientation.w = grouping_start_pose_quat[3]

        # action 1: Move to above start position
        waypoint_1 = grouping_start_pose_msg
        waypoint_1.pose.position.z += 0.05
        # self.move_utensil_to_pose(waypoint_1)
        print(f"waypoint_1: {waypoint_1}")
        self.move_spoon_to_pose(waypoint_1)


        # action 2: Move down until tip touches plate
        waypoint_2 = grouping_start_pose_msg
        print(f"waypoint_2: {waypoint_2}")
        # self.move_utensil_to_pose(waypoint_2)
        self.move_spoon_to_pose(waypoint_2)

        # action 3: Move to end position
        waypoint_3 = grouping_end_pose_msg
        print(f"waypoint_3: {waypoint_3}")
        # self.move_utensil_to_pose(waypoint_3)
        self.move_spoon_to_pose(waypoint_3)

        # action 4: Move a bite up
        waypoint_4 = grouping_end_pose_msg
        waypoint_4.pose.position.z += 0.05
        print(f"waypoint_4: {waypoint_4}")
        # self.move_utensil_to_pose(waypoint_4) 
        self.move_spoon_to_pose(waypoint_4)

        # action 5: Move to above start position
        self.robot_controller.reset()

        return