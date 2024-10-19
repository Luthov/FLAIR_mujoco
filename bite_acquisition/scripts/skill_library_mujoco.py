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
import tf2_geometry_msgs
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

        # Create a TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.print_waypoints = True

        print("Skill library initialized")

    def transform_orientation(self, euler_angles):

        pose_world = None

        # Specify the axis sequence, for example, 'xyz'
        rotation_sequence = 'xyz'

        # Create a rotation object from Euler angles
        rotation = Rotation.from_euler(rotation_sequence, euler_angles, degrees=True)

        # Convert to quaternion
        quaternion = rotation.as_quat()

        pose_link_tcp = tf2_geometry_msgs.PoseStamped()
        pose_link_tcp.header.frame_id = "link_tcp"
        pose_link_tcp.header.stamp = rospy.Time.now()
        pose_link_tcp.pose.position.x = 0.0
        pose_link_tcp.pose.position.y = 0.0
        pose_link_tcp.pose.position.z = 0.0
        pose_link_tcp.pose.orientation.x = quaternion[0]
        pose_link_tcp.pose.orientation.y = quaternion[1]
        pose_link_tcp.pose.orientation.z = quaternion[2]
        pose_link_tcp.pose.orientation.w = quaternion[3] 

        while pose_world is None:
            try:
                # Look up the transform from 'link_tcp' to 'world'
                transform = self.tf_buffer.lookup_transform('world', 'link_tcp', rospy.Time(0), rospy.Duration(1.0))

                # Transform the pose
                pose_world = tf2_geometry_msgs.do_transform_pose(pose_link_tcp, transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Transform lookup failed!")
   
        pose_world_quat = [pose_world.pose.orientation.x, pose_world.pose.orientation.y, pose_world.pose.orientation.z, pose_world.pose.orientation.w]
        return pose_world_quat
    
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
        
        # Get end 3d point from mujoco. Probs can go from 1 end of the bowl to the next
        start_vec_3d = np.array(start)
        end_vec_3d = np.array(end)

        # Get angle between start and end points
        push_angle = utils.angle_between_points(np.array(start), np.array(end)) 
        print(f"Push angle: {push_angle}")
        print("Executing pushing action.")

        grouping_start_pose = np.zeros((4,4))
        grouping_start_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,push_angle], degrees=True).as_matrix()
        grouping_start_pose[:3,3] = start_vec_3d.reshape(1,3)
        grouping_start_pose[3,3] = 1

        # I think need to get the orientation in terms of link_tcp from world frame

        # grouping_start_pose_quat = self.matrix_to_quaternion(grouping_start_pose)
        grouping_start_pose_quat = self.transform_orientation([0, 0, push_angle])
        print("quat_start: ", grouping_start_pose_quat)

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
        if self.print_waypoints:
            print(f"waypoint_1: {waypoint_1}")
        self.move_spoon_to_pose(waypoint_1)

        waypoint_2 = grouping_start_pose_msg
        waypoint_2.pose.position.z -= 0.05
        if self.print_waypoints:
            print(f"waypoint_2: {waypoint_2}")
        self.move_spoon_to_pose(waypoint_2)

        waypoint_3 = grouping_end_pose_msg
        if self.print_waypoints:
            print(f"waypoint_3: {waypoint_3}")
        self.move_spoon_to_pose(waypoint_3)

        waypoint_4 = grouping_end_pose_msg
        waypoint_4.pose.position.z += 0.05
        if self.print_waypoints:
            print(f"waypoint_4: {waypoint_4}")
        self.move_spoon_to_pose(waypoint_4)

        # action 5: Move to above start position
        self.robot_controller.reset()

        return