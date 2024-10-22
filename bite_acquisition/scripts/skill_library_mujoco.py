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
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
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

PLATE_HEIGHT = 0.15
class SkillLibrary:

    def __init__(self):

        self.robot_controller = MujocoRobotController()

        # Create a TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.print_waypoints = True

        print("Skill library initialized")

    def transform_pose(self, pose, source_frame, target_frame):

        # transformed_pose = None

        # while transformed_pose is None:
        #     try:
        #         # Look up the transform from 'source_frame' to 'target_frame'
        #         transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))

        #         # Transform the pose
        #         transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
        #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #         rospy.logerr("Transform lookup failed!")

        # return transformed_pose

            # Wait for the transform to be available
        try:
            self.tf_buffer.can_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(3.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Transform error: {e}")
            return None

        # Transform the pose
        try:
            transformed_pose = tf2_geometry_msgs.do_transform_pose(
                pose,
                self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
            )
            return transformed_pose
        except Exception as e:
            rospy.logerr(f"Error transforming pose: {e}")
            return None

    def transform_orientation(self, euler_angles):

        pose_world = None

        # Specify the axis sequence, for example, 'xyz'
        rotation_sequence = 'xyz'

        # Create a rotation object from Euler angles
        rotation = Rotation.from_euler(rotation_sequence, euler_angles, degrees=True)

        # Convert to quaternion
        quaternion = rotation.as_quat()

        pose_spoon_frame = tf2_geometry_msgs.PoseStamped()
        # pose_spoon_frame.header.frame_id = "spoon_frame"
        pose_spoon_frame.header.frame_id = "link_tcp"
        pose_spoon_frame.header.stamp = rospy.Time.now()
        pose_spoon_frame.pose.position.x = 0.0
        pose_spoon_frame.pose.position.y = 0.0
        pose_spoon_frame.pose.position.z = 0.14
        pose_spoon_frame.pose.orientation.x = quaternion[0]
        pose_spoon_frame.pose.orientation.y = quaternion[1]
        pose_spoon_frame.pose.orientation.z = quaternion[2]
        pose_spoon_frame.pose.orientation.w = quaternion[3] 

        while pose_world is None:
            try:
                # Look up the transform from 'link_tcp' to 'world'
                transform = self.tf_buffer.lookup_transform('world', 'link_tcp', rospy.Time(0), rospy.Duration(1.0))

                # Transform the pose
                pose_world = tf2_geometry_msgs.do_transform_pose(pose_spoon_frame, transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Transform lookup failed!")
   
        pose_world_quat = [pose_world.pose.orientation.x, pose_world.pose.orientation.y, pose_world.pose.orientation.z, pose_world.pose.orientation.w, pose_world.pose.position.x, pose_world.pose.position.y, pose_world.pose.position.z]
        return pose_world_quat
    
    def get_target_eef_pose_in_world(self, pose_position, pose_quat):

        target_position_spoon = None
        target_position_link_tcp = None
        final_target_position_world = None

        target_position_world = tf2_geometry_msgs.PoseStamped()
        target_position_world.header.frame_id = "world"
        target_position_world.header.stamp = rospy.Time.now()
        target_position_world.pose.position.x = pose_position[0]
        target_position_world.pose.position.y = pose_position[1]
        target_position_world.pose.position.z = pose_position[2]

        while target_position_spoon is None:
            try:
                # Look up the transform from 'world' to 'spoon_frame
                transform = self.tf_buffer.lookup_transform('spoon_frame', 'world', rospy.Time(0), rospy.Duration(1.0))
                # print(f"world to spoon transform: {transform}")

                # Transform the pose
                target_position_spoon = tf2_geometry_msgs.do_transform_pose(target_position_world, transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Transform lookup failed!")

        target_position_spoon.pose.orientation.x = pose_quat[0]
        target_position_spoon.pose.orientation.y = pose_quat[1]
        target_position_spoon.pose.orientation.z = pose_quat[2]
        target_position_spoon.pose.orientation.w = pose_quat[3]

        while target_position_link_tcp is None:
            try:
                # Look up the transform from 'spoon_frame' to 'link_tcp'
                transform = self.tf_buffer.lookup_transform('link_tcp', 'spoon_frame', rospy.Time(0), rospy.Duration(1.0))
                # print(f"spoon to link_tcp transform: {transform}")

                # Transform the pose
                target_position_link_tcp = tf2_geometry_msgs.do_transform_pose(target_position_spoon, transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Transform lookup failed!")

        while final_target_position_world is None:
            try:
                # Look up the transform from 'link_tcp' to world
                transform = self.tf_buffer.lookup_transform('world', 'link_tcp', rospy.Time(0), rospy.Duration(1.0))
                # print(f"link_tcp to world transform: {transform}")

                # Transform the pose
                final_target_position_world = tf2_geometry_msgs.do_transform_pose(target_position_spoon, transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Transform lookup failed!")

        # print(f"target_position_world: {target_position_world}")
        # print(f"target_position_spoon: {target_position_spoon}")
        # print(f"target_position_link_tcp: {target_position_link_tcp}")
        # print(f"final_target_position_world: {final_target_position_world}")

        return final_target_position_world
    
    def get_target_eef_pose_in_world_chat(self, pose_position, pose_quat):

        T_world_spoon_orientation = None
        T_world_eef = None

        target_position_spoon = tf2_geometry_msgs.PoseStamped()
        target_position_spoon.pose.orientation.x = pose_quat[0]
        target_position_spoon.pose.orientation.y = pose_quat[1]
        target_position_spoon.pose.orientation.z = pose_quat[2]
        target_position_spoon.pose.orientation.w = pose_quat[3]
            
        while T_world_spoon_orientation is None:
            try:
                # Look up the transform from 'spoon_frame' to 'world'
                transform = self.tf_buffer.lookup_transform('world', 'spoon_frame', rospy.Time(0), rospy.Duration(1.0))
                print(f"world to spoon transform: {transform}")

                # Transform the pose
                T_world_spoon_orientation = tf2_geometry_msgs.do_transform_pose(target_position_spoon, transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Transform lookup failed!")

        while T_world_eef is None:

            try:
                # Step 1: Look up the static transform from 'eef' to 'spoon'
                transform_eef_spoon = self.tf_buffer.lookup_transform('link_tcp', 'spoon_frame', rospy.Time(0), rospy.Duration(1.0))
                
                # Step 2: Inverse the transform from 'eef' to 'spoon' to get 'spoon' to 'eef'
                T_eef_spoon = tf2_geometry_msgs.transform_to_kdl(transform_eef_spoon)
                T_spoon_eef = T_eef_spoon.Inverse()  # Invert the transform

                # Step 3: Obtain the desired pose of the spoon in the world frame (target pose)
                # Assume T_world_spoon is the target pose of the spoon in the world frame
                T_world_spoon = tf2_geometry_msgs.PoseStamped()
                # Assign the target translation and rotation to T_world_spoon (example values)
                T_world_spoon.pose.position.x = pose_position[0]
                T_world_spoon.pose.position.y = pose_position[1]
                T_world_spoon.pose.position.z = pose_position[2]
                T_world_spoon.pose.orientation.x = T_world_spoon_orientation.pose.orientation.x
                T_world_spoon.pose.orientation.y = T_world_spoon_orientation.pose.orientation.y
                T_world_spoon.pose.orientation.z = T_world_spoon_orientation.pose.orientation.z
                T_world_spoon.pose.orientation.w = T_world_spoon_orientation.pose.orientation.w

                # Step 4: Multiply the world->spoon transform by the spoon->eef transform
                T_world_eef = tf2_geometry_msgs.do_transform_pose(T_world_spoon, transform_eef_spoon)

                # Now T_world_eef contains the desired pose of the EEF in the world frame
                rospy.loginfo("Desired EEF Pose in World Frame: \n{}".format(T_world_eef))

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
                rospy.logwarn("Transform not available: {}".format(ex))

        return T_world_eef


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
    
    def cutting_skill(self, color_image, depth_image, camera_info, keypoint = None, cutting_angle = None):

        if keypoint is not None:
            (center_x, center_y) = keypoint

            # shift cutting point perpendicular to the cutting angle in the direction of the fork tines (less y value)
            pt = cmath.rect(23, np.pi/2-cutting_angle)
            center_x = center_x + int(pt.real)
            center_y = center_y - int(pt.imag)
            # cv2.line(color_image_vis, (center_x-x2,center_y+y2), (cut_point[0]+x2,cut_point[1]-y2), (255,0,0), 2)

            cutting_angle = math.degrees(cutting_angle)
            cutting_angle = cutting_angle + 180 # Rajat ToDo - remove this hack bruh
        else:
            clicks = self.pixel_selector.run(color_image, num_clicks=2)
            (left_x, left_y) = clicks[0]
            (right_x, right_y) = clicks[1]
            print("Left: ", left_x, left_y)
            print("Right: ", right_x, right_y)
            if left_y < right_y:
                center_x, center_y = left_x, left_y
                clicks[0], clicks[1] = clicks[1], clicks[0]
            else:
                center_x, center_y = right_x, right_y
            cutting_angle = utils.angle_between_pixels(np.array(clicks[0]), np.array(clicks[1]), color_image.shape[1], color_image.shape[0], orientation_symmetry = False)

        # visualize cutting point and line between left and right points
        cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
        # cv2.line(color_image, (left_x, left_y), (right_x, right_y), (0, 0, 255), 2)

        cv2.imshow('vis', color_image)
        cv2.waitKey(0)

        # get 3D point from depth image
        validity, point = utils.pixel2World(camera_info, center_x, center_y, depth_image)

        if not validity:
            print("Invalid point")
            return
        
        fork_rotation = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]
        
        # action 1: Set wrist state to cutting angle
        self.wrist_controller.set_to_cut_pos()
        self.wrist_controller.set_to_cut_pos()
        
        fork_rotation_cut = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]
        wrist_rotation = np.linalg.inv(fork_rotation) @ fork_rotation_cut

        print('Cutting angle: ', cutting_angle)
        # update cutting angle to take into account incline of fork tines
        cutting_angle = cutting_angle + 25

        cutting_pose = np.zeros((4,4))
        cutting_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,cutting_angle], degrees=True).as_matrix() @ wrist_rotation
        cutting_pose[:3,3] = point.reshape(1,3)
        cutting_pose[3,3] = 1

        cutting_pose = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ cutting_pose

        cutting_pose[2,3] = max(cutting_pose[2,3], PLATE_HEIGHT)

        self.visualizer.visualize_food(cutting_pose)

        waypoint_1_tip = np.copy(cutting_pose)
        waypoint_1_tip[2,3] += 0.03 

        self.move_utensil_to_pose(waypoint_1_tip)

        # action 2: Move down until tip touches the plate

        waypoint_2_tip = np.copy(cutting_pose)
        waypoint_2_tip[2,3] = PLATE_HEIGHT - 0.009
        self.move_utensil_to_pose(waypoint_2_tip)

        tip_to_wrist = self.tf_utils.getTransformationFromTF('fork_tip', 'tool_frame')

        # action 2.5: slightly turn the fork tines so that the food item flips over / separates
        self.wrist_controller.cutting_tilt()

        # action 3: Push orthogonal to the cutting angle, in direction of towards the robot (+y relative to the fork)
        waypoint_3_tip = np.copy(cutting_pose)
        waypoint_3_tip[2,3] = PLATE_HEIGHT - 0.009
        y_displacement = np.eye(4)
        y_displacement[1,3] = 0.02
        waypoint_3_tip = waypoint_3_tip @ y_displacement
        self.move_utensil_to_pose(waypoint_3_tip, tip_to_wrist)

        ## action 3: Move up
        
        waypoint_4_tip = np.copy(waypoint_3_tip)
        waypoint_4_tip[2,3] += 0.035 
        
        self.move_utensil_to_pose(waypoint_4_tip, tip_to_wrist)

    def cutting_skill_mujoco(self, keypoint = None, cutting_angle = None):

        self.robot_controller.move_to_acq_pose()

        (center_x, center_y, top_z) = keypoint

        # shift cutting point perpendicular to the cutting angle in the direction of the fork tines (less y value)
        # pt = cmath.rect(23, np.pi/2-cutting_angle)
        # center_x = center_x + int(pt.real)
        # center_y = center_y - int(pt.imag)
        # cv2.line(color_image_vis, (center_x-x2,center_y+y2), (cut_point[0]+x2,cut_point[1]-y2), (255,0,0), 2)

        cutting_angle = math.degrees(cutting_angle)
        # cutting_angle = cutting_angle + 180 # Rajat ToDo - remove this hack bruh
        print(f"Cutting angle: {cutting_angle}")
        
        # action 1: Set wrist state to cutting angle

        point = np.array([center_x, center_y, top_z])

        cutting_pose = np.zeros((4,4))
        cutting_pose[:3,:3] = Rotation.from_euler('xyz', [0,90,cutting_angle], degrees=True).as_matrix()
        cutting_pose[:3,3] = point.reshape(1,3)
        cutting_pose[3,3] = 1

        # cutting_pose_quat = self.transform_orientation([0, 90, cutting_angle])
        # print("quat_start: ", cutting_pose_quat)

        cutting_pose_quat_spoon_frame = Rotation.from_euler('xyz', [0,90,cutting_angle], degrees=True).as_quat()

        cutting_pose[2,3] = max(cutting_pose[2,3], PLATE_HEIGHT)

        # cutting_pose_msg = tf2_geometry_msgs.PoseStamped()
        # cutting_pose_msg.header.frame_id = "world"
        # cutting_pose_msg.header.stamp = rospy.Time.now()
        # cutting_pose_msg.pose.position.x = cutting_pose[0,3]
        # cutting_pose_msg.pose.position.y = cutting_pose[1,3]
        # cutting_pose_msg.pose.position.z = cutting_pose[2,3]

        # cutting_pose_msg.pose.orientation.x = cutting_pose_quat[0]
        # cutting_pose_msg.pose.orientation.y = cutting_pose_quat[1]
        # cutting_pose_msg.pose.orientation.z = cutting_pose_quat[2]
        # cutting_pose_msg.pose.orientation.w = cutting_pose_quat[3]

        cutting_pose_msg = self.get_target_eef_pose_in_world([cutting_pose[0,3], cutting_pose[1,3], cutting_pose[2,3]], cutting_pose_quat_spoon_frame)
        # cutting_pose_msg = self.get_target_eef_pose_in_world_chat([cutting_pose[0,3], cutting_pose[1,3], cutting_pose[2,3]], cutting_pose_quat_spoon_frame)

        waypoint_1_tip = cutting_pose_msg
        waypoint_1_tip.pose.position.z += 0.03 

        if self.print_waypoints:
            print(f"waypoint_1: {waypoint_1_tip}")
        self.move_spoon_to_pose(waypoint_1_tip)

        # action 2: Move down until tip touches the plate

        waypoint_2_tip = cutting_pose_msg
        waypoint_2_tip.pose.position.z = PLATE_HEIGHT - 0.009
        if self.print_waypoints:
            print(f"waypoint_2: {waypoint_2_tip}")
        self.move_spoon_to_pose(waypoint_2_tip)

        # action 2.5: slightly turn the fork tines so that the food item flips over / separates

        self.robot_controller.rotate_eef(-0.39269908169)

        # # action 3: Push orthogonal to the cutting angle, in direction of towards the robot (+y relative to the fork)
        waypoint_3_tip = cutting_pose_msg
        waypoint_3_tip.pose.position.z = PLATE_HEIGHT - 0.009
        print(f"waypoint_3: {waypoint_3_tip}")
        waypoint_3_tip = self.transform_pose(waypoint_3_tip, "world", "link_tcp")
        print(f"waypoint_3_2: {waypoint_3_tip}")
        waypoint_3_tip.pose.position.y += 0.02
        print(f"waypoint_3_3: {waypoint_3_tip}")
        waypoint_3_tip = self.transform_pose(waypoint_3_tip, "link_tcp", "world")
        print(f"waypoint_3_4: {waypoint_3_tip}")
        if self.print_waypoints:
            print(f"waypoint_3: {waypoint_3_tip}")
        self.move_spoon_to_pose(waypoint_3_tip)

        ## action 3: Move up
        
        waypoint_4_tip = waypoint_3_tip
        waypoint_4_tip.pose.position.z += 0.035 
        if self.print_waypoints:
            print(f"waypoint_4: {waypoint_4_tip}")
        self.move_spoon_to_pose(waypoint_4_tip)
