#!/usr/bin/env python
import time
import numpy as np

import rospy
from feeding_msgs.srv import GetMotionPlan, GetMotionPlanRequest
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.msg import RobotState, Constraints, PositionConstraint, OrientationConstraint
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState


class MoveItPlanner:
    def __init__(self, dt=0.01):
        self._planned_path_pub = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=10)

        self._dt = dt
        self._motion_plan_done = False
        self._curr_robot_trajectory = None
        self._curr_waypoint_index = 0

        self._curr_joint_positions = []
        self._curr_time_from_start = []

    def step(self):
        """
        Steps the MoveItPlanner and returns the next state of the robot

        Returns:
            joint_pose (np.ndarray): Numpy array of the next joint positions
            done (bool): Boolean indicating if the motion plan is done
        """
        if len(self._curr_joint_positions) == 0:
            rospy.logwarn("No trajectory found. Please call get_motion_plan() first.")
            return None, None, True

        # Increment the waypoint index
        self._curr_waypoint_index += 1
        
        done = False
        if (self._curr_waypoint_index + 1) >= len(self._curr_joint_positions):
            done = True
            self._motion_plan_done = True
        
        # Get the next waypoint
        joint_pose = self._curr_joint_positions[self._curr_waypoint_index]

        return joint_pose, done
    
    def reset(self):
        """
        Resets the MoveItPlanner
        """
        self._curr_robot_trajectory = None
        self._curr_waypoint_index = 0
        self._curr_joint_positions = []
        self._curr_time_from_start = []
        self._motion_plan_done = False

    def get_motion_plan(self, start_joint_position, goal, goal_type, velocity_scaling_factor):
        """
        Gets a motion plan from from MoveIt.

        Args:
            start_joint_pose ([list]): Joint positions of the robot starting state
            goal (Pose or JointState or str): Goal to move to.
            goal_type (str): Type of goal. Can be "pose_goal", "joint_goal", or "named_goal".
            velocity_scaling_factor (float, optional): Velocity scaling factor. Defaults to 1.0.

        Returns:
            bool or RobotTrajectory: If success, returns RobotTrajectory. Otherwise, returns None
        """
        rospy.wait_for_service('feeding_xarm/get_motion_plan', timeout=5)
        try:
            # Create a service proxy for the GetMotionPlan service
            get_motion_plan_srv = rospy.ServiceProxy('feeding_xarm/get_motion_plan', GetMotionPlan)
            
            # Create a request
            req_get_motion_plan = GetMotionPlanRequest()

            # Set the start state (e.g., current robot state)
            req_get_motion_plan.start_joint_position = start_joint_position
            req_get_motion_plan.velocity_scaling_factor = velocity_scaling_factor
            rospy.loginfo(f"Start joint position: {start_joint_position}")

            # Set the goal
            req_get_motion_plan.goal_type = goal_type
            
            if goal_type == "joint_goal":
                msg_joint_goal = JointState()
                msg_joint_goal.position = goal
                req_get_motion_plan.goal_joint_state = msg_joint_goal
            
            elif goal_type == "pose_goal":
                req_get_motion_plan.goal_pose = goal

            elif goal_type == "named_goal":
                req_get_motion_plan.named_goal = goal

            # Call the service
            response = get_motion_plan_srv(req_get_motion_plan)

            # Handle the response
            if response.success:
                rospy.loginfo("Motion plan found!")
                self._curr_robot_trajectory = response.trajectory
                self._save_joint_trajectory(response.trajectory)
                self._interpolate_trajectory(time_scale_factor=1.0)
                return response.trajectory      # RobotTrajectory
            else:
                rospy.logerr("Failed to find a valid motion plan.")
                return None

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return None
    
    def _interpolate_trajectory(self, time_scale_factor=1.0):
        """
        Interpolate the trajectory based on the time step `dt`.
        """
        interpolated_positions = []

        # Original trajectory points and times
        original_points = self._curr_robot_trajectory.joint_trajectory.points
        original_times = [point.time_from_start.to_sec() for point in original_points]

        # Scale the time
        scaled_times = [time_scale_factor * time for time in original_times]
        
        # Time at which each interpolated point should be placed
        total_time = scaled_times[-1]
        interpolated_times = np.arange(0, total_time, self._dt)

        # Interpolate for each joint
        for joint_idx in range(len(original_points[0].positions)):
            original_positions = [point.positions[joint_idx] for point in original_points]

            # Interpolating positions
            interp_positions = np.interp(interpolated_times, scaled_times, original_positions)
            interpolated_positions.append(interp_positions)

        # Transpose the list of interpolated positions and velocities for each joint
        self._curr_joint_positions = list(map(list, zip(*interpolated_positions)))
        self._curr_joint_velocities = []

        rospy.loginfo(f"Interpolated {len(self._curr_joint_positions)} waypoints based on dt={self._dt:.4f}s.")

    
    def _save_joint_trajectory(self, trajectory):
        """
        Saves joint positions and velocities internally from the RobotTrajectory.

        Args:
            trajectory (RobotTrajectory): The planned trajectory
        """
        self._curr_joint_positions = []
        self._curr_time_from_start = []

        for point in trajectory.joint_trajectory.points:
            self._curr_joint_positions.append(point.positions)
            self._curr_time_from_start.append(point.time_from_start.to_sec())


    def _visualize_trajectory(self, trajectory):

        # Create a DisplayTrajectory message
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = RobotState()  # Optionally fill with start state
        display_trajectory.trajectory.append(trajectory)

        # Publish the trajectory to RViz
        self._planned_path_pub.publish(display_trajectory)

if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node('moveit_planner_sim')

    mip = MoveItPlanner()

    mip.get_motion_plan(start_joint_position=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        goal="reset",
                        goal_type="named_goal",
                        velocity_scaling_factor=0.1)


        
