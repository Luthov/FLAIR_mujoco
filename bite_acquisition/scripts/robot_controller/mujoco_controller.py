import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
from geometry_msgs.msg import PoseStamped

from bite_acquisition.srv import PoseCommand, PoseCommandRequest, PoseCommandResponse
from bite_acquisition.srv import JointCommand, JointCommandRequest, JointCommandResponse

from .base import RobotController

try:
    from environments.arm_base import ArmBaseEnv
    from planners.moveit_planner import MoveItPlanner
    from utils.transform_utils import quat2euler, quat_distance, quat_error, quat2axisangle, quat2mat, mat2quat
    from controllers import load_controller_config
except:
    from mujoco_files.environments.arm_base import ArmBaseEnv
    from mujoco_files.planners.moveit_planner import MoveItPlanner
    from mujoco_files.utils.transform_utils import quat2euler, quat_distance, quat_error, quat2axisangle, quat2mat, mat2quat
    from mujoco_files.controllers import load_controller_config

class MujocoRobotController(RobotController):

    def __init__(self, config):

        controller_config = load_controller_config(default_controller="IK_POSE")
        controller_config["control_delta"] = False

        env_config = {
            "model_path": "/home/luthov_ubuntu/School/FYP/learning_mujoco/models/envs/mujoco_scooping_test/feeding.xml",
            # "model_path": "models/robots/xarm6/xarm6_with_ft_sensor_gripper_with_spoon.xml",
            "sim_timestep": 0.001,
            "controller_config": controller_config,
            "control_freq": 100,
            "policy_freq": 25,
            "render_mode": "human",
            "obs_mode": None,
        }

        controller_config["control_freq"] = env_config["control_freq"]
        controller_config["policy_freq"] = env_config["policy_freq"]

        self.env = ArmBaseEnv(
            model_path = env_config["model_path"],
            sim_timestep = env_config["sim_timestep"],
            controller_config = controller_config,
            control_freq = env_config["control_freq"],
            policy_freq = env_config["policy_freq"],
            render_mode = env_config["render_mode"],
            camera = -1,    # Default global camera
            obs_mode = env_config["obs_mode"],
        )

        self.planner = MoveItPlanner(dt = 1/self.env._policy_freq)

    def reset(self):
        # TODO: Probs can look at 
        self.move_to_acq_pose()

    def move_to_pose(self, pose):

        target_pos = pose[:3, 3].reshape(3)
        target_quat = R.from_matrix(pose[:3,:3]).as_quat()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "world"
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.pose.position.x = target_pos[0]
        goal_pose.pose.position.y = target_pos[1]
        goal_pose.pose.position.z = target_pos[2]
        goal_pose.pose.orientation.x = target_quat[0]
        goal_pose.pose.orientation.y = target_quat[1]
        goal_pose.pose.orientation.z = target_quat[2]
        goal_pose.pose.orientation.w = target_quat[3]

        msg_trajectory = self.planner.get_motion_plan(
            start_joint_position=start_joint_position,
            goal=goal_point_chicken,
            goal_type="pose_goal",
            velocity_scaling_factor=0.1
        )
        

    def move_to_acq_pose(self):
        pass

    def move_to_transfer_pose(self):
        pass