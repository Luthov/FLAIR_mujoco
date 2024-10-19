#! /usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
import actionlib
from geometry_msgs.msg import PoseStamped
from bite_acquisition.msg import mujoco_action_serverAction, mujoco_action_serverResult, mujoco_action_serverFeedback

try:
    from environments.arm_base import ArmBaseEnv
    from planners.moveit_planner import MoveItPlanner
    from utils.transform_utils import quat2euler, quat_distance, quat_error, quat2axisangle, quat2mat, mat2quat
    from controllers import load_controller_config
except:
    from feeding_mujoco.environments.arm_base import ArmBaseEnv
    from feeding_mujoco.planners.moveit_planner import MoveItPlanner
    from feeding_mujoco.utils.transform_utils import quat2euler, quat_distance, quat_error, quat2axisangle, quat2mat, mat2quat
    from feeding_mujoco.controllers import load_controller_config

class MujocoAction(object):

    def __init__(self):

        self.result = mujoco_action_serverResult()

        controller_config = load_controller_config(default_controller="IK_POSE")
        controller_config["control_delta"] = False

        env_config = {
            "model_path": "/home/luthov_ubuntu/School/FYP/learning_mujoco/models/envs/mujoco_scooping_test/feeding.xml",
            "sim_timestep": 0.002,
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
        self.env._robot.hard_set_joint_positions([1.3103,  0.1289, -1.2169, -0.9229,  1.6615,  1.6824], self.env._robot._arm_joint_ids)
        
        # Step the simulator to update the robot and environment state
        self.env.sim.step()

        self.acq_pos = np.radians([0.0, -65.0, -25.0, 0.0, 65.0, -90.0])
        self.transfer_pos = np.radians([0.0, -65.0, -25.0, 0.0, 0.0, -90.0])

        self.action_name = "mujoco_action_server"
        self.action_server = actionlib.SimpleActionServer(self.action_name, mujoco_action_serverAction, execute_cb=self.execute_callback, auto_start = False)
        self.action_server.start()
     
    def execute_callback(self, goal):

        try:
            if goal.function_name == "move_to_pose":
                self.move_to_pose(goal.goal_point)
            elif goal.function_name == "move_to_acq_pose":
                self.move_to_acq_pose()
            elif goal.function_name == "move_to_transfer_pose":
                self.move_to_transfer_pose()
            elif goal.function_name == "reset":
                self.reset()
            else:
                rospy.logerr("Unknown command: %s", goal.function_name)
                self.action_server.set_aborted(self.result, "Unknown command")
                return

            self.result.success = True
            self.action_server.set_succeeded(self.result)
        except Exception as e:
            rospy.logerr("Exception in execute_cb: %s", str(e))
            self.result.success = False
            self.action_server.set_aborted(self.result, str(e))

    def execute_trajectory(self, msg_trajectory):

        # Visualize the trajectory in RViz
        self.planner._visualize_trajectory(msg_trajectory)

        # Set the start state of the controller
        self.env._robot._controller.set_start()

        # Execute the trajectory
        while True:
            # Step the motion planner to get next waypoint
            joint_position, final_wp = self.planner.step()
            y = self.env._robot.get_eef_fk(joint_position)
            
            policy_step = True

            # Control loop 
            for i in range(int((1/self.env._control_freq)/self.env._sim_timestep)):
                # Forward sim
                self.env._sim.forward()
                    
                # Get the control action for the controller (ie next eef pose for IK_POSE controller)
                control_action_pos = y[:3] 
                control_action_orn = quat2axisangle(y[3:])
                control_action = np.concatenate([control_action_pos, control_action_orn])
                self.env._robot.control(control_action, policy_step=policy_step)  

                # Step simulator
                self.env._sim.step()
                policy_step = False       
            
            if final_wp:
                # curent sim step
                sim_time = self.env._sim.time

                # Get error between the current pose and desired pose
                pos_error = y[:3] - self.env._robot.get_eef_pose()[:3]
                pos_error_norm = np.linalg.norm(pos_error)
                orn_error = np.linalg.norm(quat_error(self.env._robot.get_eef_pose()[3:], y[3:]))
                pos_error_threshold = 0.001    # 2 cm error  
                quat_error_threshold = 0.1

                # Control loop to correct the error
                while pos_error_norm > pos_error_threshold or orn_error > quat_error_threshold:
                    # Compute the error between the current pose and desired pose
                    control_action_pos = self.env._robot.get_eef_pose()[:3] + 1.0 * pos_error
                    control_action_orn = quat2axisangle(y[3:])
                    control_action = np.concatenate([control_action_pos, control_action_orn])
                    self.env._robot.control(control_action, policy_step=True)
                    self.env._sim.step()

                    # Recalculate the error
                    pos_error = y[:3] - self.env._robot.get_eef_pose()[:3]
                    pos_error_norm = np.linalg.norm(pos_error)                        
                    orn_error = np.linalg.norm(quat_error(self.env._robot.get_eef_pose()[3:], y[3:]))
                    # print("pos_error:", pos_error)
                    # print("orn_error:", orn_error)

                    # Break if the error is not reducing
                    if self.env._sim.time - sim_time > 2.0:
                        print("Failed to reach goal pose. Exiting...")
                        break

                correction_time = self.env._sim.time - sim_time
                # print("Correction time:", correction_time, "steps:", int(correction_time/env._sim_timestep))

                print("Final joint pose:", joint_position)
                print("final eef pose:", self.env._robot.get_eef_pose())
                break
                # print("Moved to pose successfuly!")
                # break

    def set_joint_position(self, joint_position):

        # Reset the planner
        self.planner.reset()

        # Step the simulator to update the robot and environment state
        self.env.sim.step()

        # Get the current joint positions
        start_joint_position = self.env._robot._joint_positions

        msg_trajectory = self.planner.get_motion_plan(
            start_joint_position=start_joint_position,
            goal=joint_position,
            goal_type="joint_goal",
            velocity_scaling_factor=0.1
        )

        if msg_trajectory is not None:
            self.execute_trajectory(msg_trajectory)

    def move_to_pose(self, pose):

        print(pose)
        # Reset the planner
        self.planner.reset()

        # Step the simulator to update the robot and environment state
        self.env.sim.step()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "world"
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.pose.position.x = pose.pose.position.x
        goal_pose.pose.position.y = pose.pose.position.y
        goal_pose.pose.position.z = pose.pose.position.z
        goal_pose.pose.orientation.x = pose.pose.orientation.x
        goal_pose.pose.orientation.y = pose.pose.orientation.y
        goal_pose.pose.orientation.z = pose.pose.orientation.z
        goal_pose.pose.orientation.w = pose.pose.orientation.w

        # Get the current joint positions
        start_joint_position = self.env._robot._joint_positions

        msg_trajectory = self.planner.get_motion_plan(
            start_joint_position=start_joint_position,
            goal=goal_pose,
            goal_type="pose_goal",
            velocity_scaling_factor=0.1
        )

        if msg_trajectory is not None:
            self.execute_trajectory(msg_trajectory)
                
    def move_to_acq_pose(self):
        self.set_joint_position(self.acq_pos)

    def move_to_transfer_pose(self):
        self.set_joint_position(self.transfer_pos)

    def reset(self):
        self.move_to_acq_pose()

if __name__ == '__main__':
    rospy.init_node('mujoco_action_server', anonymous=True)
    mujoco_action_server = MujocoAction()

    rospy.spin()