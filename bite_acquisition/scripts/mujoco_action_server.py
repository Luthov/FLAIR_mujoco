#! /usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation as R
import time

import rospy
import rospkg
import actionlib
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from moveit_msgs.msg import RobotTrajectory, MoveItErrorCodes
from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal

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

        model_folder = rospkg.RosPack().get_path("feeding_mujoco") + "/src/feeding_mujoco/models"
        env_config = {
            "model_path": model_folder + "/envs/feeding_luke/feeding.xml",
            # "model_path": model_folder + "/robots/xarm6/xarm6_with_ft_sensor_gripper_with_spoon.xml",
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

        # Step the simulator to update the robot and environment state
        self.env.sim.step()

        # Sync moveit robot state with mujoco robot state (must be done after step)
        self._action_name_arm_exec = "/execute_trajectory"
        initial_eef_pose = self.env._robot.get_eef_pose()
        msg_initial_eef_pose = PoseStamped()
        msg_initial_eef_pose.header.frame_id = "world"
        msg_initial_eef_pose.pose.position.x = initial_eef_pose[0]
        msg_initial_eef_pose.pose.position.y = initial_eef_pose[1]
        msg_initial_eef_pose.pose.position.z = initial_eef_pose[2]
        msg_initial_eef_pose.pose.orientation.w = initial_eef_pose[3]
        msg_initial_eef_pose.pose.orientation.x = initial_eef_pose[4]
        msg_initial_eef_pose.pose.orientation.y = initial_eef_pose[5]
        msg_initial_eef_pose.pose.orientation.z = initial_eef_pose[6]

        msg_joint_traj = self.planner.get_motion_plan(start_joint_position=[0, 0, 0, 0, 0, 0], 
                                                      goal=msg_initial_eef_pose, 
                                                      goal_type="pose_goal", 
                                                      velocity_scaling_factor=1.0)
        # self.execute_traj_in_moveit(msg_joint_traj)
        rospy.loginfo(f"Moved moveit robot to initial pose: {self.env._robot.get_eef_pose()}")
        
        self.reset_pos = np.array([0.0, 0.0, 0.0, 0.0, -1.3988, 0.0])   # initial pose of the robot
        # TODO: Luke: these poses should be defined by the client
        self.acq_pos = np.radians([0.0, -65.0, -25.0, 0.0, 65.0, -90.0])
        # TODO: Luke: this transfer pose also seems a bit high (in terms of z position)
        self.transfer_pos = np.radians([0.0, -65.0, -25.0, 0.0, 20.0, 0.0])

        self.action_name = "mujoco_action_server"
        self.action_server = actionlib.SimpleActionServer(self.action_name, mujoco_action_serverAction, execute_cb=self.execute_callback, auto_start = False)
        self.action_server.start()

     
    def execute_callback(self, goal):

        try:
            if goal.function_name == "move_to_pose":
                print("Moving to pose")
                self.move_to_pose(goal.goal_point)
            elif goal.function_name == "move_to_reset_pos":
                self.move_to_reset_pose()
            elif goal.function_name == "move_to_acq_pose":
                self.move_to_acq_pose()
            elif goal.function_name == "move_to_transfer_pose":
                self.move_to_transfer_pose()
            elif goal.function_name == "execute_scooping":
                self.execute_scooping(goal.scooping_trajectory, goal.goal_point)
            elif goal.function_name == "reset":
                self.reset()
            elif goal.function_name == "rotate_eef":
                self.rotate_eef(goal.angle)
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

    def execute_scooping(self, scooping_trajectory, food_pose):
        """
        Executes the scooping trajectory using the controller in Mujoco.
        Note: does not use MoveIt for trajectory execution.

        Args:
            scooping_trajectory (Float32MultiArray): The scooping trajectory to execute.
            food_pose (PoseStamped): The pose of the food item. Only positions are used.
        """
        print("Executing scooping trajectory...")
        assert scooping_trajectory.layout.dim[1].size == 7, "Invalid scooping trajectory format. Expected 7 elements per waypoint."
        traj_len = scooping_trajectory.layout.dim[0].size
        
        traj = scooping_trajectory.data
        # Reshape the trajectory
        traj = np.array(traj).reshape(traj_len, 7)

        # Move to first waypoint
        print("first wp traj[0]:", traj[0])
        print("SCOOPING TRAJ LENGTH:", traj_len)
        msg_pose = PoseStamped()
        msg_pose.header.frame_id = "world"
        msg_pose.pose.position.x = traj[0][0]
        msg_pose.pose.position.y = traj[0][1]
        msg_pose.pose.position.z = traj[0][2]
        # Note: somehow cannot plan to the DMP orientation 
        # TODO: JN: check if DMP error or robot workspace limits
        msg_pose.pose.orientation.w = -0.1638 #traj[0][3]
        msg_pose.pose.orientation.x = 0.5587 #traj[0][4]
        msg_pose.pose.orientation.y = -0.6944 #traj[0][5]
        msg_pose.pose.orientation.z = 0.4229 #traj[0][6]

        food_pose_arr = [food_pose.pose.position.x, food_pose.pose.position.y, food_pose.pose.position.z]
        self.env._sim.add_target_to_viewer(food_pose_arr)

        self.move_to_pose(msg_pose)
        time.sleep(0.5)

        for wp in traj:
            control_action_pos = wp[:3]
            control_action_quat = quat2axisangle(wp[3:])
            control_action = np.concatenate([control_action_pos, control_action_quat])

            policy_step = True
            # Note: A bit hacky, but you can tune this loop to control the speed of the trajectory in Mujoco
            for i in range(20):
                self.env._robot.control(control_action, policy_step=policy_step)
                policy_step = False
                self.env.sim.step()
        time.sleep(0.5)

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

    def rotate_eef(self, angle):
        target_joint_positions = self.env._robot._joint_positions

        print(f"initial_joint_positions: {target_joint_positions}")

        target_joint_positions[5] += angle

        print(f"target_joint_positions: {target_joint_positions}")

        self.set_joint_position(target_joint_positions)

    def move_to_pose(self, pose):
        """
        Args:
            pose (PoseStamped): Pose to move the robot to.
        """

        # Reset the planner
        self.planner.reset()

        # Step the simulator to update the robot and environment state
        self.env.sim.step()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "world"
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.pose = pose.pose

        # Get the current joint positions
        start_joint_position = self.env._robot._joint_positions

        msg_trajectory = self.planner.get_motion_plan(
            start_joint_position=start_joint_position,
            goal=goal_pose,
            goal_type="pose_goal",
            velocity_scaling_factor=0.1
        )

        if msg_trajectory is not None:
            print("moveit exec trajectory...")
            self.execute_trajectory(msg_trajectory)
                
    def move_to_reset_pose(self):
        self.set_joint_position(self.reset_pos)

    def move_to_acq_pose(self):
        self.set_joint_position(self.acq_pos)

    def move_to_transfer_pose(self):
        self.set_joint_position(self.transfer_pos)

    def reset(self):
        self.move_to_acq_pose()

    def execute_traj_in_moveit(self, msg_robot_traj):
        start_time = time.time()
        success = True
        # Execute the scooping trajectory
        client = actionlib.SimpleActionClient(self._action_name_arm_exec, ExecuteTrajectoryAction)
        rospy.loginfo(f"Waiting for action server... {self._action_name_arm_exec}")
        client.wait_for_server()
        rospy.loginfo(f"Action server {self._action_name_arm_exec} is up!")

        # Create goal
        goal = ExecuteTrajectoryGoal()
        goal.trajectory = msg_robot_traj
        goal.trajectory.joint_trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.1)

        # Send goal to server
        client.send_goal(goal)
        client.wait_for_result()

        # Check if action was successful or preempted
        if client.get_state() == actionlib.GoalStatus.PREEMPTED:
            rospy.loginfo("Moveit robot trajectory execution preempted")
            success = False

        elif client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            result = client.get_result()
            
            # Process the result
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                error_dict = MoveItErrorCodes.__dict__
                error_string = list(error_dict.keys())[list(error_dict.values()).index(result.error_code.val)]
                rospy.logwarn(f"Moveit robot execution failed with error code {result.error_code.val}: {error_string}")
            else:
                rospy.loginfo(f"Moveit robot trajectory executed successfully in {time.time() - start_time:.3f}(s)")
                
        else:
            rospy.loginfo(f"Moveit robot trajectory failed or was aborted with status {client.get_state()}.")
            success = False

        return success

if __name__ == '__main__':
    rospy.init_node('mujoco_action_server', anonymous=True)
    mujoco_action_server = MujocoAction()

    rospy.spin()