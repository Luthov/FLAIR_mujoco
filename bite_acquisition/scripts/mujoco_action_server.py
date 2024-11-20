#! /usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
import time

import rospy
import rospkg
import actionlib
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from moveit_msgs.msg import RobotTrajectory, MoveItErrorCodes
from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal

from bite_acquisition.msg import MujocoActionServerAction, MujocoActionServerResult, MujocoActionServerFeedback

try:
    from environments.arm_base import ArmBaseEnv
    from planners.moveit_planner import MoveItPlanner
    from utils.transform_utils import quat2euler, quat_distance, quat_error, quat2axisangle, quat2mat, mat2quat
    from controllers import load_controller_config
    # from bite_transfer import BiteTransfer
except:
    from feeding_mujoco.environments.arm_base import ArmBaseEnv
    from feeding_mujoco.planners.moveit_planner import MoveItPlanner
    from feeding_mujoco.utils.transform_utils import quat2euler, quat_distance, quat_error, quat2axisangle, quat2mat, mat2quat
    from feeding_mujoco.controllers import load_controller_config
    # from feeding_mujoco.bite_transfer_mujoco import BiteTransfer

TRANSFER_ANGLE = 80
DISTANCE_WITHIN_MOUTH = 0.06
ENTRY_INSIDE_ANGLE = 90
EXIT_DEPTH = 0.03

class MujocoAction(object):

    def __init__(self):

        self.result = MujocoActionServerResult()

        controller_config = load_controller_config(default_controller="IK_POSE")
        controller_config["control_delta"] = False

        model_folder = rospkg.RosPack().get_path("feeding_mujoco") + "/src/feeding_mujoco/models"
        env_config = {
            # "model_path": model_folder + "/envs/feeding_luke/feeding.xml",
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
        # self.bite_transfer = BiteTransfer()

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
        # self.acq_pos = np.radians([0.0, -65.0, -25.0, 0.0, 65.0, -90.0])
        # TODO: Luke: this transfer pose also seems a bit high (in terms of z position)
        # self.transfer_pos = np.radians([0.0, -65.0, -25.0, 0.0, 20.0, 0.0])

        # self.mouth_pose = np.array([0.70, 0.0, 0.545])
        self.initial_transfer_pose_reached = False

        self.action_name = "mujoco_action_server"
        self.action_server = actionlib.SimpleActionServer(self.action_name, MujocoActionServerAction, execute_cb=self.execute_callback, auto_start = False)
        self.action_server.start()

     
    def execute_callback(self, goal):

        try:
            if goal.function_name == "move_to_pose":
                self.move_to_pose(goal.goal_point)

            elif goal.function_name == "move_to_reset_pos":
                self.move_to_reset_pose()

            elif goal.function_name == "move_to_acq_pose":
                self.move_to_acq_pose(goal.goal_point)

            elif goal.function_name == "move_to_transfer_pose":
                self.move_to_transfer_pose(goal.goal_point)

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

        goal_pose_arr = [goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z]
        # self.env._sim.add_target_to_viewer(goal_pose_arr)

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

    def move_to_acq_pose(self, pose):
        self.move_to_pose(pose)
        # self.set_joint_position(self.acq_pos)

    def reset(self):
        self.move_to_acq_pose()

    def generate_pose(self, position, orientation, offset= [0,0,0]):
        """
        Generate a PoseStamped with given position, orientation, and optional offset.

        Parameters:
            position (array): Array of [x, y, z] coordinates.
            orientation (array): Array of [w, x, y, z] quaternion values.
            position_offset (array, optional): Array of offset values to add to position.

        Returns:
            PoseStamped: PoseStamped message with the specified position and orientation.
        """
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = position[0] + offset[0]
        pose.pose.position.y = position[1] + offset[1]
        pose.pose.position.z = position[2] + offset[2]
        pose.pose.orientation.w = orientation[0]
        pose.pose.orientation.x = orientation[1]
        pose.pose.orientation.y = orientation[2]
        pose.pose.orientation.z = orientation[3]

        return pose        
    
    def move_to_initial_transfer_pose(self, mouth_pose):

        # quat_mouth_pose = np.array([0.4090, 0.5783, 0.5719, 0.4137])
        # quat_mouth_pose = np.array([0.1204765,  -0.6980302, -0.1129327, -0.6967678])
        quat_mouth_pose = np.array(R.from_euler('xyz', [0, 90, 0], degrees=True).as_quat())
        quat_mouth_pose_wxyz = quat_mouth_pose[[3, 0, 1, 2]]
        # print("quat_mouth_pose_wxyz:", quat_mouth_pose_wxyz)
        # quat_mouth_pose = np.array([0, 0.707, 0, 0.707])        

        # Make sure the quaternion is normalized
        quat_mouth_pose = quat_mouth_pose / np.linalg.norm(quat_mouth_pose)


        mouth_pose_1 = np.concatenate((np.array(mouth_pose), quat_mouth_pose))
        

        # Convert from spoon pose to eef pose in world frame
        eef_in_world = self.get_eef_pose_from_spoon_pose(mouth_pose_1, np.array([0.0, 0.0, 0.13]))
        # print("eef_in_world:", eef_in_world)
        eef_pose_1_in_world = self.generate_pose(eef_in_world[:3], eef_in_world[3:])
        self.move_to_pose(eef_pose_1_in_world)
        self.initial_transfer_pose_reached = True

    def move_to_inside_mouth_pose(self, mouth_pose):

        # quat_mouth_pose = np.array([0.4090, 0.5783, 0.5719, 0.4137])
        # quat_mouth_pose = np.array([0.1204765,  -0.6980302, -0.1129327, -0.6967678])
        quat_mouth_pose = np.array(R.from_euler('xyz', [0, ENTRY_INSIDE_ANGLE, 0], degrees=True).as_quat())
        quat_mouth_pose_wxyz = quat_mouth_pose[[3, 0, 1, 2]]
        print("quat_mouth_pose_wxyz:", quat_mouth_pose_wxyz)
        # quat_mouth_pose = np.array([0, 0.707, 0, 0.707])        

        # Make sure the quaternion is normalized
        quat_mouth_pose = quat_mouth_pose / np.linalg.norm(quat_mouth_pose)
        mouth_pose_2 = np.concatenate((np.array(mouth_pose), quat_mouth_pose))
        # Convert from spoon pose to eef pose in world frame
        eef_in_world = self.get_eef_pose_from_spoon_pose(mouth_pose_2, np.array([0.0, 0.0, 0.13]))
        print("eef_in_world:", eef_in_world)
        eef_pose_2_in_world = self.generate_pose(eef_in_world[:3], eef_in_world[3:], offset=[DISTANCE_WITHIN_MOUTH, 0, 0.01])
        self.move_to_pose(eef_pose_2_in_world)

    def move_to_outside_mouth_pose(self, mouth_pose):

        quat_mouth_pose = np.array(R.from_euler('xyz', [0, 90, 0], degrees=True).as_quat())
        quat_mouth_pose_wxyz = quat_mouth_pose[[3, 0, 1, 2]]
        print("quat_mouth_pose_wxyz:", quat_mouth_pose_wxyz)
        # quat_mouth_pose = np.array([0, 0.707, 0, 0.707])        

        # Make sure the quaternion is normalized
        quat_mouth_pose = quat_mouth_pose / np.linalg.norm(quat_mouth_pose)
        mouth_pose_3 = np.concatenate((np.array(mouth_pose), quat_mouth_pose))
        # Convert from spoon pose to eef pose in world frame
        eef_in_world = self.get_eef_pose_from_spoon_pose(mouth_pose_3, np.array([0.0, 0.0, 0.13]))
        print("eef_in_world:", eef_in_world)
        eef_pose_3_in_world = self.generate_pose(eef_in_world[:3], eef_in_world[3:], offset=[EXIT_DEPTH, 0, 0.01])

        self.move_to_pose(eef_pose_3_in_world)


    def move_to_exit_pose(self, mouth_pose):
        
        # quat_mouth_pose = np.array([0.2979085, 0.6439846, 0.6369263, 0.2979085]) 
        quat_mouth_pose = np.array(R.from_euler('xyz', [0, TRANSFER_ANGLE, 0], degrees=True).as_quat())

        # Make sure the quaternion is normalized
        quat_mouth_pose = quat_mouth_pose / np.linalg.norm(quat_mouth_pose)

        mouth_pose_4 = np.concatenate((np.array(mouth_pose), quat_mouth_pose))

        # Convert from spoon pose to eef pose in world frame
        eef_4_in_world = self.get_eef_pose_from_spoon_pose(mouth_pose_4, np.array([0.0, 0.0, 0.13]))
        eef_pose_4_in_world = self.generate_pose(eef_4_in_world[:3], eef_4_in_world[3:], offset=[-0.03, 0 , 0.03])
        self.move_to_pose(eef_pose_4_in_world)

    def get_eef_pose_from_spoon_pose(self, spoon_pose, spoon_offset):
        """
        Removes the spoon offset from the spoon trajectory to get the eef trajectory

        Args:
            spoon_pose (ndarray): Numpy array of the spoon pose in [x, y, z, w, x, y, z]
            spoon_offset (ndarray): Offset to be applied to the spoon in [x, y, z]

        Returns:
            ndarray: Numpy array of the eef pose in [x, y, z, w, x, y, z]
        """
        # get rotation matrix of eef orientation
        spoon_pos = spoon_pose[: 3]
        spoon_quat = spoon_pose[3 :]
        spoon_quat = np.quaternion(spoon_quat[0], spoon_quat[1], spoon_quat[2], spoon_quat[3])
        spoon_rot_mat = quaternion.as_rotation_matrix(spoon_quat)

        # calculate eef position and orientation in world frame
        eef_offset_world = spoon_rot_mat.dot(spoon_offset)
        eef_pos = spoon_pos - eef_offset_world

        # spoon quat should be the same as eef_quat
        eef_quat = spoon_quat

        # update eef_traj
        eef_pose = np.hstack((eef_pos, eef_quat.components))
        
        return eef_pose
    
    def close_mouth(self):
        i = 0
        # print(self.env.sim.get_actuator_id_from_name("jaw_pitch"))
        while self.env.sim.get_actuator_ctrl(41) < 20:
            self.env.sim.step()
            # forces = self.env._sim.plot_contacts_with_force_colors([-1, 17], 20)
            skull_forces = self.env._sim.plot_contacts_with_force_colors([27, 17], 20)
            # jaw_forces = self.env._sim.plot_contacts_with_force_colors([28, 17], 20)
            # if forces is not None:
                # self.contact_forces.append(forces)
                # self.mesh_sim_times.append(self.env._sim.time)
            if skull_forces is not None:
                self.skull_contact_forces.append(skull_forces)
                self.skull_sim_times.append(self.env._sim.time)
            # if jaw_forces is not None:
                # self.jaw_contact_forces.append(jaw_forces)
                # self.jaw_sim_times.append(self.env._sim.time)
            self.env.sim.set_actuator_ctrl(i, 41)
            i += 0.02
            self.env.sim.step()
        self.env.sim.step()


    def wait_time(self, time):
        rospy.sleep(time)
    
    def move_to_transfer_pose(self, pose):
        # self.move_to_pose(pose)
        # self.set_joint_position(self.transfer_pos)
        mouth_pose = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        self.move_to_initial_transfer_pose(mouth_pose)
        # print("initial transfer pose reached")
        self.move_to_inside_mouth_pose(mouth_pose)
        # print("inside mouth pose reached")
        # self.close_mouth()
        # print("mouth closed")
        self.move_to_outside_mouth_pose(mouth_pose)
        # print("outside mouth pose reached")
        self.move_to_exit_pose(mouth_pose)
        # print("exit pose reached")

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