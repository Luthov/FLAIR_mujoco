import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
from geometry_msgs.msg import PoseStamped

# from bite_acquisition.srv import PoseCommand, PoseCommandRequest, PoseCommandResponse
# from bite_acquisition.srv import JointCommand, JointCommandRequest, JointCommandResponse

from .base import RobotController

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

class MujocoRobotController(RobotController):

    def __init__(self):

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
        # self.acq_pos = [0.02961542490849557, 0.06645885626898033, 5.281153370174079, 3.1258331315231405, 2.04992230955969, 4.737864779372388]
        # self.transfer_pos = [0.3333664491938215, 1.4858324332736625, 1.5856359930210362, 0.8180422599332581, 1.5794872866962613, 4.604932028296647]

    def reset(self):
        # TODO: Probs can look at 
        self.move_to_acq_pose()

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
                    print("Moved to pose successfuly!")
                    break
                    
                    

    def move_to_pose(self, pose):

        # Reset the planner
        self.planner.reset()

        # Step the simulator to update the robot and environment state
        self.env.sim.step()

        # Get the target pose
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

        # Get the current joint positions
        start_joint_position = self.env._robot._joint_positions

        msg_trajectory = self.planner.get_motion_plan(
            start_joint_position=start_joint_position,
            goal=goal_pose,
            goal_type="pose_goal",
            velocity_scaling_factor=0.1
        )

        if msg_trajectory is not None:
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
                    print("Moved to pose successfuly!")
                    break
                

    def move_to_acq_pose(self):
        self.set_joint_position(self.acq_pos)

    def move_to_transfer_pose(self):
        self.set_joint_position(self.transfer_pos)

if __name__ == '__main__':
    rospy.init_node('robot_controller', anonymous=True)
    robot_controller = MujocoRobotController()

    input('Press enter to move to acquisition position...')
    robot_controller.move_to_acq_pose()

    input('Press enter to move to transfer position...')
    robot_controller.move_to_transfer_pose()

    input('Press enter to reset the robot...')
    robot_controller.reset()