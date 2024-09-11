"""
This is deprecated. Use ik_pybullet instead.
"""
import rospkg
import numpy as np
from tracikpy import TracIKSolver

from .joint_velocity_controller import JointVelocityController
from .joint_position_controller import JointPositionController
from .linear_interpolator import LinearInterpolator

from utils.transform_utils import (
    mat2quat, 
    mat2pose,
    quat2mat,
    quat_inverse,   
    quat_multiply,  
    quat2euler,
    pose2mat,
    pose_inv,
    pose_in_A_to_pose_in_B,
    axisangle2quat,
    clip_rotation_wxyz,
    clip_translation,
    quat_distance,
    quat2axisangle
)
from utils.controller_utils import (
    nullspace_torques, 
    opspace_matrices,
    set_goal_position,
    set_goal_orientation,
    orientation_error
)

class InverseKinematicsController(JointPositionController):
    def __init__(
            self,
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
            eef_rot_offset,     # Quat (w,x,y,z) offset to convert mujoco eef to pybullet eef
            qpos_limits,
            control_dim=6,
            control_freq=20,
            policy_freq=10,
            kp=100,
            damping_ratio=0.05,
            ramp_ratio=0.5,
            input_max=1,
            input_min=-1,
            output_max=1.0,
            output_min=-1.0,
            control_delta = True,
            **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms
    ):
                
        # Run superclass inits
        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
            qpos_limits=qpos_limits,
            control_dim=control_dim,
            input_max=input_max,
            input_min=input_min,
            output_max=output_max,
            output_min=output_min,
            kp=kp,
            damping_ratio=damping_ratio,
            ramp_ratio=ramp_ratio,
            control_freq=control_freq,
            policy_freq=policy_freq,
            **kwargs,  
        )

        self.robot_name = "XArm6"

        self._control_dim = control_dim

        # Determine whether we want to use delta or absolute values as inputs
        self._use_delta = control_delta

        # Rotation offsets (for mujoco eef -> pybullet eef) and rest poses
        self.eef_rot_offset = eef_rot_offset
        self.rotation_offset = None
        self.rest_poses = None

        # Set the reference robot target pos / orientation (to prevent drift / weird ik numerical behavior over time)
        self.reference_target_pos = self.ee_pos
        self.reference_target_orn = mat2quat(self.ee_ori_mat)

        # IK solver
        pkg_path = rospkg.RosPack().get_path('feeding_mujoco')
        self._ik_solver = TracIKSolver(
            urdf_file = pkg_path + "/src/feeding_mujoco/models/robots/xarm6/xarm6_with_ft_sensor_gripper.urdf",
            base_link = "link_base",
            tip_link = "link_tcp",
            solve_type="Speed",
            epsilon=1e-3,
        )

        # Interpolator
        self.interpolator_pos = LinearInterpolator(3, control_freq, policy_freq, ramp_ratio)
        self.interpolator_ori = LinearInterpolator(4, control_freq, policy_freq, ramp_ratio, ori_interpolate="quat")

        # Interpolator-related attributes
        self.ori_ref = None
        self.relative_ori = None

        # Commanded pos and resulting commanded vel
        self.commanded_joint_positions = None
        self.commanded_joint_velocities = None

        # Should be in (0, 1], smaller values mean less sensitivity.
        self.user_sensitivity = 1.0
    
    def get_control(self, pos=None, rotation=None, update_targets=False):
        """
        Returns joint positions to control the robot after the target end effector
        position and orientation are updated from arguments @dpos and @rotation.
        If no arguments are provided, joint positions will be computed based
        on the previously recorded target.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): a rotation matrix of shape (3, 3) corresponding
                to the desired rotation from the current orientation of the end effector.
            update_targets (bool): whether to update ik target pos / ori attributes or not

        Returns:
            np.array: a flat array of joint position commands to apply to try and achieve the desired input control.
        """

        # Compute new target joint positions if arguments are provided
        if (pos is not None) and (rotation is not None):
            self.commanded_joint_positions = np.array(
                self.joint_positions_for_eef_command(pos, rotation, update_targets)
            )
        
        # Absolute joint positions
        positions = self.commanded_joint_positions

        return positions
    
    def joint_positions_for_eef_command(self, pos, rotation, update_targets=False):
        """
        This function runs inverse kinematics to back out target joint positions
        from the provided end effector command.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): a rotation matrix of shape (3, 3) corresponding
                to the desired rotation from the current orientation of the end effector.
            update_targets (bool): whether to update ik target pos / ori attributes or not

        Returns:
            list: A list of size @num_joints corresponding to the target joint angles.
        """

        des_pos = self.interpolator_pos.get_interpolated_goal()
        ori_error = self.interpolator_ori.get_interpolated_goal()

        # print("[IK joint pos] des_pos:", des_pos)
        # print("[IK joint pos] des ori:", ori_error)

        arm_joint_pos = self.inverse_kinematics(des_pos, ori_error)

        # calc fk again to check
        pose_out = self._ik_solver.fk(arm_joint_pos)
        # convert mat to quat
        # print("[IK joint pos] pose_out:", pose_out[:3, 3], mat2quat(pose_out[:3, :3]))


        return arm_joint_pos
    
    def inverse_kinematics(self, pos, rotation, qinit=None, max_iterations=1000):

        if qinit is None:
            qinit = self.joint_pos

        targets = (pos, rotation)

        des_pose = pose2mat(targets)

        arm_joint_pos = None
        
        iterations = 0
        while arm_joint_pos is None and iterations < max_iterations:
            import time
            start_t = time.time()
            arm_joint_pos = self._ik_solver.ik(des_pose,
                                            qinit=qinit)   
            # print(f"iter {iterations}: time taken: {time.time() - start_t}")     
            iterations += 1                                       
            
        # print("[IK] des_pos:", des_pose[:3, 3], mat2quat(des_pose[:3, :3]), "arm_joint_pos:", arm_joint_pos)

        assert arm_joint_pos is not None, "IK failed to find a solution"

        return arm_joint_pos
    
    def set_start(self):
        """
        Sets the start state of the controller
        """
        self.update(force=True)
        
        self.reset_goal()
    
    def set_goal(self, action, set_ik=None):
        """
        Sets internal goal state of the controller based on input @action.

        Note that this controller wraps a JointVelocityController, and so determines the desired velocities
        to achieve the inputted pose, and sets its internal setpoint in terms of joint velocities

        Args:
            action (Iterable): Desired relative position / orientation goal state
            set_ik (Iterable): If set, overrides @action and sets the desired absolute joint position goal state
        """
        # Update robot arm state
        self.update()

        # Get requested delta inputs if we're using interpolators
        # Quat here is in wxyz format
        # (dpos, dquat) = self._clip_ik_input(action[:3], action[3:])

        # If using deltas:
        if self._use_delta:
            # Remove clipping
            dpos = action[:3]
            dquat = axisangle2quat(action[3:])

            set_pos = dpos * self.user_sensitivity + self.ee_pos

            # TODO: this calc is wrong
            set_ori = set_goal_orientation(
                action[3:], self.ee_ori_mat, orientation_limit=None, set_ori=quat2mat(axisangle2quat(action[3:6]))
            )
            
        # Interpret actions as absolute IK goals
        else:
            set_pos = action[:3]
            set_ori = quat2mat(axisangle2quat(action[3:]))
        
        self.goal_pos = set_pos
        self.goal_ori = set_ori

        # print("[set goal] goal_pos:", self.goal_pos)
        # print("[set goal] goal_ori:", mat2quat(self.goal_ori))
        
        # Set interpolated goals (eef pos and eef ori)
        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start

            # print("quat distance: ", quat2axisangle(quat_distance(mat2quat(self.ori_ref), mat2quat(self.goal_ori))))
            # print("axisangle distance: ", orientation_error(self.goal_ori, self.ori_ref))
            
        if self.interpolator_ori is not None:
            self.interpolator_ori.set_goal(
                # quat_distance(mat2quat(self.goal_ori), mat2quat(self.ori_ref))
                mat2quat(self.goal_ori)
            )
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0
            
        # Compute desired joint positions to achieve eef pos / ori
        positions = self.get_control(pos=self.goal_pos, rotation=self.goal_ori, update_targets=True)

        # Set the goal positions for the underlying position controller
        super().set_goal(action = positions-self.joint_pos)    

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        # Update interpolated action if necessary
        desired_pos = None
        rotation = None

        # Get the updated desired target eef pos
        desired_pos = self.interpolator_pos.get_interpolated_goal()

        # Relative orientation based on difference between current ori and ref

        self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)
        des_ori = self.interpolator_ori.get_interpolated_goal()
        rotation = quat2mat(des_ori)        

        # print("[IK Run controller] desired_pos:", desired_pos)
        # print("[IK Run controller] des ori:", des_ori)

        # Perform IK to get the desired joint positions
        positions = self.get_control(pos=desired_pos, rotation=rotation, update_targets=True)
        
        super().set_goal(action = positions-self.joint_pos)

        # Run controller with given action
        return super().run_controller()

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # Then, update the rest pose from the initial joints
        self.rest_poses = list(self.initial_joint_pos)

    def reset_goal(self):
        """
        Resets the goal to the current pose of the robot
        """

        # Also reset interpolators if required
        self.goal_ori = np.array(self.ee_ori_mat)
        self.goal_pos = np.array(self.ee_pos)

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                quat_distance(mat2quat(self.goal_ori), mat2quat(self.ori_ref))
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    def _clip_ik_input(self, dpos, rotation):
        """
        Helper function that clips desired ik input deltas into a valid range.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): relative rotation in scaled axis angle form (ax, ay, az)
                corresponding to the (relative) desired orientation of the end effector.

        Returns:
            2-tuple:

                - (np.array) clipped dpos
                - (np.array) clipped rotation
        """
        # scale input range to desired magnitude
        if dpos.any():
            dpos, _ = clip_translation(dpos, self.ik_pos_limit)

        # Map input to quaternion
        rotation = axisangle2quat(rotation)

        # Clip orientation to desired magnitude
        rotation, _ = clip_rotation_wxyz(rotation, self.ik_ori_limit)

        return dpos, rotation

    @staticmethod
    def _get_current_error(current, set_point):
        """
        Returns an array of differences between the desired joint positions and current
        joint positions. Useful for PID control.

        Args:
            current (np.array): the current joint positions
            set_point (np.array): the joint positions that are desired as a numpy array

        Returns:
            np.array: the current error in the joint positions
        """
        error = current - set_point
        return error

    @property
    def control_limits(self):
        """
        The limits over this controller's action space, as specified by self.ik_pos_limit and self.ik_ori_limit
        and overriding the superclass method

        Returns:
            2-tuple:

                - (np.array) minimum control values
                - (np.array) maximum control values
        """
        max_limit = np.concatenate([self.ik_pos_limit * np.ones(3), self.ik_ori_limit * np.ones(3)])
        return -max_limit, max_limit

    @property
    def name(self):
        return "IK_POSE"