import numpy as np
import math

from .base_controller import BaseController
from .linear_interpolator import LinearInterpolator
from utils.transform_utils import (
    mat2quat, 
    mat2euler,
    euler2mat,
    quat2mat,
    axisangle2quat
)
from utils.controller_utils import (
    nullspace_torques, 
    opspace_matrices,
    set_goal_position,
    set_goal_orientation,
    orientation_error
)

"""
This file is modified from the original file at:
https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/controllers/osc.py
"""

class OperationalSpaceController(BaseController):
    """
    Controller for controlling robot arm via operational space control. Allows position and / or orientation control
    of the robot's end effector. For detailed information as to the mathematical foundation for this controller, please
    reference http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

    NOTE: Control input actions can either be taken to be relative to the current position / orientation of the
    end effector or absolute values. In either case, a given action to this controller is assumed to be of the form:
    (x, y, z, ax, ay, az) if controlling pos and ori or simply (x, y, z) if only controlling pos

    Args:
        
    """
    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                 output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
                 kp = 150,
                #  impedance = 0.01,
                 damping_ratio = 1.0,
                 ramp_ratio = 0.2,
                 control_freq = 20,
                 policy_freq = 10,
                 position_limits = None,
                 orientation_limits = None,
                 control_ori = True,
                 control_delta = True,
                 uncouple_pos_ori = True,
                 ** kwargs,  # does nothing; used so no error raised when dict is passed with extra terms
                 ):
        
        self._control_freq = control_freq
        
        super().__init__(sim, eef_name, joint_indexes, actuator_range, ramp_ratio, self._control_freq, policy_freq)

        # Determine whether this is pos ori or just pos
        self._use_ori = control_ori

        # Determine whether we want to use delta or absolute values as inputs
        self._use_delta = control_delta

        # Set control dimension
        self._control_dim = 6 if self._use_ori else 3
        self._name_suffix = "POSE" if self._use_ori else "POSITION"

        # Input and output max and min (allow for either explicit lists or single numbers)
        self._input_max = self.nums2array(input_max, self._control_dim)
        self._input_min = self.nums2array(input_min, self._control_dim)
        self._output_max = self.nums2array(output_max, self._control_dim)
        self._output_min = self.nums2array(output_min, self._control_dim)

        # Gains
        self._kp = self.nums2array(kp, 6)
        self._kd = 2 * np.sqrt(self._kp) * damping_ratio

        # Limits
        self._position_limits = np.array(position_limits) if position_limits is not None else position_limits
        self._orientation_limits = np.array(orientation_limits) if orientation_limits is not None else orientation_limits

        # Interpolator
        self.interpolator_pos = LinearInterpolator(3, control_freq, policy_freq, ramp_ratio)
        if self._use_ori:
            self.interpolator_ori = LinearInterpolator(3, control_freq, policy_freq, ramp_ratio, ori_interpolate="euler")
        else:
            self.interpolator_ori = None

        # Whether or not pos and ori should be coupled
        self._uncouple_pos_ori = uncouple_pos_ori

        # Initialize goals based on initial pos / ori
        self.goal_ori = np.array(self.initial_ee_ori_mat)
        self.goal_pos = np.array(self.initial_ee_pos)

        self.relative_ori = np.zeros(3)
        self.ori_ref = None

        # self.reset()
    
    def set_start(self):
        """
        Sets the start state of the controller
        """
        self.update(force=True)
        
        self.reset_goal()

    def set_goal(self, action, set_pos=None, set_ori=None):
        """
        Sets goal based on input @action.

        """
        action = np.array(action)

        # Update robot arm state
        self.update()

        # If using deltas
        if self._use_delta:
            if action is not None:
                scaled_action = self.scale_action(action)
                
                if not self._use_ori and set_ori is None:
                    # Set default control for ori since user isn't actively controlling ori
                    # Used to define a global orientation for set_goal_orientation calculation
                    set_ori = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


            else:
                scaled_action = []
        
        # else interpret actions as absolute values
        else:
            if set_pos is None:
                set_pos = action[:3]
            # Set default control for ori if we're only using position control
            if set_ori is None:
                if self._use_ori:
                    set_ori = quat2mat(axisangle2quat(action[3:6]))
                else:
                    set_ori = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

            # No need to scale since these are absolute values
            scaled_action = action

        # Only update goal orientation if there is a valid action orientation OR if we're using absolute ori
        bools = [0.0 if math.isclose(elem, 0.0) else 1.0 for elem in scaled_action[3:]]

        if sum(bools) > 0.0 or set_ori is not None:
            self.goal_ori = set_goal_orientation(
                scaled_action[3:], self.ee_ori_mat, orientation_limit=self._orientation_limits, set_ori=set_ori
            )
            
        self.goal_pos = set_goal_position(
            scaled_action[:3], self.ee_pos, position_limit=self._position_limits, set_pos=set_pos
        )
        # print(">>> [OSC] curr eef pos:", self.ee_pos)
        # print(">>> [OSC] scaled action:", scaled_action)
        # print(">>> [OSC] goal pos:", self.goal_pos)
        # print(f">>> [OSC] goal ori: (quat) {mat2quat(self.goal_ori)} (euler) {mat2euler(self.goal_ori)}")

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)    # ref is the current orientation at the start

            # Setting orientation error as goal
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )   # goal is the total orientation error
            self.relative_ori = np.zeros(3)             # relative orientation always starts at 0
            
    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint.

        Executes Operational Space Control (OSC) -- either position only or position and orientation.

        A detailed overview of derivation of OSC equations can be seen at:
        http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

        Returns:
             np.array: Command torques
        """
        # Update robot arm state
        self.update()

        desired_pos = self.interpolator_pos.get_interpolated_goal()
        # print(">>> [OSC] desired pos:", desired_pos)

        if self.interpolator_ori is not None:
            # Get relative orientation based on difference between current ori and ref
            # print(">>> [OSC] current ori:", mat2quat(self.ee_ori_mat))
            # print(">>> [OSC] ref ori:", mat2quat(self.ori_ref))
            self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)
            ori_error = self.interpolator_ori.get_interpolated_goal()  
            # print(">>> [OSC] ori error (interpolator goal):", ori_error)      
        else:
            desired_ori = np.array(self.goal_ori)
            ori_error = orientation_error(desired_ori, self.ee_ori_mat)

        # Calculate the desired force and torque based on errors
        position_error = desired_pos - self.ee_pos
        vel_pos_error = -self.ee_pos_vel
        # print(">>> [OSC] position error:", position_error)
        # print(">>> [OSC] xvelp error:", vel_pos_error)

        # F_r = kp * pos_err + kd * vel_err
        desired_force = np.multiply(np.array(position_error), np.array(self._kp[0:3])) + np.multiply(
            vel_pos_error, self._kd[0:3]
        )

        vel_ori_error = -self.ee_ori_vel
        # print(">>> [OSC] orientation error:", ori_error)
        # print(">>> [OSC] xvelr error:", vel_ori_error)

        # Tau_r = kp * ori_err + kd * vel_err
        desired_torque = np.multiply(np.array(ori_error), np.array(self._kp[3:6])) + np.multiply(
            vel_ori_error, self._kd[3:6]
        )

        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            self.mass_matrix, self.J_full, self.J_pos, self.J_ori
        )

        # Decouples desired positional control from orientation control
        if self._uncouple_pos_ori:
            decoupled_force = np.dot(lambda_pos, desired_force)
            decoupled_torque = np.dot(lambda_ori, desired_torque)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(lambda_full, desired_wrench)

        # Gamma (without null torques) = J^T * F 
        self.torques = np.dot(self.J_full.T, decoupled_wrench)

        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        self.torques += nullspace_torques(
            self.mass_matrix, nullspace_matrix, self.initial_joint_pos, self.joint_pos, self.joint_vel
        )

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def reset_goal(self):
        """
        Resets the goal to the current state of the robot
        """
        self.goal_ori = np.array(self.ee_ori_mat)
        self.goal_pos = np.array(self.ee_pos)

        # Also reset interpolators if required

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    def reset(self):
        """
        Resets the state of the controller
        """
        pass


    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # We also need to reset the goal in case the old goals were set to the initial confguration
        self.reset_goal()

    @property 
    def control_limits(self):
        return self._input_min, self._input_max

    @property
    def name(self):
        return "OSC_" + self._name_suffix

