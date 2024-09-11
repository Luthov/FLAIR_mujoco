import numpy as np

from .base_controller import BaseController
from .linear_interpolator import LinearInterpolator

from utils.controller_utils import set_goal_position

class JointPositionController(BaseController):
    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 qpos_limits,
                 control_dim,
                 input_max=1,
                 input_min=-1,
                 output_max=0.8,
                 output_min=-0.8,
                 kp = 300,
                #  impedance = 0.01,
                 damping_ratio = 1.0,
                 ramp_ratio = 0.2,
                 control_freq = 20,
                 policy_freq = 10,
                 ** kwargs,  # does nothing; used so no error raised when dict is passed with extra terms
                 ):
        
        self._control_freq = control_freq
        
        super().__init__(sim, eef_name, joint_indexes, actuator_range, ramp_ratio, self._control_freq, policy_freq)

        # Set control dimension
        self._control_dim = control_dim

        # input and output max and min (allow for either explicit lists or single numbers)
        self._input_max = self.nums2array(input_max, self._control_dim)
        self._input_min = self.nums2array(input_min, self._control_dim)
        self._output_max = self.nums2array(output_max, self._control_dim)
        self._output_min = self.nums2array(output_min, self._control_dim)

        self._position_limits = np.array(qpos_limits)

        damping_ratio = self.nums2array(damping_ratio, self._control_dim)
        self._kp = self.nums2array(kp, self._control_dim)
        self._kd = 2 * np.sqrt(self._kp) * damping_ratio
        # self._ki = self.nums2array(impedance, self._control_dim)

        # Set interpolator
        self.interpolator = LinearInterpolator(self.joint_dim, control_freq, policy_freq, ramp_ratio)

        # Initialize
        self.goal_qpos = None
        # self.reset()

    def set_start(self):
        """
        Sets the start state of the interpolator
        """
        self.update(force=True)
        self.interpolator.set_start(self.joint_pos)
    
    def set_goal(self, action, set_qpos=None):
        """
        Sets goal based on input @action. 

        Note that @action expected to be in the joint pos command format

        Args:
            action (Iterable): Desired relative joint position goal state
            set_qpos (Iterable): If set, overrides @action and sets the desired absolute joint position goal state

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        action = np.array(action)

        # Update robot arm state
        self.update()

        # Check that goal dimension is correct
        assert len(action) == self._control_dim, "Goal dimension must match robot's joint dimension"

        # Scale action
        scaled_action = self.scale_action(action.copy())
        
        # Set goal position - force set_pos to desired qpos
        self.goal_qpos = set_goal_position(
            scaled_action, self.joint_pos, None, set_pos=None
        )
        # print(">>> [JPC] scaled action:", scaled_action)

        self.interpolator.set_goal(self.goal_qpos)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired goal (joint pos)

        Returns:
            np.ndarray: Joint torques
        """
        # Check that goal has been set
        if self.goal_qpos is None:
            self.set_goal(np.zeros(self._control_dim))
        
        # Update robot arm state
        self.update()
        
        # Use interpolator to get desired joint position
        desired_qpos = self.interpolator.get_interpolated_goal()
        self._des_qpos = desired_qpos
        # print(">>> [JPC] int step:", self.interpolator._step, "des qpos:", desired_qpos)

        # Torques = pos error * kp * vel_err * kd
        pos_error = self.joint_sum(desired_qpos, -self.joint_pos)
        vel_error = -self.joint_vel
        desired_torque = np.multiply(np.array(pos_error), np.array(self._kp)) + np.multiply(vel_error, self._kd)
        self.torques = desired_torque
        
        # print("pos error:", pos_error)
        # print("vel error:", vel_error)

        # always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def reset_goal(self):
        """
        Resets joint position goal to current joint position
        """

        self.goal_qpos = self.joint_pos.copy()

        self.interpolator.set_goal(self.goal_qpos)

    
    @property 
    def control_limits(self):
        return self._input_min, self._input_max

    @property
    def name(self):
        return "JOINT_POSITION"