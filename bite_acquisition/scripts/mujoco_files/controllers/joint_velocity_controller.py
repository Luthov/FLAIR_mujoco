"""
Note: THIS SCRIPT HAS NOT BEEN TESTED
"""
import numpy as np

from .base_controller import BaseController
from .linear_interpolator import LinearInterpolator
from utils.buffers import RingBuffer

class JointVelocityController(BaseController):
    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 velocity_limits,
                 control_dim,
                 input_max=1,
                 input_min=-1,
                 output_max=1.0,
                 output_min=-1.0,
                 kp=0.25,
                 ramp_ratio = 0.5,
                 control_freq=20,
                 policy_freq=10,
                 **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms
                 ):
        
        self._control_freq = control_freq

        super().__init__(sim, eef_name, joint_indexes, actuator_range, self._control_freq, policy_freq)

        # Set control dimension
        self._control_dim = control_dim

        # input and output max and min (allow for either explicit lists or single numbers)
        self._input_max = self.nums2array(input_max, self._control_dim)
        self._input_min = self.nums2array(input_min, self._control_dim)
        self._output_max = self.nums2array(output_max, self._control_dim)
        self._output_min = self.nums2array(output_min, self._control_dim)

        self._kp = self.nums2array(kp, self._control_dim)

        if type(kp) is float or type(kp) is int:
            # Scale kpp according to how wide the actuator range is for this robot
            low, high = self.actuator_limits
            self._kp = kp * (high - low)
        
        self._ki = self._kp * 0.005
        self._kd = self._kp * 0.001
        self._last_err = np.zeros(self._control_dim)
        self._derr_buf = RingBuffer(dim=self._control_dim, length=5)
        self._summed_err = np.zeros(self._control_dim)
        self._saturated = False
        self._last_joint_vel = np.zeros(self._control_dim)

        # Limits
        self._velocity_limits = np.array(velocity_limits) if velocity_limits is not None else None

        # Set interpolator
        self.interpolator = LinearInterpolator(self.joint_dim, control_freq, policy_freq, ramp_ratio)

        # Initialize
        self.goal_vel = None    # Goal velocity desired, pre-compensation
        self.des_vel = np.zeros(self._control_dim)  # Current velocity setpoint, pre-compensation

    def set_start(self):
        """
        Sets the start state of the interpolator
        """
        self.update(force=True)
        self.interpolator.set_start(self.joint_vel)
    
    def set_goal(self, velocities):
        """
        Sets goal based on input @velocities.

        Args:
            velocities (Iterable): Desired joint velocities
        
        Raises:
            AssertionError: [Invalid action dimension size]
        """
        # Update state
        self.update()

        # Check that goal dimension is correct
        assert (
            len(velocities) == self._control_dim
        ), "Goal action must match the robot's joint dimension space! Expected: {}, Got: {}".format(
            self._control_dim, len(velocities)
        )

        self.goal_vel = self.scale_action(velocities)

        if self._velocity_limits is not None:
            self.goal_vel = np.clip(self.goal_vel, self._velocity_limits[0], self._velocity_limits[1])

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_vel)
    
    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint (joint velocity)

        Returns:
            np.ndarray: Joint torques
        """
        # Check that goal has been set
        if self.goal_vel is None:
            self.set_goal(np.zeros(self._control_dim))
        
        # Update state
        self.update()

        # Use interpolator to get desired joint velocity
        desired_vel = self.interpolator.get_interpolated_goal()
        self.des_vel = desired_vel

        err = self.des_vel - self.joint_vel
        derr = err - self._last_err
        self._last_err = err
        self._derr_buf.push(derr)

        # Only add to I component if not saturated (anti-windup)
        if not self._saturated:
            self._summed_err += err
        
        # Calculate torques bia PID velocity controller
        torques = self._kp * err + self._ki * self._summed_err + self._kd * self._derr_buf.average

        # Clip torques
        self.torques = self.clip_torques(torques)

        # Check if saturated
        self._saturated = False if np.sum(np.abs(self.torques - torques)) == 0 else True

        # Always runs superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def reset_goal(self):
        """
        Resets joint velocity goal to current joint velocity
        """
        self.goal_vel = np.zeros(self._control_dim)

        # Reset interpolator
        self.interpolator.set_goal(self.goal_vel)

    @property 
    def control_limits(self):
        return self._input_min, self._input_max

    @property
    def name(self):
        return "JOINT_VELOCITY"