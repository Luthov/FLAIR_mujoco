import numpy as np
from utils.transform_utils import quat2mat, mat2quat, mat2euler, euler2mat, quat_slerp

class LinearInterpolator():
    """
    Simple class for implementing a linear interpolator.

    Abstracted to interpolate n-dimensions

    Args:
        ndim (int): Number of dimensions to interpolate

        controller_freq (float): Frequency (Hz) of the controller

        policy_freq (float): Frequency (Hz) of the policy model

        ramp_ratio (float): Percentage of interpolation timesteps across which we will interpolate to a goal position.

            :Note: Num total interpolation steps will be equal to np.ceil(ramp_ratio * controller_freq / policy_freq)
                    i.e.: how many controller steps we get per action space update

        ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
            Specified string determines assumed type of input:

                `'euler'`: Euler orientation inputs
                `'quat'`: Quaternion inputs
    """

    def __init__(
        self,
        ndim,
        controller_freq,
        policy_freq,
        ramp_ratio=0.2,
        ori_interpolate=None,
    ):
        self._ndim = ndim                         # Number of dimensions to interpolate
        self._ori_interpolate = ori_interpolate  # Whether this is interpolating orientation or not
        self._step = 0  # Current step of the interpolator

        self.total_steps = np.ceil(
            ramp_ratio * controller_freq / policy_freq
        )  # Total number of steps per interpolator action

        # print("Controller freq:", controller_freq)
        # print("Policy freq:", policy_freq)
        # print("Total interpolation steps:", self.total_steps)

        self.set_states()

    def set_states(self):
        """
        Initializes self.start and self.goal with correct dimensions.

        Args:
            ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
                Specified string determines assumed type of input:

                    `'euler'`: Euler orientation inputs
                    `'quat'`: Quaternion inputs
        """
        # Set start and goal states
        if self._ori_interpolate is not None:
            if self._ori_interpolate == "euler":
                self.start = np.zeros(3)
            else:  # quaternions - w,x,y,z
                self.start = np.array((1.0, 0, 0, 0))     
        else:
            self.start = np.zeros(self._ndim)

        self.goal = np.array(self.start)

    def set_start(self, start):
        """
        Takes a requested (absolute) start and updates internal parameters for next interpolation step

        Args:
            np.array: Requested start (absolute value). Should be same dimension as self.dim
        """
        self.start = np.array(start)
        self.goal = np.array(start)
        # print("[LI] SETTING START:", self.start, self.goal)

    def set_goal(self, goal):
        """
        Takes a requested (absolute) goal and updates internal parameters for next interpolation step

        Args:
            np.array: Requested goal (absolute value). Should be same dimension as self.dim
        """
        goal = np.array(goal)
        # First, check to make sure requested goal shape is the same as self.dim
        if goal.shape[0] != self._ndim:
            print("Requested goal: {}".format(goal))
            raise ValueError(
                "LinearInterpolator: Input size wrong for goal; got {}, needs to be {}!".format(goal.shape[0], self._ndim)
            )

        # Update start and goal
        self.start = np.array(self.goal)
        self.goal = np.array(goal)

        # print("[LI] SETTING START AND GOAL:", self.start, self.goal)
        # Reset interpolation steps
        self._step = 0

    def get_interpolated_goal(self):
        """
        Provides the next step in interpolation given the remaining steps.

        NOTE: If this interpolator is for orientation, it is assumed to be receiving either euler angles or quaternions

        Returns:
            np.array: Next position in the interpolated trajectory
        """
        # Grab start position
        x = np.array(self.start)

        # Calculate the desired next step based on remaining interpolation steps
        if self._ori_interpolate is not None:
            # This is an orientation interpolation, so we interpolate linearly around a sphere instead
            goal = np.array(self.goal)
            if self._ori_interpolate == "euler":
                # this is assumed to be euler angles (x,y,z), so we need to first map to quat
                x = mat2quat(euler2mat(x))
                goal = mat2quat(euler2mat(self.goal))

            # Interpolate to the next sequence
            x_current = quat_slerp(x, goal, fraction=(self._step + 1) / self.total_steps)
            if self._ori_interpolate == "euler":
                # Map back to euler
                x_current = mat2euler(quat2mat(x_current))
        else:
            # This is a normal interpolation
            dx = (self.goal - x) / (self.total_steps - self._step)
            x_current = x + dx

        # Increment step if there's still steps remaining based on ramp ratio
        if self._step < self.total_steps - 1:
            self._step += 1
        # print("[LI] START:", self.start)
        # print("[LI] GOAL:", self.goal)
        # print("[LI] CURRENT:", x_current)

        # Return the new interpolated step
        return x_current
        