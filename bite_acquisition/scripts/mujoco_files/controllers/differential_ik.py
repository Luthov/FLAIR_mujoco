import numpy as np
import mujoco

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
class DiffIKController(BaseController):
    def __init__(self, 
                 sim, 
                 eef_name, 
                 joint_indexes, 
                 actuator_range, 
                 control_dim,
                 ramp_ratio=0.5, 
                 damping_ratio=1e-4,
                 control_freq=20, 
                 policy_freq=10,
                 control_ori=True,
                 control_delta=True,
                 **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms
                 ):
        
        self._control_freq = control_freq
        
        super().__init__(sim, eef_name, joint_indexes, actuator_range, ramp_ratio, control_freq, policy_freq)

        # Determine whether this is pos quat or just pos
        self._use_ori = control_ori

        # Determine whether we want to use delta or absolute values as inputs
        self._use_delta = control_delta

        # Set control dimension
        self._control_dim = control_dim
        self._name_suffix = "POSE" if self._use_ori else "POSITION"

        # Should be in (0, 1], smaller values mean less sensitivity.
        self.user_sensitivity = 1.0

        # Gains
        self._kp = damping_ratio * np.eye(control_dim)

        # Interpolator
        self.interpolator_pos = LinearInterpolator(3, control_freq, policy_freq, ramp_ratio)
        if self._use_ori:
            self.interpolator_ori = LinearInterpolator(4, control_freq, policy_freq, ramp_ratio, ori_interpolate="quat")
        else:
            self.interpolator_ori = None
        
        # Initialize goals based on initial pos / ori
        self.goal_ori = np.array(self.initial_ee_ori_mat)
        self.goal_pos = np.array(self.initial_ee_pos)

        self.ori_ref = None
    
    def set_start(self):
        """
        Sets the start state of the controller
        """
        self.update(force=True)
        
        self.reset_goal()
    
    def set_goal(self, action, set_pos=None, set_ori=None):
        """
        Sets goal based on input @action
        """
        action = np.array(action)

        # Update robot arm state
        self.update()

        # If using deltas:
        if self._use_delta:
            # NOTE: use_delta = True not tested yet...
            # Remove clipping
            dpos = action[:3]
            dquat = axisangle2quat(action[3:])

            set_pos = dpos * self.user_sensitivity + self.ee_pos

            set_ori = set_goal_orientation(
                action[3:], self.ee_ori_mat, orientation_limit=None, set_ori=quat2mat(dquat)
            )
        
        # Interpret actions as absolute IK goals
        else:
            set_pos = action[:3]
            set_ori = quat2mat(axisangle2quat(action[3:]))

        self.goal_pos = set_pos
        self.goal_ori = set_ori

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)    # ref is the current orientation at the start

            # Setting orientation error as goal
            self.interpolator_ori.set_goal(
                mat2quat(self.goal_ori)
            )   
            self.relative_ori = np.zeros(3)     # relative orientation error

        # print("[set goal] goal_pos:", self.goal_pos)
        # print("[set goal] goal_ori:", mat2quat(self.goal_ori))
    
    def run_controller(self):
        """
        Calculates the torques required to reach the desired pose

        Returns:
             np.array: Command torques
        """
        # Update robot arm state
        self.update()

        # Get desired position and orientation
        desired_pos = self.interpolator_pos.get_interpolated_goal()

        if self.interpolator_ori is not None:
            self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)
            desired_ori = self.interpolator_ori.get_interpolated_goal() # quat
            desired_ori_mat = quat2mat(desired_ori)
        else:
            desired_ori_mat = np.array(self.goal_ori)
            desired_ori = mat2quat(desired_ori_mat)

        # Calculate position error
        print(">> Desired pos:", desired_pos)
        pos_error = desired_pos - self.ee_pos

        # Calculate orientation error
        print("eef orientation:", mat2quat(self.ee_ori_mat))
        ori_error = orientation_error(desired_ori_mat, self.ee_ori_mat) # error in axis angle
        print("ori_error original (axis angle):", ori_error)
        print("ori_error original (quat):", axisangle2quat(ori_error))

        twist = np.zeros(6)

        # # TODO: check this
        # # kevin zakka orientation error calculation
        # # theirs uses jac site, we are using jacbody -> need to check
        # print(">> Desired quat:", desired_ori)
        # eef_quat = np.zeros(4)
        # quat_conj = np.zeros(4)
        # error_quat = np.zeros(4)
        # jac = np.zeros((6, self._sim._model.nv))
        # mujoco.mju_mat2Quat(eef_quat, self.ee_ori_mat.flatten())
        # mujoco.mju_negQuat(quat_conj, eef_quat)
        # mujoco.mju_mulQuat(error_quat, desired_ori, quat_conj)
        # print("quat error kevin zakka:", error_quat)
        # mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
        # print("ori error kevin zakka:", twist[3:])

        twist[:3] = pos_error 
        twist[3:] = ori_error

        # get the jacobian
        dq = self.J_full.T @ np.linalg.solve(self.J_full @ self.J_full.T + self._kp, twist)

        q = self._sim.qpos.copy()
        # append 0s to dq to match q shape
        dq = np.append(dq, np.zeros(q.shape[0] - dq.shape[0]))
        mujoco.mj_integratePos(self._sim._model, q, dq, 1.0)    # 1.0 = integration dt

        # always run superclass call for any cleanups at the end
        super().run_controller()

        return q[:6]
    
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
                mat2quat(self.goal_ori)
            )  
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

        
    @property
    def name(self):
        return "DIFFIK_" + self._name_suffix
