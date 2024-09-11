from abc import ABC, abstractmethod
from collections.abc import Iterable

import os
import json
import mujoco
import numpy as np

# REFERENCES:
# https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/controllers/base_controller.py

def load_controller_config(custom_fpath=None, default_controller=None):
    """
    Utility function that loads the desired controller and returns the loaded configuration as a dict

    If @default_controller is specified, any value inputted to @custom_fpath is overridden and the default controller
    configuration is automatically loaded. See specific arg description below for available default controllers.

    Args:
        custom_fpath (str): Absolute filepath to the custom controller configuration .json file to be loaded
        default_controller (str): If specified, overrides @custom_fpath and loads a default configuration file for the
            specified controller.
            Choices are: {"JOINT_POSITION", "OSC_POSITION", "OSC_POSE", "IK_POSE"}

    Returns:
        dict: Controller configuration

    Raises:
        AssertionError: [Unknown default controller name]
        AssertionError: [No controller specified]
    """
    # First check if default controller is not None; if it is not, load the appropriate controller
    if default_controller is not None:

        # Assert that requested default controller is in the available default controllers
        from controllers import ALL_CONTROLLERS

        assert (
            default_controller in ALL_CONTROLLERS
        ), "Error: Unknown default controller specified. Requested {}, " "available controllers: {}".format(
            default_controller, list(ALL_CONTROLLERS)
        )

        # Store the default controller config fpath associated with the requested controller
        custom_fpath = os.path.join(
            os.path.dirname(__file__), "..", "controllers/config/{}.json".format(default_controller.lower())
        )

    # Assert that the fpath to load the controller is not empty
    assert custom_fpath is not None, "Error: Either custom_fpath or default_controller must be specified!"

    # Attempt to load the controller
    try:
        with open(custom_fpath) as f:
            controller_config = json.load(f)
    except FileNotFoundError:
        print("Error opening controller filepath at: {}. " "Please check filepath and try again.".format(custom_fpath))

    # Return the loaded controller
    return controller_config

class BaseController(ABC):
    """
    General controller interface.

    Requires reference to mujoco simulation object, robot name, 
    eef_name of robot and relevant joint indexes of the robot.

     Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range
    """

    def __init__(
            self,
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
            ramp_ratio,
            control_freq=20,
            policy_freq=10
    ):
        
        # Actuator range
        self._actuator_min = actuator_range[0]
        self._actuator_max = actuator_range[1]

        # Attributes for scaling / clipping inputs to outputs
        self._action_scale = None
        self._action_input_transform = None
        self._action_output_transform = None


        # Private property attributes
        self._control_dim = None
        self._output_min = None
        self._output_max = None
        self._input_min = None
        self._input_max = None

        # mujoco simulator state
        self._sim = sim
        self._eef_name = eef_name
        self._eef_id = self._sim.get_body_id_from_name(self._eef_name)
        self._joint_index = joint_indexes["joints"]
        self._qpos_index = joint_indexes["qpos"]
        self._qvel_index = joint_indexes["qvel"]

        # Robot states
        self.ee_pos = None
        self.ee_ori_mat = None
        self.joint_pos = None
        self.joint_vel = None

        # Dynamics and kinematics
        self.J_pos = None
        self.J_ori = None
        self.J_full = None
        self.mass_matrix = None

        # Joint dimension
        self.joint_dim = len(joint_indexes["joints"])

        # Torques being outputted by the controller
        self.torques = None

        # Update flag to prevent redundant update calls
        self._new_update = True

        # Move forward one timestep to propagate updates before taking first update
        self._sim.forward()
        
        self.update()
        self.initial_joint_pos = self.joint_pos
        self.initial_ee_pos = self.ee_pos
        self.initial_ee_ori_mat = self.ee_ori_mat

    @abstractmethod
    def run_controller(self):
        """
        Runs the controller to generate torques to be applied to the robot. Must be implemented by child class
        Additionally, resets the self._new_update flag so that the next self.update call will occur
        """
        self._new_update = True

    def scale_action(self, action):
        """
        Clips @action to be within self.input_min and self.input_max, and then re-scale the values to be within
        the range self.output_min and self.output_max

        Args:
            action (Iterable): Actions to scale

        Returns:
            np.array: Re-scaled action
        """

        if self._action_scale is None:
            self._action_scale = abs(self._output_max - self._output_min) / abs(self._input_max - self._input_min)
            self.action_output_transform = (self._output_max + self._output_min) / 2.0
            self.action_input_transform = (self._input_max + self._input_min) / 2.0
        action = np.clip(action, self._input_min, self._input_max)
        transformed_action = (action - self.action_input_transform) * self._action_scale + self.action_output_transform

        return transformed_action
    
    def update(self, force=False):
        """
        Updates the state of the robot arm, including end effector pose / orientation / velocity, joint pos/vel,
        jacobian, and mass matrix. By default, since this is a non-negligible computation, multiple redundant calls
        will be ignored via the self.new_update attribute flag. However, if the @force flag is set, the update will
        occur regardless of that state of self.new_update. This base class method of @run_controller resets the
        self.new_update flag

        Args:
            force (bool): Whether to force an update to occur or not
        """

        # Only run update if self._new_update or force flag is set
        if self._new_update or force:
            self._sim.forward()
            self.ee_pos = np.array(self._sim.get_body_pos(self._eef_id))
            self.ee_ori_mat = np.array(self._sim.get_body_mat(self._eef_id).reshape((3, 3)))
            
            self.ee_pos_vel = np.array(self._sim.get_body_xvelp(self._eef_id))
            self.ee_ori_vel = np.array(self._sim.get_body_xvelr(self._eef_id))

            self.joint_pos = np.array(self._sim.qpos[self._qpos_index])
            self.joint_vel = np.array(self._sim.qvel[self._qvel_index])

            self.J_pos = np.array(self._sim.get_body_jacp(self._eef_id).reshape((3, -1))[:, self._qvel_index])
            self.J_ori = np.array(self._sim.get_body_jacr(self._eef_id).reshape((3, -1))[:, self._qvel_index])
            self.J_full = np.array(np.vstack([self.J_pos, self.J_ori]))

            mass_matrix = np.ndarray(shape=(self._sim.nv, self._sim.nv), dtype=np.float64, order="C")
            mujoco.mj_fullM(self._sim._model, mass_matrix, self._sim.qM)
            mass_matrix = np.reshape(mass_matrix, (len(self._sim.qvel), len(self._sim.qvel)))
            self.mass_matrix = mass_matrix[self._qvel_index, :][:, self._qvel_index]

            # Clear self.new_update
            self._new_update = False
    
    def update_initial_joints(self, initial_joint_pos):
        """
        Updates the internal attribute self.initial_joints. This is useful for updating changes in controller-specific
        behavior, such as with OSC where self.initial_joints is used for determine nullspace actions

        This function can also be extended by subclassed controllers for additional controller-specific updates

        Args:
            initial_joints (Iterable): Array of joint position values to update the initial joints
        """
        self.initial_joint_pos = np.array(initial_joint_pos)
        self.update(force=True)
        self.initial_ee_pos = self.ee_pos
        self.initial_ee_ori_mat = self.ee_ori_mat
    
    def clip_torques(self, torques):
        """
        Clips the torques to be within the actuator limits

        Args:
            torques (Iterable): Torques to clip

        Returns:
            np.array: Clipped torques
        """
        return np.clip(torques, self._actuator_min, self._actuator_max)
    
    def reset_goal(self):
        """
        Resets the goal -- usually by setting to the goal to all zeros, but in some cases may be different (e.g.: OSC)
        """
        raise NotImplementedError
    
    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array

        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to env.sim.data.actuator_force

        Returns:
            np.array: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")

        # Check if input is an Iterable, if so, we simply convert the input to np.array and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums

    @staticmethod
    def joint_sum(a, b):
        """
        Sums two joint position vectors, @a and @b
        """
        return ((a+b)+np.pi)%(2.*np.pi)-np.pi

    @property
    def actuator_limits(self):
        """
        Torque limits for this controller

        Returns:
            2-tuple:

                - (np.array) minimum actuator torques
                - (np.array) maximum actuator torques
        """
        return self._actuator_min, self._actuator_max
    
    @property
    def control_limits(self):
        """
        Limits over this controller's action space, which defaults to input min/max
        """
        return self._input_min, self._input_max
    
    @property
    def torque_compensation(self):
        """
        Torque compensation for this controller

        Returns:
            np.array: Torque compensation
        """
        return self._sim.get_joint_torque_compensation(self._joint_index)
        
    
    @property
    def name(self):
        """
        Name of this controller

        Returns:
            str: controller name
        """
        raise NotImplementedError

        

