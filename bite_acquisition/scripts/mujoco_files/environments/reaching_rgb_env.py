import math
import numpy as np

from gymnasium import spaces

try:
    from simulator.mujoco_simulator import MujocoSimulator
    from robots.xarm6 import XArm6Robot
    from environments.arm_base import ArmBaseEnv
except:
    from feeding_mujoco.simulator.mujoco_simulator import MujocoSimulator
    from feeding_mujoco.robots.xarm6 import XArm6Robot
    from feeding_mujoco.environments.arm_base import ArmBaseEnv

class XArmReachRGBEnv(ArmBaseEnv):
    """
    ## Description
    This environment is a simple reaching task for the xArm6 robot. 
    The robot has to reach a randomly placed target in 3D space.

    ## Action Space
    The action space is a `Box(-1, 1, (3,), float64)`. An action represents the desired end effector position in 3D space.

    | Num | Action                 | Control Min | Control Max | Unit |
    | --- | ---------------------- | ----------- | ----------- | ---- |
    | 0   | x-coordinate of eef    | -1          | 1           | m    |
    | 1   | y-coordinate of eef    | -1          | 1           | m    |
    | 2   | z-coordinate of eef    | -1          | 1           | m    |

    ## Observation Space
    The observation space consists of the following parts (in order):
    - eef_pose (3 elements): The 3D position of the end effector
    - target_pose (3 elements): The 3D position of the target

    The observation space is a `Dict` with the following structure:
    ```python
    spaces.Dict(
        {
            "eef_pose": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64),
            "image": Box(0, 255, (W, H, C), uint8),
        }
    )
    # where W=width, H=height, C=channels, or
    spaces.Dict(
        {
            "eef_pose": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64),
            "target_pose": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64),
        }
    )
    ```
    `eef_pose`:
    | Num | Action                         | Control Min | Control Max | Unit |
    | --- | ------------------------------ | ----------- | ----------- | ---- |
    | 0   | x-coordinate of current eef    | -1          | 1           | m    |
    | 1   | y-coordinate of current eef    | -1          | 1           | m    |
    | 2   | z-coordinate of current eef    | -1          | 1           | m    |


    ## Rewards
    The reward is calculated as follows:
    - reward_dist: -np.linalg.norm(eef_pose - target_pose)
    - reward_success: 10.0 if the target is reached, 0.0 otherwise
    - reward = reward_dist + reward_success

    `info` contains the `reward_dist` term.

    ## Starting State
    The robot starts at a fixed position in the environment.
    The position of the goal is randomly sampled at at least 5cm away from the robot's initial position.

    ## Episode Termination
    The episode terminates when the robot reaches the target position.

    ## Episode Truncation
    The default duration of an episode is 50 steps.
    
    Args:
        ArmBaseEnv (_type_): _description_

    Returns:
        _type_: _description_
    """

    min_goal_dist = 0.1
    goal_tresh = 0.025

    def __init__(self,
                 model_path,
                 sim_timestep=0.01,
                 controller_config=None,
                 control_freq = 50,
                 policy_freq = 25,
                 render_mode=None,
                 camera=-1,
                 obs_mode=None,
                 seed=None
                 ):
        
        super().__init__(model_path=model_path,
                         sim_timestep=sim_timestep,
                         controller_config=controller_config,
                         control_freq=control_freq,
                         policy_freq=policy_freq,
                         render_mode=render_mode,
                         camera=camera,
                         obs_mode=obs_mode)
        

        # Define observation space as xyz of end effector
        self.observation_space = self._get_basic_observation_space()
        
        # If obs_mode is not specified, we add the target pose to the observation space
        if self.obs_mode is None:
            self.observation_space.spaces["target"] = spaces.Box(low=-1.0,
                                                                high=1.0,
                                                                shape=(3,),
                                                                dtype=np.float64)
        
        # Get the keys in the observation space
        self._obs_keys = list(self.observation_space.spaces.keys())
                                                    
        # We define a continuous action space - action space should be normalized
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(3,),
                                       dtype=np.float64)


    def _set_random_target_pose(self):
        """
        Sets a random goal position for the robot to reach.

        Returns:
            ndarray: 3D position of the target pose
        """
        # get current eef pose
        eef_pose = self._robot.get_eef_pose()[:3]
        
        # Generate random target pose until it is at least 5cm away from eef_pose
        while True:
            x = eef_pose[0] + (self.np_random.random(size=1, dtype=np.float64)[0] * 0.1 + 0.1)
            y = eef_pose[1] + (self.np_random.random(size=1, dtype=np.float64)[0] * 0.1)
            z = eef_pose[2] + (self.np_random.random(size=1, dtype=np.float64)[0] * 0.05)
            
            target_pose = np.array([x, y, z])
            
            distance = np.linalg.norm(target_pose - eef_pose)
            
            if distance >= self.min_goal_dist:
                break
        
        return target_pose

    # We can assume step will not be called before reset has been called
    def reset(self, seed=None, options=None):
        """
        Resets the environment and simulation to its initial state.

        Args:
            seed (int, optional): Seed to initialise environment's PRNG (np_random). Defaults to None.
            options (dict, optional): Additional information to speciy how the environment is reset. Defaults to None.

        Returns:
            observation (ObsType): Observation of the initial state.
            info (dict): Auxiliary information complementing `observation`. 
        """
        # Resets simulation and robot to initial state
        super().reset(seed=seed)

        # Sample target location randomly
        self._target_pose = self._set_random_target_pose()

        # render the target pose
        self._sim.add_target_to_viewer(self._target_pose, target_size=0.01)        

        observation = self._get_obs()

        info = self._get_info(action=np.array([0.0, 0.0, 0.0]), 
                              terminated=False)

        self.render()

        return observation, info

    def step(self, action):        
        """
        Steps the environment with the given action.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barto Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
        """
        if self.timestep == 0:
            # Set the start state of the controller
            self._robot._controller.set_start()
            
        # Increment the timestep (should follow controller timestep)
        self.timestep += 1

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(int(self._control_timestep / self._model_timestep)):
            self._sim.forward()
            self._pre_action(action, policy_step)
            self._sim.step()
            policy_step = False

        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self._control_timestep

        if self.viewer is not None:
            self.viewer.sync()
                
        reward, terminated, info = self._post_action(action)
        
        observations = self._get_obs()
        
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `gym.make`
        return observations, reward, terminated, False, info
    
    def _get_obs(self):
        """
        Returns the observations from the environment.
        """
        
        self._sim.add_target_to_viewer(self._target_pose, target_size=0.01)


        observation = {}
        for key in self._obs_keys:

            if key == "eef_pose":
                observation[key] = self._robot.get_eef_pose()[:3]
            elif key == "target":
                observation[key] = self._target_pose
            elif key == "image":
                image_array = self.render_observation()
                observation[key] = image_array
            else:
                raise ValueError(f"Invalid observation key: {key}")

        return observation
    
    def _get_info(self, action, terminated):
        """
        Returns auxiliary information returned by step and reset.
        """
        return {
            "reward_dist": -np.linalg.norm(
                self._robot.get_eef_pose()[:3] - self._target_pose
            ),
            # "reward_ctrl": -np.square(action).sum(),
            "is_success": terminated,
        }
    
    def _pre_action(self, action, policy_step=False):
        """
        Do any preprocessing before taking an action.
        Args:
            action (ndarray): Action to execute within the environment
            policy_step (bool): Whether this current loop is an actual policy step or internal sim update step
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self._robot._controller.name == "OSC_POSE" and self._robot._controller._use_ori:
            action = np.concatenate([action, np.zeros(3)])
        elif self._robot._controller.name == "IK_POSE":
            action = np.concatenate([action, np.zeros(3)])

        # Verify that the action is the correct dimension
        assert (
                action.shape[0] == self._robot._control_dim
            ),"Control action dimension must match robot's control dimension. Expected: {}, Got: {}".format(
                self._robot._control_dim, action.shape[0]
        )

        self._robot.control(action, policy_step=policy_step)
    
    def _post_action(self, action):
        """
        Do any postprocessing after taking an action.
        Args:
            action (ndarray): Action to execute within the environment
        """
        # Get the reward
        reward = self._get_reward(action)

        # Check if the episode is done
        terminated = self._check_done()

        # Empty dict to be filled with info
        info = self._get_info(action, terminated)

        return reward, terminated, info

    def _check_done(self):
        """
        Check if the episode is done.
        """
        if np.linalg.norm(self._robot.get_eef_pose()[:3] -  self._target_pose) < self.goal_tresh:
            done = True
        else:
            done = False

        return done

    def _get_reward(self, action):
        """
        Returns the reward for the current state.
        """
        # Distance reward
        reward_dist = -np.linalg.norm(self._robot.get_eef_pose()[:3] -  self._target_pose)

        # Control reward
        # reward_ctrl = -np.square(action).sum()

        # Success reward
        if self._check_done():
            reward_success = 10.0
        else:
            reward_success = 0.0

        
        reward = reward_dist + reward_success
        return reward
      
    def number_of_particles_scooped(self):
        """
        Returns the number of particles scooped by the spoon.

        Returns:
            int: Number of particles scooped
        """
        # Get force exerted on spoon in z direction
        # Divide by particle mass (0.001 kg) and by gravity (9.81 m/s^2)
        spoon_id = self._sim.get_body_id_from_name("spoon")
        particle_id = spoon_id + 100    # Assume we have at least 100 particles
        body_mass = self._sim.get_body_mass(particle_id)
        return math.ceil(self._robot.get_contact_force("spoon")[2]/(body_mass*self._sim._gravity[2]))
