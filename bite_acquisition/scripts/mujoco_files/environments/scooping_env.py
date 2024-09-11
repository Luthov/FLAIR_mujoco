import math
import numpy as np

from gymnasium import spaces

try:
    from simulator.mujoco_simulator import MujocoSimulator
    from simulator.mujoco_modder import ParticleModder
    from robots.xarm6 import XArm6Robot
    from environments.arm_base import ArmBaseEnv
    from planners.scooping_dmp import ScoopingDMPPlanner 
    from utils.transform_utils import quat2euler, quat_distance, quat_error, quat2axisangle, quat2mat, mat2quat
    from utils.scooping_trajectory import SCOOPING_JOINT_TRAJ, SCOOPING_CART_TRAJ
except:
    from feeding_mujoco.simulator.mujoco_simulator import MujocoSimulator
    from feeding_mujoco.simulator.mujoco_modder import ParticleModder
    from feeding_mujoco.robots.xarm6 import XArm6Robot
    from feeding_mujoco.environments.arm_base import ArmBaseEnv
    from feeding_mujoco.planners.scooping_dmp import ScoopingDMPPlanner 
    from feeding_mujoco.utils.transform_utils import quat2euler, quat_distance, quat_error, quat2axisangle, quat2mat, mat2quat
    from feeding_mujoco.utils.scooping_trajectory import SCOOPING_JOINT_TRAJ, SCOOPING_CART_TRAJ

class XArmScoopEnv(ArmBaseEnv):
    """
    ## Description
    This environment is a scooping task for the xArm6 robot. 
    The robot has to scoop a desired amount of particles/food.

    ## Action Space
    The action space is a `Box(-1, 1, (3,), float64)`. An action represents the desired end effector position in 3D space.

    | Num | Action                                | Control Min | Control Max | Unit |
    | --- | ------------------------------------- | ----------- | ----------- | ---- |
    | 0   | change in x-coordinate of DMP goal    | -1          | 1           | m    |
    | 1   | change in x-coordinate of DMP goal    | -1          | 1           | m    |
    | 2   | change in x-coordinate of DMP goal    | -1          | 1           | m    |

    ## Observation Space
    The observation space consists of the following parts (in order):
    - eef_pose (3 elements): The 3D position of the end effector
    - image : The image of the environment from the camera
    - ft_sensor (6 elements): The force and torque readings from the force torque sensor
    - target_amount (1 element): The amount of particles to be scooped
    - actual_amount (1 element): The amount of particles actually scooped

    The observation space is a `Dict` with the following structure:
    ```python
    spaces.Dict(
        {
            "eef_pose": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64),
            "image": Box(0, 255, (W, H, C), uint8),
            "ft_sensor": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "target_amount": spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.int64),
            "actual_amount": spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.int64),
        }
    )
    # where W=width, H=height, C=channels, or
    ```
    `eef_pose`:
    | Num | Observation                    | Control Min | Control Max | Unit |
    | --- | ------------------------------ | ----------- | ----------- | ---- |
    | 0   | x-coordinate of current eef    | -1          | 1           | m    |
    | 1   | y-coordinate of current eef    | -1          | 1           | m    |
    | 2   | z-coordinate of current eef    | -1          | 1           | m    |

    `ft_sensor`:
    | Num | Observation                    | Control Min | Control Max | Unit |
    | --- | ------------------------------ | ----------- | ----------- | ---- |
    | 0   | force in x-direction           | -inf        | inf         | N    |
    | 1   | force in y-direction           | -inf        | inf         | N    |
    | 2   | force in z-direction           | -inf        | inf         | N    |
    | 3   | torque in x-direction          | -inf        | inf         | Nm   |
    | 4   | torque in y-direction          | -inf        | inf         | Nm   |
    | 5   | torque in z-direction          | -inf        | inf         | Nm   |

    `target_amount`:
    | Num | Observation                    | Control Min | Control Max | Unit |
    | --- | ------------------------------ | ----------- | ----------- | ---- |
    | 0   | target amount of particles     | 0           | 50          | -    |

    `actual_amount`:
    | Num | Observation                    | Control Min | Control Max | Unit |
    | --- | ------------------------------ | ----------- | ----------- | ---- |
    | 0   | actual amount of particles     | 0           | 50          | -    |

    ## Rewards
    The reward is calculated as follows:
    - reward_amount: 10.0 * (target_amount - actual_amount)
    - reward = reward_amount

    `info` contains the `reward_amount` term.

    ## Starting State
    The robot starts at a fixed position in the environment that is the start of the scooping trajectory.
    The target amount to be scooped is randomly set between 0 and 20 particles.

    ## Episode Termination
    The episode terminates when the robot reaches the end of the trajectory.

    ## Episode Truncation
    The default duration of an episode is 50 steps, i.e. it should always terminate first.
    
    Args:
        ArmBaseEnv (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self,
                 model_path,
                 sim_timestep=0.01,
                 controller_config=None,
                 control_freq = 50,
                 policy_freq = 25,
                 render_mode=None,
                 camera=-1,
                 obs_mode="depth_array",
                 action_mode = "fz",
                 reward_type="sparse",
                 particle_size=0.005,
                 particle_amount_ratio=0.7,
                 particle_damping="random",
                 target_amount="random",
                 randomize_poses = True,
                 seed=0,
                 ):
        
        super().__init__(model_path=model_path,
                         sim_timestep=sim_timestep,
                         controller_config=controller_config,
                         control_freq=control_freq,
                         policy_freq=policy_freq,
                         render_mode=render_mode,
                         camera=camera,
                         obs_mode=obs_mode)
        
        self.random_state = np.random.RandomState(seed)
        self._particle_modder = ParticleModder(self._sim,
                                               random_state=self.random_state,
                                               )
        
        # Define observation space as xyz of end effector
        # assert self.obs_mode is not None, "Observation mode must be either rgb_array, depth_array, or rgbd_array for scooping."
        self.observation_space = self._get_basic_observation_space()
        
        # Add the following to observation space
        # 1. FT sensor readings 
        # 2. Target amount to be scooped (assume 50 particles as max)
        # 3. Actual amount scooped
        self.observation_space.spaces["ft_sensor"] = spaces.Box(low=-np.inf,
                                                                high=np.inf,
                                                                shape=(6,),
                                                                dtype=np.float64)
        self.observation_space.spaces["target_amount"] = spaces.Box(low=0,
                                                                    high=50,
                                                                    shape=(1,),
                                                                    dtype=np.int64)
        self.observation_space.spaces["actual_amount"] = spaces.Box(low=0,
                                                                    high=50,
                                                                    shape=(1,),
                                                                    dtype=np.int64)
        self.observation_space.spaces["pred_food_pose"] = spaces.Box(low=-1.0,
                                                                     high=1.0,
                                                                     shape=(3,),
                                                                     dtype=np.float64)
        # Overwrite eef_pose to be 7D
        self.observation_space.spaces["eef_pose"] = spaces.Box(low=-1.0,
                                                               high=1.0,
                                                               shape=(7,),
                                                               dtype=np.float64)
        
        
        # Include other states
        self.observation_space.spaces["particle_amount"] = spaces.Box(low=-np.inf,
                                                                      high=np.inf,
                                                                      shape=(1,),
                                                                      dtype=np.float64)
        
        self.observation_space.spaces["particle_damping"] = spaces.Box(low=-np.inf,
                                                                       high=np.inf,
                                                                       shape=(1,),
                                                                       dtype=np.float64)
        
        self.observation_space.spaces["particle_size"] = spaces.Box(low=-np.inf,
                                                                    high=np.inf,
                                                                    shape=(1,),
                                                                    dtype=np.float64)
        
        
        # Get the keys in the observation space
        self._obs_keys = list(self.observation_space.spaces.keys())
                                                    
        # We define a continuous action space - action space should be normalized
        self._action_mode = action_mode
        if self._action_mode == "delta_weight":
            self.action_space = spaces.Box(low=-1.0,
                                        high=1.0,
                                        shape=(1,),
                                        dtype=np.float64)
        elif self._action_mode == "delta_goal":
            self.action_space = spaces.Box(low=-1.0,
                                        high=1.0,
                                        shape=(3,),
                                        dtype=np.float64)
        elif self._action_mode == "fz":
            self.action_space = spaces.Box(low=-1.0,
                                        high=1.0,
                                        shape=(1,),
                                        dtype=np.float64)
        elif self._action_mode == "fz_discrete":
            self.action_space = spaces.Discrete(5)
        
        # Check that controller is valid
        valid_controllers = ["IK_POSE", "JOINT_POSITION"]
        assert self._robot._controller.name in valid_controllers, "Controller must be IK_POSE or JOINT_POSITION for scooping task"
         
        # Initialize variables
        self._reward_type = reward_type
        self._particle_size = particle_size
        self._particle_amount_ratio = particle_amount_ratio
        self._particle_damping = particle_damping
        self._randomize_poses = randomize_poses
        self._particle_body_id = self._sim.get_body_id_from_name("B0_0_0")  
        self._target_amount_config = target_amount
        self._target_amount_state = None

        self._randomization_dict = {
            "size": self._particle_size,
            "amount_ratio": self._particle_amount_ratio,
            "damping": self._particle_damping,
            "rgb": [1.0, 1.0, 1.0],
            "mass": 0.001,
        }
        self._offset_pose = np.zeros(3)


        # Set initial pose to be outside the possible area of container
        self._set_arm_initial_pose([1.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Set food pose based on container pose
        self._container_body_id = self._sim.get_body_id_from_name("container")
        _default_container_pose = self._sim.get_body_pos(self._container_body_id)
        self._default_food_pose_offset = np.array([0.0, 0.0, 0.04])
        self._default_food_pose = _default_container_pose + self._default_food_pose_offset
        self._food_pose = self._default_food_pose.copy()

        # DMP Planner
        self.scooping_planner = ScoopingDMPPlanner(dt=1/policy_freq)
        self.scooping_planner.update_food_pose(self._food_pose)

    def randomize_env(self):
        # Randomize position
        if self._randomize_poses:
            self._offset_pose = self._particle_modder.randomize_position()
            self._food_pose = self._default_food_pose + self._offset_pose

        # Randomize particles
        if self._particle_size == "random":
            size = None
        else:
            size = float(self._particle_size)
        
        if self._particle_amount_ratio == "random":
            amount_ratio = self._particle_modder.random_state.uniform(0.5, 1.0)
        else:
            amount_ratio = float(self._particle_amount_ratio)

        self._randomization_dict = self._particle_modder.randomize_particles(
                                    size=size,
                                    amount_ratio = amount_ratio,
                                    rgb=None,
                                    mass=0.001, # don't randomize mass for now
                                    damping=self._particle_damping,
                                )
        
        print("-----------------------------")
        print(f">> Offset pose: {self._offset_pose} | Damping: {self._randomization_dict['damping']:.4f}")
        print(f">> Part size: {self._randomization_dict['size']} | Part ratio: {self._randomization_dict['amount_ratio']:.3f}")
        
    def reset_start_pose(self):
        # Set initial pose
        start_pose = self.scooping_planner.get_eef_pose_from_spoon_pose(
                        self.scooping_planner._dmp.y0,
                        self.scooping_planner._spoon_offset
                        )
        # print(">> Des start pose:", start_pose)
        
        if self._robot._controller.name == "IK_POSE":
            initial_joint_pos = None
            delta_trans = 100
            attempts = 0
            
            # Use pybullet to calculate IK
            initial_joint_pos=[1.3103,  0.1289, -1.2169, -0.9229,  1.6615,  1.6824]
            while delta_trans > 0.015 and attempts < 20:
                self._robot._controller.sync_ik_robot(joint_positions=initial_joint_pos)
                initial_joint_pos = self._robot._controller.inverse_kinematics(start_pose[:3], 
                                                                                start_pose[3:],
                                                                                )
                
                # print(">> IK start pose:", initial_joint_pos, "attempts:", attempts)
                # # check fk of mujoco
                pose_out = self._robot.get_eef_fk(initial_joint_pos)
                # print(">> FK of mujoco: ", pose_out)

                # Calculate distance between actual and desired start pose
                delta_trans = np.linalg.norm(start_pose[:3] - pose_out[:3])
                # dot_product = np.dot(start_pose[3:], pose_out[3:])
                # delta_quat = 2 * np.arccos(np.clip(np.abs(dot_product), -1.0, 1.0))
                
                attempts += 1
            
            if attempts >= 20:
                print(f">> WARN: IK solution after 20 attempts is {delta_trans:.4f} away from desired start.")

            # self._set_arm_initial_pose(initial_joint_pos)
            self._robot.hard_set_joint_positions(initial_joint_pos, self._robot._arm_joint_ids)
            self._sim.forward()
            # print(">> forwarded eef pose:", self._robot.get_eef_pose())

    def _set_random_target_amount(self):
        """
        Sets a random target amount for the robot to scoop. 

        Returns:
            int: The number of particles to be scooped
        """

        if self._target_amount_config == "random":
            target_amount_state = self.random_state.randint(0, 3)
            # Discretized target amount in state
        elif self._target_amount_config == "all":
            if self._target_amount_state is None:
                target_amount_state = 0
            else:
                target_amount_state = self._target_amount_state
                target_amount_state += 1
                if target_amount_state >= 3:
                    target_amount_state = 0
        else:
            target_amount_state = self._target_amount_config
 
        return target_amount_state

    # We can assume step will not be called before reset has been called
    def reset(self, seed=None, options={"wait": True}):
        """
        Resets the environment and simulation to its initial state.

        Args:
            seed (int, optional): Seed to initialise environment's PRNG (np_random). Defaults to None.
            options (dict, optional): Additional information to speciy how the environment is reset. Defaults to None.

        Returns:
            observation (ObsType): Observation of the initial state.
            info (dict): Auxiliary information complementing `observation`. 
        """
        # Set random particle size - must be done before super().reset()
        self.randomize_env()

        # Resets simulation and robot to initial state
        super().reset(seed=seed, options=options)

        # Update food pose - Find max z height of particles
        z_height = self._particle_modder.get_z_height()
        if z_height < 0.12:
            z_height = 0.12
            print("Warning: z height of particles too low. Setting to 0.12")
        self._food_pose[2] = z_height

        # Sample target amount randomly
        self._target_amount_state = self._set_random_target_amount()  
        # print("Target amount:", self._target_amount_state)

        # Reset DMP
        self.scooping_planner.reset()
        self.scooping_planner.update_food_pose(self._food_pose)

        self.reset_start_pose()

        ## Display target pose (only for debugging)
        # print("food pose:", self._food_pose)
        # self._sim.add_target_to_viewer(self._food_pose)

        observation = self._get_obs()

        info = self._get_info(action=np.array([0.0, 0.0, 0.0]), 
                              terminated=False, 
                              truncated=False)

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
        for _ in range (int(self._policy_timestep / self._control_timestep)):
            for _ in range(int(self._control_timestep / self._model_timestep)):
                self._sim.forward()
                self._pre_action(action, policy_step)
                self._sim.step()
                policy_step = False
        
        # print("DMP idx:", self.scooping_planner._step_idx, "Timestep:", self.timestep, "mujoco step:", self._sim._data.time)

        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self._control_timestep

        if self.viewer is not None:
            self.viewer.sync()
                
        reward, terminated, truncated, info = self._post_action(action)
        
        observations = self._get_obs()
        
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `gym.make`
        return observations, reward, terminated, truncated, info
    
    def _get_obs(self):
        """
        Returns the observations from the environment.
        """
        
        observation = {}
        for key in self._obs_keys:

            if key == "eef_pose":
                observation[key] = self._robot.get_eef_pose()
            elif key == "target_amount":
                observation[key] = np.array([self._target_amount_state], dtype=np.int64)
            elif key == "image":
                image_array = self.render_observation()
                observation[key] = image_array
            elif key == "ft_sensor":
                observation[key] = self._robot._ft_sensor_value.copy()
            elif key == "actual_amount":
                if self._check_done():
                    observation[key] = np.array([self.number_of_particles_scooped()], dtype=np.int64)
                else:
                    observation[key] = np.array([0], dtype=np.int64)
                # observation[key] = int(self.number_of_particles_scooped())
            elif key == "pred_food_pose":
                observation[key] = np.array(self._food_pose, dtype=np.float64)
            elif key == "dmp_idx":
                observation[key] = np.array([self.scooping_planner._step_idx], dtype=np.float64)
            elif key == "particle_amount":
                observation[key] = np.array([self._randomization_dict["amount_ratio"]], dtype=np.float64)
            elif key == "particle_damping":
                observation[key] = np.array([self._randomization_dict["damping"]], dtype=np.float64)
            elif key == "particle_size":
                observation[key] = np.array([self._randomization_dict["size"]], dtype=np.float64)
            else:
                raise ValueError(f"Invalid observation key: {key}")
        
        return observation
    
    def _get_info(self, action, terminated, truncated):
        """
        Returns auxiliary information returned by step and reset.
        """
        if self._check_done():
            # Success if number of particles scooped is within 1 of target amount
            num_particles_scooped = self.number_of_particles_scooped()
            particle_range = self.desired_amount_to_scoop(self._target_amount_state)
            
            if num_particles_scooped in range(particle_range[0], particle_range[1] + 1):
                success = True
            else:
                success = False

            print("Num particles scooped:", num_particles_scooped, 
                  "| Target:", self.desired_amount_to_scoop(self._target_amount_state),
                  "| Success:", success, 
            )

        else:
            num_particles_scooped = -1
            success = False

        return {
            "is_success": success,
            "truncated": truncated,
            "particle_size": self._randomization_dict["size"],
            "particle_ratio": self._randomization_dict["amount_ratio"],
            "particle_damping": self._randomization_dict["damping"],
            "offset_pose": self._offset_pose,
            "num_particles_scooped": num_particles_scooped,
            "target_amount": self.desired_amount_to_scoop(self._target_amount_state),
        }
    
    def _pre_action(self, action, policy_step=False):
        """
        Do any preprocessing before taking an action.
        Args:
            action (ndarray): Action to execute within the environment
            policy_step (bool): Whether this current loop is an actual policy step or internal sim update step
        """
        if self._robot._controller.name == "IK_POSE": 
            # Update the dmp goal
            if policy_step:
                self.scooping_planner.update_dmp_params({self._action_mode: action})

                # Step the dmp to get next eef pose - note orientation is in quat (w,x,y,z)
                y, dy, ddy = self.scooping_planner.step()
                # print("ACTUAL Y DES:")
                # print(">> pos:", y[:3])
                # print(">> quat:", y[3:])

                # print("-----------------------")
            else:
                y = self.scooping_planner.current_y
            
            # NOTE: Control delta should be set to False
            assert self._robot._controller._use_delta == False, "Control delta should be set to False"
            control_action_pos = y[:3]
            control_action_orn = quat2axisangle(y[3:])

            # # NOTE: Control delta should be set to True -> THIS METHOD DOESN"T GIVE CORRECT GOAL ORI
            # Convert action space of 7 (quat) to 6 (axisangle)
            # For IK_POSE controller, action should be the delta 
            # control_action_pos = y[:3] - self._robot.get_eef_pose()[:3]
            # quat_delta = quat_distance(self._robot.get_eef_pose()[3:], y[3:])
            # control_action_orn = quat2axisangle(quat_delta)
            
            control_action = np.concatenate([control_action_pos, control_action_orn])
        
        elif self._robot._controller.name == "JOINT_POSITION":
            # Action is the joint position command
            control_action = action

        # Verify that the action is the correct dimension - orientation should be axisangle (not euler or quat!)
        assert control_action.shape[0] == self._robot._control_dim, "Control action dimension must match robot's control dimension"
        self._robot.control(control_action, policy_step=policy_step)

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

        # Check if the episode should be truncated
        truncated = self._check_truncated()

        # Empty dict to be filled with info
        info = self._get_info(action, terminated, truncated)

        return reward, terminated, truncated, info

    def _check_done(self):
        """
        Check if the episode is done.
        """
        if self.timestep >= self.scooping_planner._orig_y.shape[0]:
            # print(f"Episode done! Current timestep: {self.timestep}, "
            #     f"DMP step: {self.scooping_planner._step_idx}, "
            #     f"DMP y0 shape: {self.scooping_planner._orig_y.shape[0]}")
            
            done = True
        else:
            done = False

        return done
    
    def _check_truncated(self):
        """
        Check if the episode should be truncated
        """
        truncate = False

        return truncate
    
    def _distance_to_ref_traj(self):
        """
        Check the distance between the current eef pose and the reference trajectory.
        """
        if self.scooping_planner._step_idx >= self.scooping_planner._orig_y.shape[0]:
            idx = self.scooping_planner._orig_y.shape[0] - 1
        else:
            idx = self.scooping_planner._step_idx

        # If current pose is too far from the original dmp traj
        ref_pose = self.scooping_planner.get_eef_pose_from_spoon_pose(self.scooping_planner._ref_traj[idx],
                                                                      self.scooping_planner._spoon_offset)
        distance_trans = np.linalg.norm(self._robot.get_eef_pose()[:3] - ref_pose[:3])

        distance_rot = np.linalg.norm(quat_error(self._robot.get_eef_pose()[3:], ref_pose[3:]))

        return distance_trans, distance_rot

    def _get_reward(self, action):
        """
        Returns the reward for the current state.
        """
        # Reward for each step - penalize every time a change in action is made     
        if self._action_mode == "fz_discrete":
            reward_step = -1 if action != 0 else 0
        else:
            reward_step = -np.linalg.norm(action) 

        # Reward for amount of food scooped
        if self._check_done():
            scooped_amount = self.number_of_particles_scooped()
            # abs_diff = np.abs(scooped_amount - self._target_amount)

            # if scooped_amount <= 0:
            #     reward_amount = -10.0
            # else:
            #     reward_amount = 10.0 / (1.0 + abs_diff)

            #     if abs_diff <= 2:
            #         reward_amount += 20.0   # Bonus for being close to target
            #     elif abs_diff <= 5:
            #         reward_amount += 5.0    # Smaller bonus

            # print("actual amount:", self.number_of_particles_scooped(), "target:", self._target_amount, "reward amount:", reward_amount)

            particle_range = self.desired_amount_to_scoop(self._target_amount_state)
            # if scooped_amount in range(particle_range[0], particle_range[1] + 1):
            #     reward_amount = 10.0
            # elif scooped_amount <= 0:
            #     reward_amount = -10.0
            # else:
            #     reward_amount = 2.0
            
            # Provide more informational rewards, eg difference in number of particles based on mean
            desired_amount = (particle_range[0] + particle_range[1]) / 2
            abs_diff = np.abs(scooped_amount - desired_amount)
            if scooped_amount <= 0:
                reward_amount = -10.0
            else: 
                reward_amount = 10.0 / (1.0 + abs_diff)
            

        else:
            reward_amount = 0 
        
        # Reward for colliding with table or container
        if self._robot._sim.check_body_contact(["spoon", "link6", "table1", "container"]):
            reward_collision = -1
        else:
            reward_collision = 0

        if self._reward_type == "sparse":
            reward = reward_amount
        else:
            # Dense reward
            reward = reward_amount + reward_step + reward_collision
        # print("total reward:", reward)

        return reward
      
    def number_of_particles_scooped(self):
        """
        Returns the number of particles scooped by the spoon.

        Returns:
            int: Number of particles scooped
        """
        # Get force exerted on spoon in z direction
        # Divide by particle mass (0.001 kg) and by gravity (9.81 m/s^2)
        try: 
            body_mass = self._sim.get_body_mass(self._particle_body_id)
            return round(self._robot.get_contact_force("spoon")[2]/(body_mass*self._sim._gravity[2]))

        except Exception as e:
            print("Error in number_of_particles_scooped:", e)
            return 0
    
    def desired_amount_to_scoop(self, target_amount_discretized):
        target_amount_dict = {
            0.005: {
                "0": [1, 9],
                "1": [10, 18],
                "2": [19, 26],
                "3": [27, 35],
            },
            0.006: {
                "0": [1, 5],
                "1": [6, 10],
                "2": [11, 15],
                "3": [16, 25],
            },
            0.007: {
                "0": [1, 2],
                "1": [3, 5],
                "2": [6, 8],
                "3": [9, 15],
            }
        }

        return target_amount_dict[self._randomization_dict["size"]][str(target_amount_discretized)]
    
    @property
    def reference_traj(self):
        return self.scooping_planner._ref_traj_eef
    
    @property
    def modified_traj(self):
        return self.scooping_planner.get_stepped_traj()
    
    @property
    def qacc_joints(self):
        return self._sim.get_joint_qacc(self._robot._arm_joint_ids)

