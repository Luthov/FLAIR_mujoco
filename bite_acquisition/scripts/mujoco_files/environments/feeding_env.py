import math

try:
    from .base_environment import BaseEnvironment
    from simulator.mujoco_simulator import MujocoSimulator
    from robots.xarm6 import XArm6Robot
except:
    from feeding_mujoco.environments.base_environment import BaseEnvironment
    from feeding_mujoco.simulator.mujoco_simulator import MujocoSimulator
    from feeding_mujoco.robots.xarm6 import XArm6Robot

class FeedingXArm6Env(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        
    def init_env(self):
        # Load simulator
        self._load_simulator()

        self._load_robot()

        # Stabilise the simulation
        while self._sim.time < 1.0:
            self._sim.step()
        
        self._reset_internal()
        print("Feeding environment initialized.")

    def _load_simulator(self):
        """
        Load the simulator
        """
        self._sim = MujocoSimulator(model_path=self.config['model_path'],
                                    sim_timestep=self.config['sim_timestep'],
                                    use_viewer=True,
                                    use_renderer=False,
                                    )

        # Run a single step to make sure changes have propoagated through the simulator state
        self._sim.forward()
    
    def _load_robot(self):
        # Load robot
        controller_config = self.config


        self._robot = XArm6Robot(self._sim, 
                                 controller_config=controller_config)
        
        # Check that controller is valid
        valid_controllers = ["JOINT_POSITION"]
        assert self._robot._controller.name in valid_controllers, "Controller must be JOINT_POSITION for FeedingEnv"

        # Set arm to neutral pose
        self._robot.hard_set_joint_positions(self._robot._arm_neutral_pose, self._robot._arm_joint_ids)

        # Close gripper
        self._robot.close_gripper(hard=False)

    def reset(self):
        # Reset the environment to its initial state
        # Returns env observation space
        self._sim.reset()

        # Reset any internal variables
        self._reset_internal()
        self._sim.forward()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def _get_obs(self):
        """
        Returns the observations from the environment.
        """
        observation = {
            "joint_positions": self._robot._joint_positions,
            "ft_value": self._robot.get_contact_force("spoon")
        }

        return observation
    
    def _get_info(self):
        """
        Returns auxiliary information returned by step and reset.
        """
        return {
            "time": self.cur_time,
        }
    
    def _reset_internal(self):
        """
        Resets any internal variables.
        """
        super()._reset_internal()

        # Reset robot and controller - to implement if needed

    def _pre_action(self, action, policy_step=False):
        """
        Do any preprocessing before taking an action.
        Args:
            action (np.array): Action to execute within the environment
            policy_step (bool): Whether this current loop is an actual policy step or internal sim update step
        """
        # Verify that the action is the correct dimension
        assert len(action) == self._robot._control_dim, "Action dimension must match robot's joint dimension"

        self._robot.control(action, policy_step=policy_step)
    
    def _post_action(self, action):
        """
        Do any postprocessing after taking an action.
        Args:
            action (np.array): Action to execute within the environment
        """
        # Get the reward
        reward = self.reward()

        # Check if the episode is done
        done = self._check_done()

        # Empty dict to be filled with info
        info = self._get_info()

        return reward, done, info

    def _check_done(self):
        """
        Check if the episode is done.
        """
        # Check if the episode is done
        if self.timestep > 2*self._control_freq:
            done = True
        else:
            done = False

        return done

    def step(self, action):
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
        
        reward, done, info = self._post_action(action)
        
        observations = self._get_obs()

        return observations, reward, done, info
    
    def reward(self):
        """
        Returns the reward for the current state.
        """
        reward = -1
        return reward

  
    def forward(self):
        self._sim.forward()

    def render(self, mode='human'):
        # Implement the rendering logic
        pass

    def close(self):
        # Clean up resources
        self._sim.close()
    
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

    @property
    def viewer(self):
        return self._sim._viewer
    
    def initialize_time(self):
        """
        Initialize time-related variables.

        :param control_freq: Control frequency.
        """
        self.timestep = 0
        self.cur_time = 0
        
        self._model_timestep = self.config['sim_timestep']
        if self._model_timestep <=0:
            raise ValueError("Invalid simulation timestep defined!")
        
        self._control_freq = self.config['control_freq']
        self._control_timestep = 1. / self._control_freq
        if self._control_timestep <=0:
            raise ValueError("Invalid control timestep defined!")
        
        self._policy_freq = self.config['policy_freq']
        self._policy_timestep = 1. / self._policy_freq
        if self._policy_timestep <=0:
            raise ValueError("Invalid policy timestep defined!")
        
        print("Control timestep: ", self._control_timestep)
        print("Model timestep: ", self._model_timestep)
        print("Policy timestep:", self._policy_timestep)


