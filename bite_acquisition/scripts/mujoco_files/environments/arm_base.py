import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from simulator.mujoco_simulator import MujocoSimulator
    from simulator.mujoco_modder import TextureModder
    from robots.xarm6 import XArm6Robot
except:
    from feeding_mujoco.simulator.mujoco_simulator import MujocoSimulator
    from feeding_mujoco.simulator.mujoco_modder import TextureModder
    from feeding_mujoco.robots.xarm6 import XArm6Robot

class ArmBaseEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"], 
        "render_fps": 5,
        "obs_modes": ["rgb_array", "depth_array", "rgbd_array", 
                      "segmented_rgb_array", "segmented_rgbd_array", 
                      "segmented_spoon_and_food",
                      "None", None]
    }   #TODO: add functionality to render fps 

    def __init__(self, 
                 model_path,
                 sim_timestep=0.01,
                 controller_config=None,
                 control_freq=50,
                 policy_freq=25,
                 render_mode=None,
                 camera=-1,
                 obs_mode=None
                ):

        self._model_path = model_path
        self._sim_timestep = sim_timestep
        self._control_freq = control_freq
        self._policy_freq = policy_freq
        self._controller_config = controller_config
        self._camera = camera

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        assert obs_mode is None or obs_mode in self.metadata["obs_modes"]
        if obs_mode == "None":
            obs_mode = None
        self.obs_mode = obs_mode

        self._initialize_time()

        self._load_simulator()

        self._load_robot()

        self._reset_internal(options={"wait": False})

        """
        If human-rendering is used, `self._window` will be a reference
        to the window that we draw to. `self._clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        Note: not implemented yet
        """
        self._window = None
        self._clock = None

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): Seed to initialise environment's PRNG (np_random). Defaults to None.
        """
        # Implements the seeding in gym correctly
        super().reset(seed=seed)

        # Reset the environment to its initial state
        self._sim.reset()

        # Reset robot position
        # print("initial pose:", self._initial_pose)
        # print("joint ids:", self._robot._arm_joint_ids)
        self._robot.hard_set_joint_positions(self._initial_pose, self._robot._arm_joint_ids)

        # Close gripper
        self._robot.close_gripper(hard=False)

        # Reset any internal variables
        self._reset_internal(options=options)
        self._sim.forward()

        # Other env-specific reset operations to be implemented in the child class
        return 
    
    def forward(self):
        """
        Forwards the simulation by one step.
        """
        self._sim.forward()
    
    def render(self):
        """
        Compute the render frames and return them.
        """
        if self.render_mode == "human":
            return None
        
        elif self.render_mode in ["rgb_array", "depth_array"]:
            return self._sim.render(self.render_mode)
        
        elif self.render_mode is None:
            return None
        
        else:
            raise ValueError("Invalid render mode!")
    
    def render_observation(self):
        """
        Render the image arrays for the observation space.
        """        
        if "segmented" in self.obs_mode:
            if self._sim._blacklisted_geom_ids is None:
                self._sim._blacklisted_geom_ids = [-1]
                blacklisted_body_names = ["table1", "floor"]
                for name in blacklisted_body_names:
                    body_id = self._sim.get_body_id_from_name(name)
                
                    geom_start = self._sim._model.body_geomadr[body_id]
                    geom_end = geom_start + self._sim._model.body_geomnum[body_id]
                
                    geoms = list(range(geom_start, geom_end))
                    self._sim._blacklisted_geom_ids.extend(geoms)

        if self.obs_mode == "rgb_array":
            image_arr = self._sim.render("rgb_array")
        
        elif self.obs_mode == "depth_array":
            image_arr = self._sim.render("depth_array")

        elif self.obs_mode == "rgbd_array":
            image_arr = self._sim.render("rgbd_array")
        
        elif self.obs_mode == "segmented_rgb_array":
            image_arr = self._sim.render("segmented_rgb_array")
         
        elif self.obs_mode == "segmented_rgbd_array":
            image_arr = self._sim.render("segmented_rgbd_array")

        elif self.obs_mode == "segmented_spoon_and_food":
            image_arr = self._sim.render("segmented_spoon_and_food")
        
        else:
            raise ValueError("Invalid obs_mode!")
        
        # # Normalize the depth array to [0,1]
        # # NOTE: should NOT normalize if using SB3 - it will be done automatically
        # image_arr = image_arr / 255.0

        # Check that image is in np.uint8 format
        if image_arr.dtype != np.uint8:
            raise ValueError("Image array is not in np.uint8 format!")

        return image_arr
    
    def close(self):
        """
        Perform any necessary cleanup operations.
        """
        self._sim.close()
    
    def _initialize_time(self):
        """
        Initialize time-related variables.
        """
        self.timestep = 0
        self.cur_time = 0

        self._model_timestep = self._sim_timestep
        if self._model_timestep <=0:
            raise ValueError("Invalid simulation timestep defined!")
        
        self._control_timestep = 1. / self._control_freq
        if self._control_timestep <=0:
            raise ValueError("Invalid control timestep defined!")
        
        self._policy_timestep = 1. / self._policy_freq
        if self._policy_timestep <=0:
            raise ValueError("Invalid policy timestep defined!")
    
    def _load_simulator(self):
        """
        Load the simulator
        """

        if self.render_mode == "human":
            _use_viewer = True
            _use_renderer = False
        elif self.render_mode is None:
            _use_viewer = False
            _use_renderer = False
        else:
            _use_viewer = False
            _use_renderer = True
        
        if self.obs_mode is not None:
            _use_renderer = True

        self._sim = MujocoSimulator(model_path=self._model_path,
                                    sim_timestep=self._sim_timestep,
                                    render_mode=self.render_mode,
                                    use_viewer=_use_viewer,
                                    use_renderer=_use_renderer,
                                    camera=self._camera
                                    )
        
        self._modder = TextureModder(self._sim,
                                    geom_names=["spoon", "table1_geom", "+x", "-x", "+y", "-y", "ground"],
                                    )

        # Run a single step to make sure changes have propoagated through the simulator state
        self._sim.forward()
    
    def _load_robot(self):
        """
        Initializes the robot and sets it to its neutral pose.
        """
        # Load robot - Assumes XArm6 with force torque sensor and spoon attached
        self._robot = XArm6Robot(self._sim, 
                                 controller_config=self._controller_config)
        
        self._initial_pose = self._robot._arm_neutral_pose

        # Set arm to neutral pose
        self._robot.hard_set_joint_positions(self._initial_pose, self._robot._arm_joint_ids)

        # Close robot gripper - assumes spoon is always attached
        self._robot.close_gripper(hard=False)
        
    def _reset_internal(self, options=None):
        """
        Resets any internal variables.
        """
        # Stabilise the simulation
        if options is None or ("wait" in options and options["wait"]==True):
            while self._sim.time < 2.0:
                self._sim.step()

        self.cur_time = 0
        self.timestep = 0

    def _get_basic_observation_space(self):
        """
        Returns a basic observation space based on obs_mode.

        If obs_mode is None, observation space will only include eef_pose.
        If obs_mode is an image type, observation space will include the eef_pose and image.

        Returns:
            spaces.Dict: Observation space
        """

        if self.obs_mode is None:
            observation_space = spaces.Dict(
                {
                    "eef_pose": spaces.Box(low=-1.0, 
                                            high=1.0, 
                                            shape=(3,), 
                                            dtype=np.float64)
                }
            )
        
        else:
            h = self._sim._renderer.height
            w = self._sim._renderer.width

            if self.obs_mode == "rgb_array" or self.obs_mode == "segmented_rgb_array":
                shape = (h, w, 3)
            elif self.obs_mode == "depth_array":
                shape = (h, w, 1)
            elif self.obs_mode == "rgbd_array" or self.obs_mode == "segmented_rgbd_array":
                shape = (h, w, 4)
            elif self.obs_mode == "segmented_spoon_and_food":
                shape = (100, 200, 3)

            observation_space = spaces.Dict(
                {
                    "eef_pose": spaces.Box(low=-1.0, 
                                            high=1.0, 
                                            shape=(3,), 
                                            dtype=np.float64),

                    "image": spaces.Box(low=0,
                                        high=255,
                                        shape=shape,
                                        dtype=np.uint8),
                }
            )
        
        return observation_space
    
    def _set_arm_initial_pose(self, initial_pose):
        """
        Set the initial pose of the arm.

        Args:
            initial_pose (ndarray): Initial pose of the arm
        """
        self._initial_pose = initial_pose

    @property
    def viewer(self):
        return self._sim._viewer
    
    @property
    def sim(self):
        return self._sim
    
    @property
    def robot(self):
        return self._robot
    
    @property
    def get_sim_freq(self):
        return 1.0/self._sim_timestep
    
    @property
    def get_policy_freq(self):
        return self._policy_freq
    
    @property
    def get_control_freq(self):
        return self._control_freq
