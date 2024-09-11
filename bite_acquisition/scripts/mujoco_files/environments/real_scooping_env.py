import rospy, rospkg
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import time

import gymnasium as gym
from gymnasium import spaces

from xarm.wrapper import XArmAPI

from geometry_msgs.msg import PointStamped
from feeding_msgs.srv import GetScoopingPoint, GetScoopingPointRequest

try:
    from planners.scooping_dmp import ScoopingDMPPlanner 
    from utils.transform_utils import quat2euler, quat_distance, quat_error, quat2axisangle, quat2mat, mat2quat, euler2mat
except:

    from feeding_mujoco.planners.scooping_dmp import ScoopingDMPPlanner 
    from feeding_mujoco.utils.transform_utils import quat2euler, quat_distance, quat_error, quat2axisangle, quat2mat, mat2quat, euler2mat

class RealScoopingEnv(gym.Env):

    metadata = {
        "render_modes": ["rgb_array", None],
        "obs_modes": ["rgb_array", "depth_array", "rgbd_array", 
                      "None", None]
    }

    def __init__(self,
                 dmp_freq = 100,
                 obs_mode=None,
                 action_mode = "fz",
                 reward_type="sparse",
                 target_amount=1,
                 robot_ip = "192.168.1.201",
                 food_detection_pose = None,
                 ppo_obs_hist = 5,
                 all_ft_axes=True,
                 get_scooping_point=False,
                 ):
        
        self.observation_space = spaces.Dict({})
        self.observation_space["eef_pose"] = spaces.Box(low=-1.0,
                                                        high=1.0,
                                                        shape=(ppo_obs_hist, 7),
                                                        dtype=np.float64)
        self.observation_space["ft_sensor"] = spaces.Box(low=-np.inf,
                                                        high=np.inf,
                                                        shape=(ppo_obs_hist, 6),
                                                        dtype=np.float64)
        self.observation_space["target_amount"] = spaces.Box(low=1,
                                                            high=100,
                                                            shape=(1,),
                                                            dtype=np.int64)
        self.observation_space["actual_amount"] = spaces.Box(low=-1,
                                                            high=100,
                                                            shape=(1,),
                                                            dtype=np.int64)
        self.observation_space["pred_food_pose"] = spaces.Box(low=-1.0,
                                                            high=1.0,
                                                            shape=(3,),
                                                            dtype=np.float64)
        self.observation_space["dmp_idx"] = spaces.Box(low=0,
                                                        high=50,
                                                        shape=(1,),
                                                        dtype=np.int64)
        

        # If obs_mode is not None, then we need to add the corresponding observation space
        # TODO: set up if image is in obs space
        if obs_mode is not None:
            h = 280
            w = 320

            if obs_mode == "rgb_array":
                self.observation_space.spaces["image"] = spaces.Box(low=0,
                                                                    high=255,
                                                                    shape=(h, w, 3),
                                                                    dtype=np.uint8)
        
        
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
        elif self._action_mode == "fxyz":
            self.action_space = spaces.Box(low=-1.0,
                                        high=1.0,
                                        shape=(3,),
                                        dtype=np.float64)
        
        self._robot_speed = 70

        # Initialize variables
        self.timestep=0
        self.cur_time=0
        self._ppo_obs_hist = ppo_obs_hist
        self._reward_type = reward_type
        self._target_amount_state = target_amount
        self._all_ft_axes = all_ft_axes
        self._get_scooping_point = get_scooping_point

        """
        For reference: Original bowl position: [0.5094, -0.2887, 0.2000]
        """
        self._sim_default_pose = np.array([0.5, 0.1, 0.14])
        self._original_dmp_food_pos = np.array([0.5, 0.1, 0.12])
        self._offset_pose = self._sim_default_pose - self._original_dmp_food_pos

        # DMP Planner
        self.scooping_planner = ScoopingDMPPlanner(dt=1/dmp_freq)
        self.scooping_planner.update_food_pose(self._original_dmp_food_pos)

        # Set up arm
        self.setup_arm(robot_ip, reset=False)

        # Set up impedance control
        self.setup_arm_impedance_control(enable=True)

        # ROS init
        rospy.wait_for_service('food_perception/get_scooping_point', timeout=5.0)
    
    def setup_arm(self, ip, reset=False):
        """
        Set up the xArm robot.
        """
        self._robot = XArmAPI(ip)
        self._robot.motion_enable(enable=True)
        self._robot.set_mode(0)                   # Set to position control mode  
        self._robot.set_state(state=0)            # Set the robot to ready
        time.sleep(0.1)

        self._ft_sensor_enabled = False
        self._impedance_control_enabled = False

        if reset:
            self._robot.reset(wait=True)
    
    def setup_arm_impedance_control(self, enable=True):
        """
        Set up the xArm robot in impedance control mode.
        """
        # Set tool impedance parameters:
        K_POS = 300         #  x/y/z linear stiffness coefficient, range: 0 ~ 2000 (N/m)
        K_ORI = 4           #  Rx/Ry/Rz rotational stiffness coefficient, range: 0 ~ 20 (Nm/rad)

        # Attention: for M and J, smaller value means less effort to drive the arm, but may also be less stable, please be careful. 
        M = float(0.06)  #  x/y/z equivalent mass; range: 0.02 ~ 1 kg
        J = M * 0.01     #  Rx/Ry/Rz equivalent moment of inertia, range: 1e-4 ~ 0.01 (Kg*m^2)

        C_AXIS = [0,0,1,0,0,0]  # set z axis as compliant axis
        REF_FRAME = 0           # 0 : base , 1 : tool

        self._robot.set_impedance_mbk([M, M, M, J, J, J], [K_POS, K_POS, K_POS, K_ORI, K_ORI, K_ORI], [0]*6) # B(damping) is reserved, give zeros
        self._robot.set_impedance_config(REF_FRAME, C_AXIS)
        
        # Enable ft sensor communication
        self._robot.ft_sensor_enable(1)
        self._ft_sensor_enabled = True

        if enable:
            self.enable_impedance_control()
            print("Impedance control setup and enabled.")
        time.sleep(0.1)

    def enable_impedance_control(self):
        self._robot.ft_sensor_app_set(1)
        self._robot.set_state(0)        # Will start impedance mode after set_state(0)
        self._impedance_control_enabled = True

    def disable_impedance_control(self):
        self._robot.ft_sensor_app_set(0)
        self._robot.set_state(0)       
        self._impedance_control_enabled = False
        print("Disabled impedance control")

    def disconnect_arm(self, reset=False):
        # Reset mode back to position mode
        self._robot.set_mode(0)
        self._robot.set_state(0)

        # Disable impednace control
        if self._impedance_control_enabled:
            self.disable_impedance_control()

        # Disable ft sensor communication
        if self._ft_sensor_enabled:
            self._robot.ft_sensor_enable(0)
        
        if reset:
            self._robot.reset(wait=True)
        
        self._robot.disconnect()

    def get_eef_pose_to_set(self, pose):
        """
        Converts the end-effector pose in meters and in quaternion (wxyz) format to 
        a format required by xArm Python SDK (in mm and euler angles in radians)
        """
        # Convert the orientation component from quat to euler
        orn_rpy = quat2euler(pose[3:])
        
        for i in range(3):
            pose[i] = pose[i] * 1000
        pose_rpy_mm = np.concatenate([pose[:3], orn_rpy])

        assert len(pose_rpy_mm) == 6, "End effector pose must be 6 elements"

        return pose_rpy_mm
    
    def get_eef_pose(self):
        """
        Get the current end effector pose of the robot and return it in meters and in quaternion (wxyz) format.
        """
        pose_rpy_mm = self._robot.get_position(is_radian=True)[1]
        orn_quat = mat2quat(euler2mat(pose_rpy_mm[3:]))

        for i in range(3):
            pose_rpy_mm[i] = pose_rpy_mm[i] / 1000
        pose_quat_m = np.concatenate([pose_rpy_mm[:3], orn_quat])

        assert len(pose_quat_m) == 7, "End effector pose must be 7 elements"

        return np.array(pose_quat_m)
    
    def get_ft_reading(self, all_axes=True):
        """
        Returns the current force torque reading and transforms based on readings when scooping without food
        """
        ft_ext = self._robot.ft_ext_force
        air_mins = np.array([-0.7028087973594666, -1.0518187284469604, 1.5879607200622559, 0.05467132106423378, -0.07648644596338272, -0.04186679422855377])
        air_maxs = np.array([1.6891001462936401, 0.5354885458946228, 2.5736215114593506, 0.19799873232841492, 0.05250909924507141, 0.04186679422855377])

        transformed_ft_values = np.zeros_like(ft_ext)
        for i in range(6):
            if all_axes:
                transformed_ft_values[i] = (ft_ext[i] - air_mins[i])/(air_maxs[i] - air_mins[i])
            else:
                # only include fx, fz, ty
                if i in [0,2,4]:
                    transformed_ft_values[i] = (ft_ext[i] - air_mins[i])/(air_maxs[i] - air_mins[i])
                else:
                    transformed_ft_values[i] = 0

        return transformed_ft_values
    
    def move_to_scooping_wp0(self):
        start_pose = self.scooping_planner.get_eef_pose_from_spoon_pose(
                        self.scooping_planner._dmp.y0,
                        self.scooping_planner._spoon_offset
        )
        start_pose_with_z_offset = start_pose
        offset = 0.05
        start_pose_with_z_offset[2] = start_pose_with_z_offset[2] + offset
        pose_to_set_with_z_offset = self.get_eef_pose_to_set(start_pose_with_z_offset)
        ret = self._robot.set_position(*pose_to_set_with_z_offset,
                                       speed=self._robot_speed,
                                       is_radian=True,
                                       motion_type=2,
                                       wait=True)
        if ret == 0:
            print("Moved to wp0 with z offset")
        
        # Move relative to current position
        ret = self._robot.set_position(z=-offset*1000,
                                       relative=True,
                                       wait=True)
        
        if ret == 0:
            print("Moved to wp0")
        #TODO: how to handle error?

    def move_to_pose(self, pose_m_quat, speed=None):
        print("Moving arm to:", pose_m_quat)
        pose_to_set = self.get_eef_pose_to_set(pose_m_quat)
        ret = self._robot.set_position(*pose_to_set,
                                        speed=speed,
                                        is_radian=True,
                                        motion_type=2,
                                        wait=True)
        if ret == 0:
            print(f"Moved to pose {pose_m_quat}.")

    def get_scooping_point(self):
        """
        Get scooping point from food perception

        Returns:
            geometry_msgs/PointStamped: Point to be scooped
        """
        rospy.loginfo("Getting scooping point from food perception...")
        try:
            srv_get_scooping_point = rospy.ServiceProxy("food_perception/get_scooping_point", GetScoopingPoint)

            start_time = time.time()
            req_food_detection = GetScoopingPointRequest()
            resp_food_detection = srv_get_scooping_point(req_food_detection)

            if resp_food_detection.success:
                scooping_point = resp_food_detection.scooping_point
                rospy.loginfo(f"Obtained scooping point in {(time.time() - start_time):.3f}(s)")
            else:
                rospy.logerr("Failed to get scooping point")
                scooping_point = None
        
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s"%e)
            scooping_point = None
        
        return scooping_point

    def _reset_internal(self):
        """
        Resets any internal variables.
        """
        self.cur_time = 0
        self.timestep = 0

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
        # Implementes the seeding in gym correctly
        super().reset(seed=seed)
        self._reset_internal()

        print("Target amount:", self._target_amount_state)

        # (1) Move to food detection pose
        start_pose = self.scooping_planner.get_eef_pose_from_spoon_pose(
                        self.scooping_planner._dmp.y0,
                        self.scooping_planner._spoon_offset
        )
        food_detection_pose = start_pose
        food_detection_pose[2] = food_detection_pose[2] + 0.05
        self.move_to_pose(food_detection_pose)

        # (2) Food perception
        if self._get_scooping_point:
            food_pose_msg = None
            retries = 0
            while food_pose_msg is None:
                food_pose_msg = self.get_scooping_point()
                food_pose_msg.point.y += 0.02
                food_pose_msg.point.x += 0.02
                food_pose_msg.point.z += 0.02

                retries += 1
                if retries > 5:
                    print("Failed to get food pose. Exiting...")
                    return None, None

            # convert food pose into numpy array
            self._dmp_food_pose = np.array([food_pose_msg.point.x, 
                                            food_pose_msg.point.y, 
                                            food_pose_msg.point.z,
                                            ])
            self._offset_pose = self._sim_default_pose - self._dmp_food_pose
            print("Offset pose:", self._offset_pose)
            print("Food pose:", self._dmp_food_pose)
        else:
            self._dmp_food_pose = self._original_dmp_food_pos.copy()
            
        # (3) Update food pose
        # Reset DMP
        self.scooping_planner.reset()
        self.scooping_planner.update_food_pose(self._dmp_food_pose)
        
        # Reset robot to start of trajectory
        self.move_to_scooping_wp0()

        observation = self._get_obs()
        info = self._get_info(action=np.array([0.0, 0.0, 0.0]), 
                              terminated=False, 
                              truncated=False)
        

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
        # Increment the timestep
        self.timestep += 1

        eef_pose_hist = []
        ft_sensor_hist = []

        for j in range(self._ppo_obs_hist):

            self.scooping_planner.update_dmp_params({self._action_mode: action})

            wp, dwp, ddwp = self.scooping_planner.step()
            
            pose_rpy_mm = self.get_eef_pose_to_set(wp)

            ret = self._robot.set_position(*pose_rpy_mm,
                                            speed=self._robot_speed,
                                            is_radian=True,
                                            wait=True)
            
            # TODO: handle errors
            if ret != 0:
                print("!!! ERROR !!!:", ret)
            
            # Save eef_pose and ft_sensor values
            eef_pose_hist.append(self.get_eef_pose())
            ft_sensor_hist.append(self.get_ft_reading(all_axes=self._all_ft_axes))

        # print("DMP idx:", self.scooping_planner._step_idx, "Timestep:", self.timestep)

        # Post-actions
        reward = self._get_reward(action)
        terminated = False      # always set to false so arm does not reset itself
        truncated = self._check_truncated()
        info = self._get_info(action, terminated, truncated)
        
        observations = self._get_obs(eef_pose_hist, ft_sensor_hist)
        
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `gym.make`
        return observations, reward, terminated, truncated, info
    
    def _get_obs(self, eef_pose_hist=None, ft_sensor_hist=None):
        """
        Returns the observations from the environment.
        """
        
        observation = {}
        for key in self._obs_keys:

            if key == "eef_pose":
                if eef_pose_hist is None:
                    curr_eef_pose = np.array(self.get_eef_pose())
                    reshaped_pose = np.reshape(curr_eef_pose, (1, 7))
                    # pad past with 0s
                    padding = np.array([np.zeros_like(self.get_eef_pose()) for _ in range(self._ppo_obs_hist - 1)])
                    observation[key] = np.concatenate((padding, reshaped_pose), axis=0)
                else:
                    observation[key] = np.array(eef_pose_hist)
            elif key == "target_amount":
                observation[key] = np.array([self._target_amount_state], dtype=np.int64)
            # elif key == "image":
            #     image_array = self.render_observation()
            #     observation[key] = image_array
            elif key == "ft_sensor":
                if ft_sensor_hist is None:
                    ft_sensor_val = np.array(self.get_ft_reading(all_axes=self._all_ft_axes))
                    reshaped_ft_sensor = np.reshape(ft_sensor_val, (1, 6))
                    # pad past with 0s
                    padding = np.array([np.zeros_like(ft_sensor_val) for _ in range(self._ppo_obs_hist - 1)])
                    observation[key] = np.concatenate((padding, reshaped_ft_sensor), axis=0)
                else:
                    observation[key] = np.array(ft_sensor_hist)
            elif key == "actual_amount":
                # TODO: always return 0 for now
                observation[key] = np.array([0], dtype=np.int64)
            elif key == "pred_food_pose":
                observation[key] = np.array(self._dmp_food_pose, dtype=np.float64)
            elif key == "dmp_idx":
                observation[key] = np.array([self.scooping_planner._step_idx], dtype=np.int64)
            else:
                raise ValueError(f"Invalid observation key: {key}")
        
        return observation
    
    def _get_info(self, action, terminated, truncated):
        """
        Returns auxiliary information returned by step and reset.
        """
        if self._check_done():
            success = True
            num_particles_scooped = 1
            # TODO: Assume always successful for now
        else:
            num_particles_scooped = -1
            success = False

        return {
            "is_success": success,
            "truncated": truncated,
            "num_particles_scooped": num_particles_scooped,
            "target_amount": self._target_amount_state,
            "robot_error": self._robot.has_err_warn,
            "offset_pose": self._offset_pose    # For comparison with sim arm
        }
        
    def _check_done(self):
        """
        Check if the episode is done.
        """
        if self.scooping_planner._step_idx >= self.scooping_planner._orig_y.shape[0]:
            print(f"Episode done! Current timestep: {self.timestep}, "
                f"DMP step: {self.scooping_planner._step_idx}, "
                f"DMP y0 shape: {self.scooping_planner._orig_y.shape[0]}")
            
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
            scooped_amount = 1

            # TODO: we need some way to calc reward in real world (can be based on volume)
            if scooped_amount <= 0:
                reward_amount = -10.0
            else: 
                reward_amount = -100.0 # TODO: update this based on actual food amount
            
        else:
            reward_amount = 0 

        if self._reward_type == "sparse":
            reward = reward_amount
        else:
            reward = reward_amount + reward_step
        
        # print("total reward:", reward)
        return reward
    
    def close(self):
        if self._impedance_control_enabled:
            self.disable_impedance_control()
        self.disconnect_arm(reset=False)

    @property
    def reference_traj(self):
        return self.scooping_planner._ref_traj_eef
    
    @property
    def modified_traj(self):
        return self.scooping_planner.get_stepped_traj()

