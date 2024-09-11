import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from utils.transform_utils import quat_multiply, quat2axisangle
    from environments.scooping_env_v2 import XArmScoopEnvV2
except:
    from feeding_mujoco.utils.transform_utils import quat_multiply, quat2axisangle
    from feeding_mujoco.environments.scooping_env_v2 import XArmScoopEnvV2

class ScoopingNoDmpEnv(XArmScoopEnvV2):
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
                 particle_mass="random",
                 target_amount="random",
                 randomize_poses = True,
                 ppo_obs_hist=5,
                 all_ft_axes=True,
                 seed=0,
                 ):
        
        super().__init__(model_path,
                         sim_timestep=sim_timestep,
                         controller_config=controller_config,
                         control_freq=control_freq,
                         policy_freq=policy_freq,
                         render_mode=render_mode,
                         camera=camera,
                         obs_mode=obs_mode,
                         action_mode=action_mode,
                         reward_type=reward_type,
                         particle_size=particle_size,
                         particle_amount_ratio=particle_amount_ratio,
                         particle_damping=particle_damping,
                         particle_mass=particle_mass,
                         target_amount=target_amount,
                         randomize_poses=randomize_poses,
                         ppo_obs_hist=ppo_obs_hist,
                         all_ft_axes=all_ft_axes,
                         seed=seed,)
        

        # Action space is delta of cartesian eef pose
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))
        
        self.observation_space = self._get_basic_observation_space()
        self.observation_space.spaces["ft_sensor"] = spaces.Box(low=-np.inf,
                                                                high=np.inf,
                                                                shape=(self._ppo_obs_hist, 6),
                                                                dtype=np.float64)
        self.observation_space.spaces["target_amount"] = spaces.Box(low=1,
                                                                    high=100,
                                                                    shape=(1,),
                                                                    dtype=np.int64)
        self.observation_space.spaces["actual_amount"] = spaces.Box(low=-1,
                                                                    high=100,
                                                                    shape=(1,),
                                                                    dtype=np.int64)
        self.observation_space.spaces["pred_food_pose"] = spaces.Box(low=-1.0,
                                                                     high=1.0,
                                                                     shape=(3,),
                                                                     dtype=np.float64)
        self.observation_space.spaces["eef_pose"] = spaces.Box(low=-1.0,
                                                               high=1.0,
                                                               shape=(self._ppo_obs_hist, 7),
                                                               dtype=np.float64)
        self.observation_space.spaces["dmp_idx"] = spaces.Box(low=0,
                                                               high=50,
                                                               shape=(1,),
                                                               dtype=np.int64)
        
        # self.observation_space.spaces["particle_amount"] = spaces.Box(low=-np.inf,
        #                                                               high=np.inf,
        #                                                               shape=(1,),
        #                                                               dtype=np.float64)
        
        # self.observation_space.spaces["particle_damping"] = spaces.Box(low=-np.inf,
        #                                                                high=np.inf,
        #                                                                shape=(1,),
        #                                                                dtype=np.float64)
        
        # self.observation_space.spaces["particle_size"] = spaces.Box(low=-np.inf,
        #                                                             high=np.inf,
        #                                                             shape=(1,),
        #                                                             dtype=np.float64)
        
        # Get the keys in the observation space (must override the original observation space)
        self._obs_keys = list(self.observation_space.spaces.keys())

    def convert_action_to_eef_pose(self, action):
        # Transform the delta action to the cartesian eef pose before passing to the original environment
        curr_eef_pose = self._robot.get_eef_pose() 
        delta_eef_pose = action

        # Compute the new eef pose
        new_position = curr_eef_pose[:3] + delta_eef_pose[:3]
        new_orientation = quat_multiply(curr_eef_pose[3:], delta_eef_pose[3:])

        # normalize the quaternion
        new_orientation /= np.linalg.norm(new_orientation)
        
        # Check if quat is normalized
        assert np.isclose(np.linalg.norm(new_orientation), 1.0), "Quaternion should be normalized"
        
        # print("recalculated pose:", np.concatenate([new_position, new_orientation]))

        return np.concatenate([new_position, new_orientation])
    
    def _pre_action(self, action, policy_step=False, dmp_step=False):
        # Check that IK_POSE controller is used
        assert self._robot._controller.name == 'IK_POSE', 'Controller should be IK_POSE'

        # Only update eef pose every policy step
        if policy_step:
            target_eef_pose = self.convert_action_to_eef_pose(action)
            y = target_eef_pose
            self._target_eef_pose = target_eef_pose
            # print("Policy step:", self.timestep, "Target eef pose:", y)
        else:
            y = self._target_eef_pose
        
        assert self._robot._controller._use_delta == False, "Control delta should be set to False"
        control_action_pos = y[:3]
        control_action_orn = quat2axisangle(y[3:])

        control_action = np.concatenate([control_action_pos, control_action_orn])

        # Verify that the action is the correct dimension - orientation should be axisangle (not euler or quat!)
        assert control_action.shape[0] == self._robot._control_dim, \
            f"Control action dimension {control_action.shape[0]} must match robot's control dimension {self._robot._control_dim}"
        self._robot.control(control_action, policy_step=policy_step)

    def _check_truncated(self):
        """
        Check if the episode should be truncated
        """
        min_bounds = [-0.5, -0.5, 0.0]
        max_bounds = [0.5, 0.5, 0.5]
        # Truncate if out of bounds
        eef_pose = self._robot.get_eef_pose()
        eef_pos = eef_pose[:3]
        if eef_pos[0] < min_bounds[0] or eef_pos[0] > max_bounds[0] or \
            eef_pos[1] < min_bounds[1] or eef_pos[1] > max_bounds[1] or \
            eef_pos[2] < min_bounds[2] or eef_pos[2] > max_bounds[2]:
            truncate = True
            # print("Exceeded bounds. Truncating at step:", self.timestep)
        else:
            truncate = False

        return truncate
    
    def _check_done(self):
        """
        Check if the episode is done.
        Assume done if hit 20 RL steps
        """
        if self.timestep >= 20:
            done = True
        else:
            done = False
        
        return done
