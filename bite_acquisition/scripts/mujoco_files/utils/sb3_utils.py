from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.utils import set_random_seed

from gymnasium.wrappers import RecordVideo

try:
    from wrappers import DomainRandomizationWrapper
    from wrappers import HERWrapper
except:
    from feeding_mujoco.wrappers import DomainRandomizationWrapper
    from feeding_mujoco.wrappers import HERWrapper

def make_env(env_id, 
             cfg, 
             controller_config, 
             rank=0, 
             seed=0, 
             record_dir=None, 
             ep_idx=500,
             wait=True,
             **env_kwargs):
    def _init():
        env = gym.make(env_id, 
                       model_path = cfg.env.model_path,
                       sim_timestep = cfg.env.sim_timestep,
                       control_freq = cfg.controller.control_freq,
                       policy_freq = cfg.controller.policy_freq,
                       controller_config = controller_config,
                       max_episode_steps=int(cfg.env.max_episode_steps),
                       render_mode=cfg.env.render_mode,
                       camera=cfg.env.camera,
                       obs_mode = cfg.env.obs_mode,
                       seed=seed,
                       **env_kwargs     # to pass in extra arguments
                       )
        if cfg.algo == "sac+her":
            env = HERWrapper(env, seed=seed)

        if cfg.env.randomize_textures:
            env = DomainRandomizationWrapper(env,
                                             seed = seed + rank,
                                             randomize_on_reset=True,
                                             randomize_every_n_steps=0,)

        if record_dir is not None:
            if ep_idx is None:
                episode_trigger = lambda x: True  # Record every episode
            else:
                episode_trigger = lambda x: x % ep_idx == 0  # Record every ep_idx episodes

            if cfg.env.render_mode in ["rgb_array"]:
                env = RecordVideo(env,
                                video_folder=record_dir + str(rank), 
                                episode_trigger=episode_trigger,  
                                name_prefix= "ppo_"  + str(cfg.env.camera),
                                disable_logger=True
                                )
        # No need to wait here, env is usually reset anothere time in the training loop/exec loop
        env.reset(seed = seed + rank, options={"wait": wait})
        
        return env
    set_random_seed(seed)
    return _init

# Add curriculum learning
def get_action_limits(curriculum_level):
    if curriculum_level == 1:
        return -0.0001, 0.0001
    elif curriculum_level == 2:
        return -0.001, 0.001
    elif curriculum_level == 3:
        return -0.01, 0.01

def update_action_limits(env, curriculum_level):
    low, high = get_action_limits(curriculum_level)
    for i in range(env.num_envs):
        env.envs[i].low = low
        env.envs[i].high = high

class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def __init__(self, algo):
        super(HParamCallback, self).__init__()
        self.algo = algo

    def _on_training_start(self) -> None:
        if self.algo == "ppo" or self.algo=="recurrent_ppo":
            hparam_dict = {
                "algorithm": self.model.__class__.__name__,
                "learning rate": self.model.learning_rate,
                "batch size": self.model.batch_size,
                "n_steps": self.model.n_steps,
                "n_epochs": self.model.n_epochs,
                "target_kl": self.model.target_kl,
                "gamma": self.model.gamma,
                "ent_coef": self.model.ent_coef,
            }
        elif self.algo == "sac" or self.algo =="sac+her":
            hparam_dict = {
                "algorithm": self.algo,
                "learning rate": self.model.learning_rate,
                "batch size": self.model.batch_size,
                "gamma": self.model.gamma,
                "buffer_size": self.model.buffer_size,
                "learning_starts": self.model.learning_starts,
            }
        
    
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0,
            "train/value_loss": 0.0,
        }

        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
    

class ActionClipWrapper(gym.ActionWrapper):
    def __init__(self, env, low, high):
        super().__init__(env)
        self.low = low
        self.high = high

    def action(self, action):
        # Clip action based on current bounds
        return np.clip(action, self.low, self.high)
