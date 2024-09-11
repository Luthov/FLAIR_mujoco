try: 
    from environments.reaching_env import XArmReachEnv
    from environments.reaching_rgb_env import XArmReachRGBEnv
    from environments.scooping_env import XArmScoopEnv
    from environments.scooping_env_v1 import XArmScoopEnvV1
    from environments.scooping_env_v2 import XArmScoopEnvV2
    from environments.scooping_no_dmp_env import ScoopingNoDmpEnv
    from environments.scooping_rlfd_env import ScoopingRLFDEnv
    # from environments.real_scooping_env import RealScoopingEnv
except:
    from feeding_mujoco.environments.reaching_env import XArmReachEnv
    from feeding_mujoco.environments.reaching_rgb_env import XArmReachRGBEnv
    from feeding_mujoco.environments.scooping_env_v1 import XArmScoopEnvV1
    from feeding_mujoco.environments.scooping_no_dmp_env import ScoopingNoDmpEnv
    from feeding_mujoco.environments.scooping_rlfd_env import ScoopingRLFDEnv
    # from feeding_mujoco.environments.real_scooping_env import RealScoopingEnv
    
# Register env
from gymnasium.envs.registration import register

register(
    id="XArmReachEnv-v0",
    entry_point="environments:XArmReachEnv",
    max_episode_steps=50,
    reward_threshold=10.0
)

register(
    id="XArmReachRGBEnv-v0",
    entry_point="environments:XArmReachRGBEnv",
    max_episode_steps=50,
    reward_threshold=10.0
)

register(
    id="XArmScoopEnv-v0",
    entry_point="environments:XArmScoopEnv",
    max_episode_steps=100,
    reward_threshold=10.0
)

register(
    id="XArmScoopEnv-v1",
    entry_point="environments:XArmScoopEnvV1",
    max_episode_steps=15,
    reward_threshold=10.0
)

register(
    id="XArmScoopEnv-v2",
    entry_point="environments:XArmScoopEnvV2",
    max_episode_steps=15,
    reward_threshold=10.0
)

register(
    id="XArmScoopEnvNoDMP-v0",
    entry_point="environments:ScoopingNoDmpEnv",
    max_episode_steps=15,
    reward_threshold=10.0
)

register(
    id="XArmScoopRLFDEnv-v0",
    entry_point="environments:ScoopingRLFDEnv",
    max_episode_steps=15,
    reward_threshold=10.0
)


# Register this in the other python package
# register(
#     id="RealScoopingEnv-v0",
#     entry_point="environments:RealScoopingEnv",
#     max_episode_steps=15,
#     reward_threshold=10.0
# )