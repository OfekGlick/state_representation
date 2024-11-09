import hydra
import omegaconf
from envs import simple_env
from stable_baselines3.common.env_util import DummyVecEnv
from pettingzoo.mpe import simple_v3
from constants import *
from stable_baselines3 import PPO
from state_representors.textual_state_representation import Textual_State_Representor

"""
"""

@hydra.main(
    config_path="conf",
    config_name="config",
    version_base='1.3',
)
def main(dict_config: omegaconf.DictConfig):
    print(dict_config)
    if dict_config.env.simple:
        env = simple_v3.env(render_mode='rgb_array', continuous_actions=True)
        gym_env = simple_env.CustomSimpleEnv(env)
        gym_env.print_observation_structure()
        obs, _ = gym_env.reset()
        gym_env.print_observation_structure(obs)
        obs_docs = gym_env.get_observation_info()
        print(obs_docs)
        env_description = SimpleConstants.ENV_DESCRIPTION
        env_task_distribution = SimpleConstants.TASK_DESCRIPTION
    else:
        raise ValueError(f"Please specify a valid environment in the config file.")



if __name__ == '__main__':
    main()