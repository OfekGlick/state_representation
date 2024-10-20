import hydra
import omegaconf
import simple_env
from stable_baselines3.common.env_util import DummyVecEnv
from pettingzoo.mpe import simple_v3
from constants import *
import state_representation
from stable_baselines3 import PPO


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
        env_description = SimpleConstants.ENV_DESCRIPTION
        env_task_distribution = SimpleConstants.TASK_DESCRIPTION
    else:
        raise ValueError(f"Please specify a valid environment in the config file.")


    vec_env = DummyVecEnv([lambda: gym_env])
    obs = vec_env.reset()
    state_representor = state_representation.State_Representor(
        textual_representation_model=Models.LLaVa,
        code_representation_model=Models.DeepSeekCoder,
        env_description=env_description,
        task_distribution=env_task_distribution,
    )

    model = PPO('MlpPolicy', vec_env, verbose=1)

    # Train the model
    model.learn(total_timesteps=10)

    # Save the model
    model.save("ppo_pettingzoo")

    # Load the model
    model = PPO.load("ppo_pettingzoo")
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        image = vec_env.render(mode='rgb_array')
        gym_env.display_image(image)
        text_repr = state_representor.generate_textual_state_representation(image)
        print(text_repr)
        code_repr = state_representor.generate_code_state_representation(text_repr)
        print(code_repr)
        vec_env.render()
        break


if __name__ == '__main__':
    main()