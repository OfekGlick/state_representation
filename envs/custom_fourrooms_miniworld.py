import miniworld
import gymnasium as gym
import numpy as np
from pprint import pprint
from tqdm import tqdm, trange
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO

COLOR_TO_IDX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'yellow': 3,
    'purple': 4,
    'grey': 5,
}


class CustomFourRoomsEnv(gym.Env):
    def __init__(
            self,
            relevant_params=('sky_color', 'light_pos', 'light_color', 'light_ambient'),
            relevant_entity_attributes=('color', 'size', 'position'),
            render_mode='rgb_array',
            filter_observation_space=False,
    ):
        super(CustomFourRoomsEnv, self).__init__()
        self.env = gym.make("MiniWorld-FourRooms-v0", domain_rand=True, render_mode=render_mode)
        self.render_mode = render_mode
        self.relevant_params = relevant_params
        self.relevant_entity_attributes = relevant_entity_attributes
        self.all_features = self.relevant_params + self.relevant_entity_attributes + ('box_visible',)
        self.action_space = self.env.action_space
        self.observation_space_dict = self.get_bounds_of_features()
        self.filter_observation_space = filter_observation_space
        if self.filter_observation_space:
            self.select_feature_subset()
        self.observation_space = gym.spaces.Dict(self.observation_space_dict)

    def get_bounds_of_features(self):
        param_bounds = {
            feature: gym.spaces.Box(low=-7, high=7, shape=(3,))
            for feature, bounds in self.env.params.params.items()
            if feature in self.relevant_params
        }
        param_bounds['box_visible'] = gym.spaces.Discrete(2)
        entities = self.env.entities
        for entity in entities:
            for attr in dir(entity):
                if not attr.startswith('_') and not callable(getattr(entity, attr)) and attr in ['size', 'pos']:
                    param_bounds["_".join([entity.name, attr])] = gym.spaces.Box(low=-7, high=7, shape=(3,))
                elif not attr.startswith('_') and not callable(getattr(entity, attr)) and attr in ['color']:
                    param_bounds["_".join([entity.name, attr])] = gym.spaces.Discrete(len(COLOR_TO_IDX))
        return param_bounds

    def _get_features(self):
        """
        Extracts the relevant features from the environment and organizes them in a dictionary
        """
        features = {}
        for feature in self.relevant_params:
            features[feature] = np.array(getattr(self.env, feature), dtype=np.float32)
        features['box_visible'] = np.array(int(bool(len(self.env.get_visible_ents()))), dtype=np.float32)
        entities = self.env.entities
        for entity in entities:
            for attr in dir(entity):
                if not attr.startswith('_') and not callable(getattr(entity, attr)) and attr in ['size', 'pos']:
                    features[entity.name + "_" + attr] = np.array(getattr(entity, attr), dtype=np.float32)
                elif not attr.startswith('_') and not callable(getattr(entity, attr)) and attr in ['color']:
                    features[entity.name + "_" + attr] = np.array(COLOR_TO_IDX[getattr(entity, attr)], dtype=np.float32)

        if self.filter_observation_space:
            features = {k: v for k, v in features.items() if k in self.observation_space_dict}
        return features

    def select_feature_subset(self, features=None):
        """
        Selects a subset of the features to be used in the observation space
        """
        # NOTE: Currently I am manually selecting a subset of features, in the future this will be done by an FM
        features = ['box_visible', 'Agent_pos', 'Box_pos']
        self.observation_space_dict = {k: v for k, v in self.observation_space_dict.items() if k in features}

    def reset(self, **kwargs):
        """
        Reset the environment and return the initial observation
        """
        _, _ = self.env.reset(**kwargs)
        observation = self._get_features()
        # elements = [np.array([e]) if isinstance(e, int) else e for e in observation_dict.values()]
        # observation = np.concatenate(elements)
        return observation, {}

    def step(self, action):
        """
        Take a step in the environment
        """
        _, reward, terminated, truncated, info = self.env.step(action)
        observation = self._get_features()
        # elements = [np.array([e]) if isinstance(e, int) else e for e in observation_dict.values()]
        # observation = np.concatenate(elements)
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment
        """
        return self.env.render()

    def close(self):
        """
        Close the environment
        """
        self.env.close()


if __name__ == '__main__':
    env = CustomFourRoomsEnv(
        render_mode='rgb_array',
        filter_observation_space=False,
    )
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./dqn_fourrooms_tensorboard/")
    print("Training model...")
    model.learn(total_timesteps=1000000, log_interval=4, tb_log_name="second_run")
    print("Done!")
    model.save("ppo_custom_fourrooms")
    model = PPO.load("ppo_custom_fourrooms")
    env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda episode_id: True)
    obs, info = env.reset()
    for _ in trange(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

    env_2 = CustomFourRoomsEnv(
        render_mode='rgb_array',
        filter_observation_space=True,
    )
    model_2 = PPO("MultiInputPolicy", env_2, verbose=1, tensorboard_log="./dqn_fourrooms_tensorboard/")
    print("Training model...")
    model_2.learn(total_timesteps=1000000, log_interval=4, tb_log_name="second_run")
    print("Done!")
    model_2.save("ppo_custom_fourrooms_custom")
    model_2 = PPO.load("ppo_custom_fourrooms_custom")
    env_2 = RecordVideo(env_2, video_folder='./videos_custom', episode_trigger=lambda episode_id: True)
    obs, info = env_2.reset()
    for _ in trange(1000):
        action, _states = model_2.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_2.step(action)
        env_2.render()
        if terminated or truncated:
            obs, info = env_2.reset()
    env_2.close()
