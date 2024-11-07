import miniworld
import gymnasium as gym
from pprint import pprint

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
    ):
        super(CustomFourRoomsEnv, self).__init__()
        self.env = gym.make("MiniWorld-FourRooms-v0", domain_rand=True, render_mode=render_mode)
        self.relevant_params = relevant_params
        self.relevant_entity_attributes = relevant_entity_attributes
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Dict(self.get_bounds_of_features())

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

    def get_features(self):
        """
        Extracts the relevant features from the environment and organizes them in a dictionary
        """
        features = {}
        for feature in self.relevant_params:
            features[feature] = getattr(self.env, feature)
        features['box_visible'] = int(bool(len(self.env.get_visible_ents())))
        entities = self.env.entities
        for entity in entities:
            for attr in dir(entity):
                if not attr.startswith('_') and not callable(getattr(entity, attr)) and attr in ['size', 'pos']:
                    features[entity.name + "_" + attr] = getattr(entity, attr)
                elif not attr.startswith('_') and not callable(getattr(entity, attr)) and attr in ['color']:
                    features[entity.name + "_" + attr] = COLOR_TO_IDX[getattr(entity, attr)]
        return features

    def select_feature_subset(self, features):
        pass

    def reset(self, **kwargs):
        _, _ = self.env.reset(**kwargs)
        observation = self.get_features()
        # elements = [np.array([e]) if isinstance(e, int) else e for e in observation_dict.values()]
        # observation = np.concatenate(elements)
        return observation, {}

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        observation = self.get_features()
        # elements = [np.array([e]) if isinstance(e, int) else e for e in observation_dict.values()]
        # observation = np.concatenate(elements)
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        self.env.close()


if __name__ == '__main__':
    from stable_baselines3 import DQN
    print("Creating environment...")
    env = CustomFourRoomsEnv(render_mode='human')
    print("Done!")
    print("Creating model...")
    model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log="./dqn_fourrooms_tensorboard/")
    print("Done!")
    print("Training model...")
    model.learn(total_timesteps=100000, log_interval=4, tb_log_name="first_run")
    print("Done!")
    obs, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()

