import miniworld
import gymnasium as gym
import numpy as np
from tqdm import trange
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


class CustomHallwayEnv(gym.Env):
    def __init__(
            self,
            render_mode='rgb_array',
    ):
        super(CustomHallwayEnv, self).__init__()
        self.env = gym.make("MiniWorld-Hallway-v0", domain_rand=True, render_mode=render_mode)
        self.env_description = """
        The Hallway environment is a 3D world consisting of one hallway.
        The agent has three discrete actions: turn left, turn right, move forward. 
        """
        self.task_description = """
        The agent needs to navigate to the red box. The episode ends when the agent reaches the red box.
        """
        self.render_mode = render_mode

        # self.relevant_params = ('sky_color', 'light_pos', 'light_color', 'light_ambient')
        self.relevant_params = tuple(self.env.unwrapped.params.params.keys())
        self.relevant_entity_attributes = ('color', 'size', 'position')
        self.all_features = self.relevant_params + self.relevant_entity_attributes + ('box_visible',)
        self.filter_flag = False

        self.action_space = self.env.action_space
        self.observation_space_dict = self.get_observation_space_dict()
        self.observation_space = gym.spaces.Dict(self.observation_space_dict)

    def get_observation_space_dict(self):
        param_bounds = {}
        for feature, bounds in self.env.unwrapped.params.params.items():
            try:
                getattr(self.env.unwrapped, feature)
            except AttributeError:
                continue
            if bounds.type == 'float':
                if isinstance(bounds.default, np.ndarray):
                    param_bounds[feature] = gym.spaces.Box(low=bounds.min, high=bounds.max, shape=(len(bounds.default),))
                elif isinstance(bounds.default, float):
                    param_bounds[feature] = gym.spaces.Box(low=bounds.min, high=bounds.max, shape=(1,))
            elif bounds.type == 'int':
                param_bounds[feature] = gym.spaces.Discrete(bounds.max - bounds.min)
            else:
                raise ValueError(f"Unsupported type: {bounds['type']}")

        param_bounds['box_visible'] = gym.spaces.Discrete(2)
        entities = self.env.unwrapped.entities
        for entity in entities:
            for attr in dir(entity):
                if not attr.startswith('_') and not callable(getattr(entity, attr)) and attr in ['size', 'pos', 'cam_pos', 'cam_dir', 'dir_vec']:
                    if 'pos' in attr:
                        param_bounds["_".join([entity.name, attr])] = gym.spaces.Box(
                            low=np.array((self.env.unwrapped.min_x, float('-inf'), self.env.unwrapped.min_z)),
                            high=np.array((self.env.unwrapped.max_x, float('inf'), self.env.unwrapped.max_z)),
                            shape=(3,),
                        )
                    else:
                        param_bounds["_".join([entity.name, attr])] = gym.spaces.Box(
                            low=np.array((float('-inf'), float('-inf'), float('-inf'))),
                            high=np.array((float('inf'), float('inf'), float('inf'))),
                            shape=(3,),
                        )
                elif not attr.startswith('_') and not callable(getattr(entity, attr)) and attr in ['color']:
                    param_bounds["_".join([entity.name, attr])] = gym.spaces.Discrete(len(COLOR_TO_IDX))
        observation_space_dict = {}
        for feature in param_bounds:
            new_feature = feature.replace('pos', 'position')
            new_feature = new_feature.replace('cam', 'camera')
            new_feature = new_feature.replace('dir', 'direction')
            new_feature = new_feature.replace('vec', 'vector')
            observation_space_dict[new_feature] = param_bounds[feature]
        return observation_space_dict

    def _get_observation(self):
        """
        Extracts the relevant features from the environment and organizes them in a dictionary
        """
        features = {}
        for feature in self.env.unwrapped.params.params.keys():
            try:
                features[feature] = np.array(getattr(self.env.unwrapped, feature), dtype=np.float32)
            except AttributeError:
                continue
        features['box_visible'] = np.array(int(bool(len(self.env.unwrapped.get_visible_ents()))), dtype=np.float32)
        entities = self.env.unwrapped.entities
        for entity in entities:
            for attr in dir(entity):
                if not attr.startswith('_') and not callable(getattr(entity, attr)) and attr in ['size', 'pos', 'cam_pos', 'cam_dir', 'dir_vec']:
                    features[entity.name + "_" + attr] = np.array(getattr(entity, attr), dtype=np.float32)
                elif not attr.startswith('_') and not callable(getattr(entity, attr)) and attr in ['color']:
                    features[entity.name + "_" + attr] = np.array(COLOR_TO_IDX[getattr(entity, attr)], dtype=np.float32)

        observation_dict = {}
        for feature in features:
            new_feature = feature.replace('pos', 'position')
            new_feature = new_feature.replace('cam', 'camera')
            new_feature = new_feature.replace('dir', 'direction')
            new_feature = new_feature.replace('vec', 'vector')
            observation_dict[new_feature] = features[feature]

        if self.filter_flag:
            observation_dict = {feature: value for feature, value in observation_dict.items() if feature in self.all_features}
        return observation_dict

    def select_feature_subset(self, features=None):
        """
        Selects a subset of the features to be used in the observation space
        """
        self.filter_flag = True
        self.all_features = features
        self.observation_space_dict = {k: v for k, v in self.observation_space_dict.items() if k in features}
        self.observation_space = gym.spaces.Dict(self.observation_space_dict)

    def reset(self, **kwargs):
        """
        Reset the environment and return the initial observation
        """
        _, _ = self.env.reset(**kwargs)
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        """
        Take a step in the environment
        """
        _, reward, terminated, truncated, info = self.env.step(action)
        observation = self._get_observation()
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
    env = CustomHallwayEnv(
        render_mode='rgb_array',
    )
