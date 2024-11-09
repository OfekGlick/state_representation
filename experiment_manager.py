import gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env


class ExperimentManager:
    def __init__(
            self,
            env_id,
            log_dir="./logs",
            model_save_path="./models",
    ):
        self.env_id = env_id
        self.log_dir = log_dir
        self.model_save_path = model_save_path
        self.env = None
        self.model = None

    def create_env(self):
        self.env = make_vec_env(self.env_id, n_envs=1)

    def create_model(self, algorithm, params):
        assert 'policy' in params.keys(), "Policy must be specified in the params"
        assert params['policy'] in ['MlpPolicy', 'MultiInputPolicy', 'CnnPolicy'], "Policy must be one of [MlpPolicy, MultiInputPolicy, CnnPolicy]"
        assert isinstance(algorithm, (PPO, DQN, A2C)), "Algorithm must be one of [PPO, DQN, A2C]"

        self.model = algorithm("MlpPolicy", self.env, tensorboard_log=self.log_dir)

    def train(self,total_timesteps=1_000_000,):
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_save_path)

    def run_experiment(self):
        self.create_env()
        self.create_model()
        self.train()


if __name__ == "__main__":
    manager = ExperimentManager(env_id="custom_fourrooms_miniworld")
    manager.run_experiment()
