from typing import List, Type, Optional
from pathlib import Path
from gymnasium.wrappers import RecordVideo
from tqdm import trange
from utils import Algorithm, Environment
from state_representors.function_calling_state_representation import Function_Calling_State_Representor
import wandb
import os
from wandb.integration.sb3 import WandbCallback


class ExperimentManager:
    """
    A class to manage reinforcement learning experiments with different feature sets.
    """

    def __init__(
            self,
            env_name: str,
            algorithm_name: str,
            base_log_dir: str = "results/tensorboard_logs",
            base_model_dir: str = "results/models",
            base_video_dir: str = "results/video_dir",
            total_timesteps: int = 1_000_000
    ):
        self.env_name = env_name
        self.env = None
        self.algorithm_name = algorithm_name
        self.base_log_dir = Path(base_log_dir)
        self.base_model_dir = Path(base_model_dir)
        self.base_video_dir = Path(base_video_dir)
        self.total_timesteps = total_timesteps

        # Create directories if they don't exist
        for directory in [self.base_log_dir, self.base_model_dir, self.base_video_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_environment(
            self,
            run_type: str,
    ) -> tuple[List[str], str, str, str]:
        """
        Configure the environment based on the specified run type.

        Args:
            run_type: One of 'all', 'llm', or 'manual'

        Returns:
            Tuple of (features, log_name, model_save_path, video_dir)
        """
        self.env = Environment(self.env_name)(render_mode='rgb_array')
        to_log_name = '{}_{}_{}_features'.format(self.algorithm_name, self.env_name, run_type)
        model_save_name = str(self.base_model_dir / '{}_{}_{}'.format(self.algorithm_name, self.env_name, run_type))
        video_dir = str(self.base_video_dir / '{}_{}_{}_features'.format(self.algorithm_name, self.env_name, run_type))
        if run_type == 'all':
            features = list(self.env.get_observation_space_dict().keys())
        elif run_type == 'llm':
            state_representor = Function_Calling_State_Representor(
                env_description=self.env.env_description,
                task_distribution=self.env.task_description
            )
            features = self.env.get_observation_space_dict()
            selected_features = list(state_representor.select_features(features, debug=True))
            self.env.select_feature_subset(features=selected_features)
            features = selected_features

        elif run_type == 'manual':
            features = ['Box_position', 'Agent_position', 'box_visible', 'Agent_direction_vector']
            self.env.select_feature_subset(features=features)
        else:
            raise ValueError("Invalid run type. Please select one of 'all', 'llm', or 'manual'.")

        return features, to_log_name, model_save_name, video_dir

    def train(
            self,
            tensorboard_logs_dir: str,
            to_log_name: str,
            model_save_name: str,
            video_dir: str,
            features: List[str]
    ) -> None:
        """
        Train the model and record a video of its performance.
        """

        config = {
            'env_name': self.env_name,
            'algorithm_name': self.algorithm_name,
            'total_timesteps': self.total_timesteps,
            'features': features,
        }
        run = wandb.init(
            project='rl-feature-selection',
            name=to_log_name,
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
        )

        # Initialize and train the model
        model = Algorithm(self.algorithm_name)(
            policy="MultiInputPolicy",
            env=self.env,
            verbose=1,
            tensorboard_log=os.path.join(tensorboard_logs_dir, run.id),
        )

        model.learn(total_timesteps=self.total_timesteps, log_interval=4, tb_log_name=to_log_name)
        model.save(model_save_name)

        # Record video of trained model
        env = RecordVideo(self.env, video_folder=video_dir, episode_trigger=lambda episode_id: True)
        obs, info = env.reset()
        for _ in trange(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()

        env.close()
        run.finish()

    def run_experiment(self, run_type: str) -> None:
        """
        Run a complete experiment with the specified configuration.

        Args:
            run_type: One of 'all', 'llm', or 'manual'
        """
        # Setup environment and get configuration
        features, to_log_name, model_save_name, video_dir = self.setup_environment(
            run_type=run_type,
        )
        print("Features selected: ", features)
        # Run training
        self.train(
            tensorboard_logs_dir=str(self.base_log_dir),
            to_log_name=to_log_name,
            model_save_name=model_save_name,
            video_dir=video_dir,
            features=features
        )


