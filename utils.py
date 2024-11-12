from dataclasses import dataclass, field
from typing import Type, Dict, Any, Union
from stable_baselines3 import PPO, DQN, A2C, SAC
from envs.custom_fourrooms_miniworld import CustomFourRoomsEnv
from envs.custom_hallway_miniworld import CustomHallwayEnv

def get_algorithm_map() -> Dict[str, Type]:
    return {
        'ppo': PPO,
        'dqn': DQN,
        'a2c': A2C,
        'sac': SAC,
    }

def get_env_map() -> Dict[str, Type]:
    return {
        'hallway': CustomHallwayEnv,
        'fourrooms': CustomFourRoomsEnv
    }

def get_default_configs() -> Dict[str, Dict[str, Any]]:
    return {
        'ppo': {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'verbose': 1
        },
        'dqn': {
            'learning_rate': 0.0001,
            'buffer_size': 1000000,
            'learning_starts': 50000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'verbose': 1
        },
        'a2c': {
            'learning_rate': 0.0007,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1
        },
    }



@dataclass
class Algorithm:
    """
    A dataclass for modular algorithm initialization.

    Usage:
        algo = Algorithm('ppo')(env=env, learning_rate=0.0003)
        # or
        algo_config = Algorithm('ppo')
        algo = algo_config(env=env, learning_rate=0.0003)
    """
    algorithm_name: str
    ALGORITHM_MAP: Dict[str, Type] = field(default_factory=get_algorithm_map)
    DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = field(default_factory=get_default_configs)

    def __post_init__(self):
        """Validate the algorithm name after initialization."""
        self.algorithm_name = self.algorithm_name.lower()
        if self.algorithm_name not in self.ALGORITHM_MAP:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm_name}. "
                f"Available algorithms: {list(self.ALGORITHM_MAP.keys())}"
            )

    def __call__(
            self,
            policy,
            env,
            **kwargs
    ) -> Any:
        """
        Initialize the algorithm with the given parameters.

        Args:
            env: The environment to use (can be a string or gym environment)
            policy: The policy architecture to use
            **kwargs: Additional arguments to pass to the algorithm

        Returns:
            Initialized algorithm instance
        """
        # Get the algorithm class
        algorithm_class = self.ALGORITHM_MAP[self.algorithm_name]

        # Get default config and update with provided kwargs
        config = self.DEFAULT_CONFIGS[self.algorithm_name].copy()
        config.update(kwargs)

        # Initialize and return the algorithm
        return algorithm_class(
            policy=policy,
            env=env,
            **config
        )

@dataclass
class Environment:
    """
    A dataclass for modular algorithm initialization.

    Usage:
        env = Environment('hallway')(render_mode='rgb_array')
    """
    env_name: str
    ENV_MAP: Dict[str, Type] = field(default_factory=get_env_map)

    def __post_init__(self):
        """Validate the environment name after initialization."""
        self.env_name = self.env_name.lower()
        if self.env_name not in self.ENV_MAP:
            raise ValueError(
                f"Unknown environment: {self.env_name}. "
                f"Available environment: {list(self.ENV_MAP.keys())}"
            )

    def __call__(
            self,
            render_mode: str = 'rgb_array',
            **kwargs
    ) -> Any:
        """
        Initialize the environment with the given parameters.

        Args:
            render_mode: The render mode for the environment
            **kwargs: Additional arguments to pass to the environment

        Returns:
            Initialized environment instance
        """
        # Get the algorithm class
        algorithm_class = self.ENV_MAP[self.env_name]
        # Initialize and return the algorithm
        return algorithm_class(
            render_mode=render_mode,
            **kwargs
        )


