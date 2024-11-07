from abc import ABC, abstractmethod


class AbstractStateRepresentor(ABC):
    """
    Abstract base class for generating state representations using a foundation model.
    """

    def __init__(self, env_description: str, task_distribution: str):
        """
        Initialize the State Representor.

        Args:
            env_description (str): Textual description of the environment.
            task_distribution (str): Textual description of the task distribution.
        """
        self.env_description = env_description
        self.task_distribution = task_distribution

    @abstractmethod
    def generate_state_representation(self, obs_description: str, **kwargs) -> str:
        """
        Generate a state representation using a foundation model.

        Args:
            obs_description (str): Textual description of the observation.

        Returns:
            str: The generated state representation.
        """
        pass
