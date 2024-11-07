from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoProcessor
import torch
import os
import sys
import typing

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import constants


class Zero_Shot_State_Representor:
    """
    Class to generate state representations using a foundation model
    """

    def __init__(
            self,
            zero_shot_classifier_model,
            env_description: str,
            task_distribution: str,
    ):
        """
        Initialize the State Representor
        Args:
            textual_representation_model: The dataclass model to use for generating textual state representations.
            env_description: Textual description of the environment.
            task_distribution: Textual description of the task distribution.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.zero_shot_classifier_model = zero_shot_classifier_model
        self.zero_shot_state_representation_model_pipe = None
        self.env_description = env_description
        self.task_distribution = task_distribution
        self.possible_labels = ['relevant', 'irrelevant']

    def generate_state_representation(self, features: typing.List[str]):
        # Use the foundation model to generate a state representation
        prompt_for_model = [
            constants.Prompts.ZERO_SHOT_CLASSIFIER_PROMPT.format(
                self.env_description,
                self.task_distribution,
                feature,
            ) for feature in features]
        classifier = pipeline(
            model=self.zero_shot_classifier_model.model_id,
            task=self.zero_shot_classifier_model.task,
        )
        relevant_labels = classifier(prompt_for_model, self.possible_labels)
        relevant_labels = [label['labels'][label['scores'].index(max(label['scores']))] for label in relevant_labels]
        features_and_labels = dict(zip(features, relevant_labels))
        return relevant_labels


if __name__ == '__main__':
    features = {
        0: 'Agent X position',
        1: 'Agent Y position',
        2: 'Distance from target in X axis (euclidean distance)',
        3: 'Distance from target in Y axis (euclidean distance)',
        4: 'Circle 0: Circle X position',
        5: 'Circle 0: Circle Y position',
        6: 'Circle 0: Circle radius',
        7: 'Circle 0: Circle Red color component (0-255)',
        8: 'Circle 0: Circle Green color component (0-255)',
        9: 'Circle 0: Circle Blue color component (0-255)',
        10: 'Circle 1: Circle X position',
        11: 'Circle 1: Circle Y position',
        12: 'Circle 1: Circle radius',
        13: 'Circle 1: Circle Red color component (0-255)',
        14: 'Circle 1: Circle Green color component (0-255)',
        15: 'Circle 1: Circle Blue color component (0-255)',
        16: 'Circle 2: Circle X position',
        17: 'Circle 2: Circle Y position',
        18: 'Circle 2: Circle radius',
        19: 'Circle 2: Circle Red color component (0-255)',
        20: 'Circle 2: Circle Green color component (0-255)',
        21: 'Circle 2: Circle Blue color component (0-255)',
        22: 'Circle 3: Circle X position',
        23: 'Circle 3: Circle Y position',
        24: 'Circle 3: Circle radius',
        25: 'Circle 3: Circle Red color component (0-255)',
        26: 'Circle 3: Circle Green color component (0-255)',
        27: 'Circle 3: Circle Blue color component (0-255)',
        28: 'Circle 4: Circle X position',
        29: 'Circle 4: Circle Y position',
        30: 'Circle 4: Circle radius',
        31: 'Circle 4: Circle Red color component (0-255)',
        32: 'Circle 4: Circle Green color component (0-255)',
        33: 'Circle 4: Circle Blue color component (0-255)'
    }

    env_description = constants.SimpleConstants.ENV_DESCRIPTION
    task_description = constants.SimpleConstants.TASK_DESCRIPTION

    state_representor = Zero_Shot_State_Representor(
        zero_shot_classifier_model=constants.Models.DeBERTa_v3_NLI,
        env_description=env_description,
        task_distribution=task_description,
    )
    state_representor.generate_state_representation(list(features.values()))
