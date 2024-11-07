from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoProcessor
import torch
import gymnasium as gym
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import constants


class Textual_State_Representor:
    """
    Class to generate state representations using a foundation model
    """

    def __init__(
            self,
            textual_representation_model,
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
        self.textual_state_representation_model_id = textual_representation_model.model_id
        self.textual_state_representation_task = textual_representation_model.task
        self.textual_state_representation_model_pipe = None
        self.env_description = env_description
        self.task_distribution = task_distribution

    def generate_state_representation(self, obs_description: str) -> str:
        # Use the foundation model to generate a state representation
        prompt_for_model = constants.Prompts.TEXTUAL_STATE_REPRESENTATION_PROMPT.format(
            self.env_description,
            self.task_distribution,
            obs_description
        )
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(constants.Models.LLaMaInstruct.model_id, trust_remote_code=True, )
        model = AutoModelForCausalLM.from_pretrained(constants.Models.LLaMaInstruct.model_id, trust_remote_code=True, )
        # Initialize the pipeline with the model and tokenizer
        text_representation_pipe = pipeline(
            task=self.textual_state_representation_task,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            return_full_text=True,
            device=self.device,
            torch_dtype='bfloat16',
        )
        terminators = [
            text_representation_pipe.tokenizer.eos_token_id,
            text_representation_pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        text_representation = text_representation_pipe(
            prompt_for_model,
            eos_token_id=terminators,
        )
        print(text_representation[0]['generated_text'])
        return text_representation[0]['generated_text']

    def create_gym_observation_space(self, description: str) -> gym.Space:
        if description.startswith("Box"):
            # Example description: "Box(low: float, high: float, shape=Tuple[int,...,int])"
            params_str = description[4:].strip("()")
            params = eval(f"dict({params_str})")
            return gym.spaces.Box(**params)
        elif description.startswith("Discrete"):
            # Example description: "Discrete(n=10)"
            params_str = description[9:].strip("()")
            params = eval(f"dict({params_str})")
            return gym.spaces.Discrete(**params)
        elif description.startswith("MultiDiscrete"):
            # Example description: "MultiDiscrete(nvec=[5, 2, 2])"
            params_str = description[14:].strip("()")
            params = eval(f"dict({params_str})")
            return gym.spaces.MultiDiscrete(**params)
        elif description.startswith("MultiBinary"):
            # Example description: "MultiBinary(n=10)"
            params_str = description[12:].strip("()")
            params = eval(f"dict({params_str})")
            return gym.spaces.MultiBinary(**params)
        else:
            raise ValueError(f"Unknown observation space type: {description}")


if __name__ == '__main__':
    obervation_repr = """Observation Space Structure:
    ==================================================
    
    Base Agent State:
    Index   0: Agent X position
    Index   1: Agent Y position
    Index   2: Distance from target location in X axis (euclidean distance)
    Index   3: Distance from target location in Y axis (euclidean distance)
    
    Circles Information:
    
    Circle 0:
    Index   4: Circle X position
    Index   5: Circle Y position
    Index   6: Circle radius
    Index   7: Circle Red color component (0-255)
    Index   8: Circle Green color component (0-255)
    Index   9: Circle Blue color component (0-255)
    
    Circle 1:
    Index  10: Circle X position
    Index  11: Circle Y position
    Index  12: Circle radius
    Index  13: Circle Red color component (0-255)
    Index  14: Circle Green color component (0-255)
    Index  15: Circle Blue color component (0-255)
    
    Circle 2:
    Index  16: Circle X position
    Index  17: Circle Y position
    Index  18: Circle radius
    Index  19: Circle Red color component (0-255)
    Index  20: Circle Green color component (0-255)
    Index  21: Circle Blue color component (0-255)
    
    Circle 3:
    Index  22: Circle X position
    Index  23: Circle Y position
    Index  24: Circle radius
    Index  25: Circle Red color component (0-255)
    Index  26: Circle Green color component (0-255)
    Index  27: Circle Blue color component (0-255)
    
    Circle 4:
    Index  28: Circle X position
    Index  29: Circle Y position
    Index  30: Circle radius
    Index  31: Circle Red color component (0-255)
    Index  32: Circle Green color component (0-255)
    Index  33: Circle Blue color component (0-255)"""

    env_description = constants.SimpleConstants.ENV_DESCRIPTION
    task_description = constants.SimpleConstants.TASK_DESCRIPTION

    state_representor = Textual_State_Representor(
        textual_representation_model=constants.Models.LLaMaInstruct,
        env_description=env_description,
        task_distribution=task_description,
    )
    state_representor.generate_state_representation(obervation_repr)
