from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoProcessor
import torch
import gymnasium as gym
import numpy as np
from PIL import Image
import constants


class Visual_State_Representor:
    """
    Class to generate state representations using a foundation model
    """

    def __init__(
            self,
            textual_representation_model,
            code_representation_model,
            env_description: str,
            task_distribution: str,
    ):
        """
        Initialize the State Representor
        Args:
            textual_representation_model: The dataclass model to use for generating textual state representations.
            code_representation_model: The dataclass model to use for generating code state representations.
            env_description: Textual description of the environment.
            task_distribution: Textual description of the task distribution.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.textual_state_representation_model_id = textual_representation_model.model_id
        self.textual_state_representation_task = textual_representation_model.task
        self.code_state_representation_model_id = code_representation_model.model_id
        self.code_state_representation_task = code_representation_model.task
        self.textual_state_representation_model_pipe = None
        self.code_state_representation_model = None
        self.env_description = env_description
        self.task_distribution = task_distribution

    def generate_textual_state_representation(self, observation: np.ndarray) -> np.ndarray:
        # Use the foundation model to generate a state representation
        self.textual_state_representation_model_pipe = pipeline(
            task=self.textual_state_representation_task,
            model=self.textual_state_representation_model_id,
            device=self.device,
            torch_dtype='bfloat16',
            trust_remote_code=True,
        )
        prompt_for_model = constants.Prompts.TEXTUAL_STATE_REPRESENTATION_PROMPT.format(
            self.env_description,
            self.task_distribution,
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": prompt_for_model},
                    {"type": "image"},
                ],
            },
        ]
        processor = AutoProcessor.from_pretrained(self.textual_state_representation_model_id)
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
        observation = Image.fromarray(observation)
        observation = observation.convert('RGB')
        state_representation = self.textual_state_representation_model_pipe(
            observation,
            prompt=prompt,
            generate_kwargs={"max_new_tokens": 200},
        )
        self.textual_state_representation_model_pipe.model.to('cpu')
        return state_representation[0]['generated_text']

    def generate_code_state_representation(self, description: str) -> str:
        self.code_state_representation_model = pipeline(
            task=self.code_state_representation_task,
            model=self.code_state_representation_model_id,
            device=self.device,
            torch_dtype='bfloat16',
            trust_remote_code=True,
        )
        # Use the foundation model to generate a state representation
        prompt_for_model = constants.Prompts.CODE_GENERATION_PROMPT.format(description)
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(constants.Models.DeepSeekCoder.model_id, trust_remote_code=True, )
        model = AutoModelForCausalLM.from_pretrained(constants.Models.DeepSeekCoder.model_id, trust_remote_code=True, )
        # Initialize the pipeline with the model and tokenizer
        code_representation_pipe = pipeline(
            task=self.code_state_representation_model.task,
            model=model,
            tokenizer=tokenizer,
            max_length=128,
            return_full_text=True,
        )
        code_representation = code_representation_pipe(prompt_for_model)
        return code_representation[0]['generated_text']

    def create_gym_observation_space(self, description: str) -> gym.Space:
        if description.startswith("Box"):
            # Example description: "Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)"
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
