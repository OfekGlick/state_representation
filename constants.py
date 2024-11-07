from dataclasses import dataclass


@dataclass
class SimpleConstants:
    ENV_NAME = "Simple-v3"
    ENV_DESCRIPTION = """The environment is a Simple Particle Environment, where a single particle moves in a 2D plane.
    There are random circles in the environment but they have no effect on the agent at all."""
    TASK_DESCRIPTION = """The agent can move up, down, left and right in different speeds.
     The goal is to reach the target position, marked by a red circle which has a fixed location."""


@dataclass
class Prompts:
    TEXTUAL_STATE_REPRESENTATION_PROMPT = """Given the following information, choose what features are actually necessary for the agent to accomplish its task.:
                                Environment:
                                {}
                                Tasks:
                                {}
                                Possible Features:
                                {}
                                The only features the agent needs to accomplish its task are:
                                """

    ZERO_SHOT_CLASSIFIER_PROMPT = """Given the following information, choose the most relevant label for the agent's task:
    Environment:
    {}
    Tasks:
    {}
    Feature:
    {}
    """
    CODE_GENERATION_PROMPT = """Given the described observation space, generate the code to create the observation space 
    in OpenAI Gym."""
    GYM_SPACE_PROMPT = """A Gymnasium Space could be one of the following:
    Box - Supports continuous (and discrete) vectors or matrices, used for vector observations, images, etc
    Discrete - Supports a single discrete number of values with an optional start for the values
    MultiBinary - Supports single or matrices of binary values, used for holding down a button or if an agent has an object
    MultiDiscrete - Supports multiple discrete values with multiple axes, used for controller actions"""


@dataclass(frozen=True)
class LLaVaModel:
    model_id = "llava-hf/llava-1.5-7b-hf"
    task: str = "image-to-text"

@dataclass(frozen=True)
class DeBERTa_v3_NLI_Model:
    model_id = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    task: str = "zero-shot-classification"


@dataclass(frozen=True)
class LLaMaModel:
    model_id = "meta-llama/Meta-Llama-3-8B"
    task: str = "text-generation"

@dataclass(frozen=True)
class LLaMaModelInstruct:
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    task: str = "text-generation"

@dataclass(frozen=True)
class DeepSeekCoderModel:
    model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    task: str = "text-generation"



@dataclass(frozen=True)
class Models:
    LLaVa = LLaVaModel()
    LLaMa = LLaMaModel()
    LLaMaInstruct = LLaMaModelInstruct()
    DeBERTa_v3_NLI = DeBERTa_v3_NLI_Model()
    DeepSeekCoder = DeepSeekCoderModel()
