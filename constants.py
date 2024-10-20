from dataclasses import dataclass


@dataclass
class SimpleConstants:
    ENV_NAME = "Simple-v3"
    ENV_DESCRIPTION = """The environment is a Simple, single agent environment, where a single particle moves in a 2D 
    plane."""
    TASK_DESCRIPTION = """The agent can move in any direction, and the goal is to reach the target position, marked by
    a red circle which has a fixed location."""


@dataclass
class Prompts:
    STATE_REPRESENTATION_PROMPT = """You are an agent operating in an environment, described as follows:
            - Environment: {}
            Your objective is to complete tasks from the following distribution:
            - Task Distribution: {}
            Given the all this information, specify what is relevant for the agent to solve the task and how should it
             be represented"""

    CODE_GENERATION_PROMPT = """Given the described observation space, generate the code to create the observation space 
    in OpenAI Gym. Specify only the object that is meant to describe the observation space in a concise manner"""


@dataclass(frozen=True)
class LLaVaModel:
    model_id = "llava-hf/llava-1.5-7b-hf"
    task: str = "image-to-text"


@dataclass(frozen=True)
class LLaMaModel:
    model_id = "meta-llama/Meta-Llama-3-8B"
    task: str = "text-generation"


@dataclass(frozen=True)
class DeepSeekCoderModel:
    model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    task: str = "text-generation"


@dataclass(frozen=True)
class Models:
    LLaVa = LLaVaModel()
    LLaMa = LLaMaModel()
    DeepSeekCoder = DeepSeekCoderModel()
