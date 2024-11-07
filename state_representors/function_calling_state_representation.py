from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoProcessor
import torch
import gymnasium as gym
import os
import sys
from base_state_representor import AbstractStateRepresentor
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import constants


class Function_Calling_State_Representor(AbstractStateRepresentor):
    """
    Class to generate state representations using a foundation model
    """
    def __init__(self, env_description: str, task_distribution: str, api_key: str = None):
        """
        Initialize the State Representor
        Args:
            env_description: Textual description of the environment.
            task_distribution: Textual description of the task distribution.
        """
        super().__init__(env_description, task_distribution)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.env_description = env_description
        self.task_distribution = task_distribution
        self.api_key = api_key
        if not self.api_key:
            try:
                self.api_key = os.environ["GOOGLE_API_KEY"]
            except:
                raise ValueError("Please set GOOGLE_API_KEY environment variable")

    def setup_parser(self):
        """Set up the output parser for structured JSON responses"""
        response_schemas = [
            ResponseSchema(
                name="objects",
                description="A dictionary where keys are object names and values are [x,y] coordinate pairs",
            )
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)

    def create_prompt(self):
        """Create the prompt template for image description parsing"""
        template = """
        You are an expert at inferring which semantic information is relevant to performing tasks.
        Given a an observation, a description of an environment, and a description of a task - extract all relevant 
        objects. 
        Convert descriptions in natural language to a PDDL representation.
                
        The description: {image_description}

        {format_instructions}

        Return ONLY a valid JSON object.
        """
        return PromptTemplate(
            template=template,
            input_variables=["image_description"],
            partial_variables={"format_instructions": self.setup_parser().get_format_instructions()}
        )

    def parse_image_description(self, description: str) -> dict:
        """
        Parse an image description and return object coordinates

        Args:
            description: Natural language description of image

        Returns:
            Dictionary mapping object names to their [x,y] coordinates
        """
        # Initialize components
        llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.api_key,
            temperature=0.1  # Lower temperature for more consistent outputs
        )
        prompt = self.create_prompt()
        parser = self.setup_parser()

        # Generate prompt with image description
        formatted_prompt = prompt.format(image_description=description)

        try:
            # Get response from LLM
            response = llm.invoke(formatted_prompt)
            # Parse the response into structured format
            parsed_output = parser.parse(response)
            return parsed_output["objects"]
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {}

    def generate_state_representation(self, obs_description: str, **kwargs):
        coordinates = self.parse_image_description(obs_description)
        print(f"Coordinates: {coordinates}")

    def create_gym_observation_space(self, description: str, **kwargs) -> gym.Space:
        pass


if __name__ == '__main__':
    description = """
        A large oak tree stands in the center of the image. 
        There's a red bird perched on a branch in the top right corner.
        A small rabbit sits near the base of the tree, slightly to the left.
        In the far background, there's a wooden fence running along the bottom.
        """

    env_description = constants.SimpleConstants.ENV_DESCRIPTION
    task_description = constants.SimpleConstants.TASK_DESCRIPTION

    state_representor = Function_Calling_State_Representor(
        env_description=env_description,
        task_distribution=task_description,
        # api_key=os.getenv("GOOGLE_API_KEY")
    )
    state_representor.generate_state_representation(description)
