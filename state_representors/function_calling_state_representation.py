import torch
import os
import sys
from .base_state_representor import AbstractStateRepresentor
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


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

        self.llm = GoogleGenerativeAI(model="gemini-pro",
                                      google_api_key=self.api_key,
                                      temperature=0)
        self.setup_parser()
        self.prompt = self.create_prompt()

    def setup_parser(self):
        """Set up the output parser for structured JSON responses"""
        selected_features_schema = ResponseSchema(
            name="selected_features",
            description="Dictionary of selected features that are relevant for the task",
            type="dict"
        )

        reasoning_schema = ResponseSchema(
            name="reasoning",
            description="Explanation for why each feature was selected or excluded",
            type="str"
        )

        self.output_parser = StructuredOutputParser.from_response_schemas([
            selected_features_schema,
            reasoning_schema
        ])
        self.format_instructions = self.output_parser.get_format_instructions()

    def create_prompt(self):
        """Create the prompt for the LLM to select features"""
        template = """You are an AI tasked with selecting relevant features from an environment for a reinforcement learning agent.

Environment Description: {env_description}
Task Description: {task_distribution}

Given the current features dictionary: {features_dict}

Select only the features that are relevant for completing the task. Consider:
1. Which features directly affect the agent's ability to complete the task?
2. Which features provide necessary information about the environment state?
3. Which features can be safely ignored without impacting task performance?

{format_instructions}

Return:
1. A dictionary of selected features and their values
2. A brief explanation of why each feature was selected or excluded

Remember to maintain the original data types of the features (numbers, strings, booleans, etc)."""

        return PromptTemplate(
            template=template,
            input_variables=["features_dict"],
            partial_variables={
                "env_description": self.env_description,
                "task_distribution": self.task_distribution,
                "format_instructions": self.format_instructions
            }
        )

    def select_features(self, features_dict: str, **kwargs):
        """
        Select relevant features from the features dictionary
        Args:
            features_dict: Dictionary of all available features and their values
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary of selected features
        """
        # Convert dictionary to string if it isn't already
        if isinstance(features_dict, dict):
            features_dict = {k: repr(v) for k, v in features_dict.items()}
            features_dict = json.dumps(features_dict, indent=2)

        # Format the prompt with the features dictionary
        formatted_prompt = self.prompt.format(features_dict=features_dict)

        # Get response from LLM
        response = self.llm.invoke(formatted_prompt)

        try:
            # Parse the response into structured output
            parsed_output = self.output_parser.parse(response)
            selected_features = parsed_output["selected_features"]

            # Log the reasoning if debug information is requested
            if kwargs.get('debug', False):
                print("Selection reasoning:", parsed_output["reasoning"])

            return selected_features

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response}")
            # Return original features as fallback
            return features_dict


if __name__ == '__main__':
    # Example usage
    env_description = """
    A 2D grid world environment with dimensions 10x10. The agent can move in four directions: up, down, left, right.
    The environment contains various colored squares (red, blue, green) placed randomly on the grid.
    """

    task_description = """
    The agent needs to find and reach the red square while avoiding blue squares.
    The episode ends when the agent reaches the red square or hits a blue square.
    """

    # Example features dictionary
    features = {
        "agent_position": (2, 3),
        "agent_orientation": "north",
        "red_square_position": (8, 8),
        "blue_squares_positions": [(3, 3), (4, 6)],
        "green_squares_positions": [(1, 1), (5, 5)],
        "sky_color": "blue",
        "temperature": 22,
        "wind_speed": 5,
        "time_of_day": "noon",
        "visibility_radius": 5
    }

    # Initialize the state representor
    state_representor = Function_Calling_State_Representor(
        env_description=env_description,
        task_distribution=task_description
    )

    # Get selected features
    selected_features = state_representor.select_features(features, debug=True)
    print("\nSelected features:", selected_features.keys())
