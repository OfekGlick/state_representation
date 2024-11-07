import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from gymnasium import spaces


class CustomSimpleEnv(gym.Env):
    def __init__(self, env, render_mode='rgb_array'):
        self.env = env
        self.base_action_space = env.action_space(env.possible_agents[0])
        self.base_observation_space = env.observation_space(env.possible_agents[0])
        self.render_mode = render_mode

        # Initialize circles data structure
        self.num_circles = 5
        self.circles = []  # Will store (x, y, radius, r, g, b) for each circle

        self._create_observation_docs()
        # Create extended observation space including circles
        self._update_observation_space()

        self.action_space = self.base_action_space
        self.add_random_circles()

    def add_random_circles(self):
        self.circles = []
        for _ in range(self.num_circles):
            # Random position
            x = random.randint(0, 800)  # Assuming 800x600 environment
            y = random.randint(0, 600)

            # Random radius between 5 and 50
            radius = random.randint(5, 50)

            # Random color (RGB)
            color = (
                random.randint(125, 175),
                random.randint(125, 175),
                random.randint(125, 175)
            )

            self.circles.append((x, y, radius, *color))

    def _create_observation_docs(self):
        """Create documentation for the observation space indices"""
        # Base observation documentation
        self.base_obs_docs = {
            0: "Agent X position",
            1: "Agent Y position",
            2: "Distance from target in X axis (euclidean distance)",
            3: "Distance from target in Y axis (euclidean distance)"
        }

        # Circle observation documentation
        self.circle_obs_docs = {
            0: "Circle X position",
            1: "Circle Y position",
            2: "Circle radius",
            3: "Circle Red color component (0-255)",
            4: "Circle Green color component (0-255)",
            5: "Circle Blue color component (0-255)"
        }

        # Full observation documentation
        self.obs_docs = {}

        # Add base observation documentation
        self.obs_docs.update(self.base_obs_docs)

        # Add circle documentation for each circle
        for i in range(self.num_circles):
            circle_start_idx = 4 + i * 6
            for j, desc in self.circle_obs_docs.items():
                self.obs_docs[circle_start_idx + j] = f"Circle {i}: {desc}"

    def _update_observation_space(self):
        """Internal method to update the observation space based on current settings"""
        circle_low = np.array([0, 0, 5, 0, 0, 0] * self.num_circles)
        circle_high = np.array([800, 600, 50, 255, 255, 255] * self.num_circles)

        if isinstance(self.base_observation_space, spaces.Box):
            self.observation_space = spaces.Box(
                low=np.concatenate([self.base_observation_space.low, circle_low]),
                high=np.concatenate([self.base_observation_space.high, circle_high]),
                dtype=np.float32
            )

        # Update the documentation when observation space changes
        self._create_observation_docs()

    def get_observation_info(self):
        """
        Get a dictionary mapping observation indices to their descriptions

        Returns:
            Dict[int, str]: Dictionary mapping index to description
        """
        return self.obs_docs

    def get_circle_info(self, observation: np.ndarray):
        """
        Extract and document circle information from an observation

        Args:
            observation (np.ndarray): The full observation array

        Returns:
            Tuple[np.ndarray, List[Dict]]: Base observation and list of circle dictionaries
        """
        base_obs = observation[:4]
        circles_obs = observation[4:].reshape(-1, 6)

        # Create documented circle information
        circles_info = []
        for i, circle in enumerate(circles_obs):
            circles_info.append({
                "x": circle[0],
                "y": circle[1],
                "radius": circle[2],
                "color": {
                    "r": circle[3],
                    "g": circle[4],
                    "b": circle[5]
                },
                "observation_indices": {
                    "x": 4 + i * 6,
                    "y": 5 + i * 6,
                    "radius": 6 + i * 6,
                    "r": 7 + i * 6,
                    "g": 8 + i * 6,
                    "b": 9 + i * 6
                }
            })

        return base_obs, circles_info

    def print_observation_structure(self, observation: np.ndarray = None):
        """
        Print the structure of the observation space with optional current values

        Args:
            observation (np.ndarray, optional): Current observation to show values
        """
        print("\nObservation Space Structure:")
        print("=" * 50)

        # Print base observation structure
        print("\nBase Agent State:")
        for idx, desc in self.base_obs_docs.items():
            if observation is not None:
                print(f"Index {idx:3d}: {desc:30s} = {observation[idx]:.2f}")
            else:
                print(f"Index {idx:3d}: {desc}")

        # Print circles structure
        print("\nCircles Information:")
        for i in range(self.num_circles):
            print(f"\nCircle {i}:")
            base_idx = 4 + i * 6
            for j, desc in self.circle_obs_docs.items():
                idx = base_idx + j
                if observation is not None:
                    print(f"Index {idx:3d}: {desc:30s} = {observation[idx]:.2f}")
                else:
                    print(f"Index {idx:3d}: {desc}")

    def reset(self, seed=None, **kwargs):
        """
        Reset the environment and return the initial observation
        Args:
            seed (int, optional): Seed for the random number generator
        Returns:
            np.ndarray: Initial observation
        """
        self.env.reset()
        base_obs = self.env.observe(self.env.possible_agents[0])
        # Add circles to observation
        circle_obs = np.array([val for circle in self.circles for val in circle], dtype=np.float32)
        full_obs = np.concatenate([base_obs, circle_obs])
        return full_obs, {}

    def step(self, action):
        self.env.step(action)
        base_obs = self.env.observe(self.env.possible_agents[0])
        reward = self.env.rewards[self.env.possible_agents[0]]
        terminated = self.env.terminations[self.env.possible_agents[0]]
        truncated = self.env.truncations[self.env.possible_agents[0]]
        info = self.env.infos[self.env.possible_agents[0]]

        # Add circles to observation
        circle_obs = np.array([val for circle in self.circles for val in circle], dtype=np.float32)
        full_obs = np.concatenate([base_obs, circle_obs])

        return full_obs, reward, terminated, truncated, info

    def render(self, mode='rgb_array'):
        base_render = self.env.render()

        # Create a copy of the base render to draw circles on
        if isinstance(base_render, np.ndarray):
            frame = base_render.copy()

            # Draw each circle on the frame
            for x, y, radius, r, g, b in self.circles:
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    int(radius),
                    (int(b), int(g), int(r)),  # OpenCV uses BGR
                    -1  # Fill the circle
                )

            return frame
        return base_render

    @staticmethod
    def display_image(image: np.ndarray):
        plt.imshow(image)
        plt.axis('off')
        plt.show()
