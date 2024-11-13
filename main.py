import argparse
from experiment_manager import ExperimentManager

def main():
    """
    Main entry point for running experiments.
    """
    parser = argparse.ArgumentParser(description='Run training with different feature sets.')
    parser.add_argument(
        '--run_type',
        type=str,
        choices=['all', 'llm', 'manual'],
        default='all',
        help='Specify which run to execute: all features, LLM based features, or manual features.'
    )
    parser.add_argument(
        '--algorithm_name',
        type=str,
        choices=['dqn', 'ppo', 'a2c'],
        default='ppo',
        help='Specify which RL algorithm to use (default: PPO).'
    )
    parser.add_argument(
        '--env_name',
        type=str,
        choices=['fourrooms', 'hallway'],
        default='hallway',
        help='Specify which environment to use (default: hallway).'
    )
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=1_000_000,
        help='Specify the total number of timesteps to train for (default: 1,000,000).'
    )
    args = parser.parse_args()

    # Create experiment manager and run experiment
    experiment_manager = ExperimentManager(
        env_name=args.env_name,
        algorithm_name=args.algorithm_name,
        total_timesteps=args.total_timesteps,
    )
    experiment_manager.run_experiment(run_type=args.run_type)


if __name__ == '__main__':
    main()