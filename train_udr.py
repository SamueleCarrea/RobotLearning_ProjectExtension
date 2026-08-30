"""Training script for Uniform Domain Randomization (UDR) with a recurrent policy

This script trains a recurrent policy (RecurrentPPO from sb3-contrib) on the
source environment with domain randomization enabled, then evaluates it on
both source and target environments to measure the effectiveness of UDR in
bridging the sim-to-real gap.
"""

import gymnasium as gym
import argparse
import numpy as np
import json
from datetime import datetime
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from env.custom_hopper import *

VECNORM_FILE = "vecnormalize.pkl"

# PPO feedforward serve come termine di paragone: stessa UDR, stessi
# iperparametri, l'unica differenza e' la memoria ricorrente
ALGOS = {"RecurrentPPO": RecurrentPPO, "PPO": PPO}
POLICIES = {"RecurrentPPO": "MlpLstmPolicy", "PPO": "MlpPolicy"}


def _make_udr_env(seed_offset, env_id="CustomHopper-source-udr-v0"):
    # factory a livello di modulo, serve per SubprocVecEnv
    def _init():
        env = Monitor(gym.make(env_id))
        env.reset(seed=seed_offset)
        env.action_space.seed(seed_offset)
        return env
    return _init


def _wrap(venv, training, normalize):
    if not normalize:
        return venv
    # in eval le stats sono congelate e il reward resta grezzo
    return VecNormalize(venv, training=training, norm_obs=True,
                        norm_reward=training, clip_obs=10.0)


def train_udr_model(
    algorithm="RecurrentPPO",
    timesteps=1000000,
    save_path="./models/udr/",
    tag="",
    seed=None,
    n_envs=1,
    normalize=True,
    udr_env="CustomHopper-source-udr-v0",
):
    """
    Train a recurrent policy using Uniform Domain Randomization

    Args:
        algorithm: 'RecurrentPPO' o 'PPO' (feedforward, per il confronto)
        timesteps: Total number of timesteps for training
        save_path: Directory to save the trained model
        tag: Optional suffix for tensorboard log dir, to keep runs separate
        n_envs: Numero di ambienti paralleli (SubprocVecEnv se >1).
        normalize: applica VecNormalize a osservazioni e reward.
    """
    base_seed = seed if seed is not None else 0

    # seed diverso per ogni env, altrimenti campionano tutti le stesse masse
    n_eval_envs = min(n_envs, 4)
    cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    train_env = _wrap(cls([_make_udr_env(base_seed + i, udr_env)
                           for i in range(n_envs)]), True, normalize)
    eval_env = _wrap(cls([_make_udr_env(base_seed + 1000 + i, udr_env)
                          for i in range(n_eval_envs)]), False, normalize)
    # stesso numero di transizioni per update al variare di n_envs
    n_steps = max(2048 // n_envs, 64)
    batch_size = 64
    while (n_steps * n_envs) % batch_size != 0:
        batch_size -= 1

    print("=" * 50)
    print("Training with Uniform Domain Randomization (UDR)")
    print("Base environment:", udr_env)
    print("State space:", train_env.observation_space)
    print("Action space:", train_env.action_space)
    print("Domain Randomization: ENABLED")
    print("Randomized parameters: thigh_mass, leg_mass, foot_mass")
    print("=" * 50)

    if algorithm == "RecurrentPPO":
        # LSTM piu' piccola ma separata per actor e critic: con la shared LSTM
        # i risultati erano peggiori
        policy_kwargs = dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
        )
    else:
        # PPO usa la net_arch di default [64, 64], la stessa che MlpLstmPolicy
        # mette attorno all'LSTM: cosi' l'unica differenza e' la ricorrenza
        policy_kwargs = None

    # Initialize the RL algorithm
    model = ALGOS[algorithm](
        POLICIES[algorithm],
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        device="cpu",
        seed=seed,
        tensorboard_log=f"./tensorboard/udr_{algorithm}{'_' + tag if tag else ''}/",
    )

    # Set up callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        # eval_freq e' per singolo env, va diviso per n_envs
        eval_freq=max(10000 // n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1), save_path=save_path,
        name_prefix=f"udr_{algorithm}"
    )

    # Train the model
    print(f"\nStarting UDR training with {algorithm} for {timesteps} timesteps...")
    print("During training, the environment will randomize dynamics at each reset.")
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save the final model
    model_name = f"{save_path}/udr_{algorithm}_final"
    model.save(model_name)
    if normalize:
        # senza queste stats il modello ricaricato vede obs con un'altra scala
        import os
        os.makedirs(save_path, exist_ok=True)
        train_env.save(os.path.join(save_path, VECNORM_FILE))
    print(f"\nFinal UDR model saved to: {model_name}")

    return model


def test_udr_model(model_path, env_name, n_episodes=50, vecnorm_path=None,
                   algorithm="RecurrentPPO"):
    """
    Test a UDR-trained model on a specific environment

    Args:
        model_path: Path to the trained UDR model
        env_name: Name of the environment to test on
        n_episodes: Number of episodes for evaluation

    Returns:
        mean_reward: Mean reward over all episodes
        std_reward: Standard deviation of rewards
    """
    # Load the model
    model = ALGOS[algorithm].load(model_path)

    # Create test environment (WITHOUT randomization for evaluation)
    test_env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])
    if vecnorm_path is not None:
        import os
        if not os.path.exists(vecnorm_path):
            # senza le stats non da' errore, da' solo reward molto piu' bassi
            raise FileNotFoundError(
                f"Atteso {vecnorm_path} ma non esiste. Usa --no-normalize se la "
                "policy e' stata allenata senza VecNormalize."
            )
        test_env = VecNormalize.load(vecnorm_path, test_env)
        test_env.training = False
        test_env.norm_reward = False

    print("=" * 50)
    print(f"Testing UDR model on: {env_name}")
    print("=" * 50)

    # Evaluate the policy (evaluate_policy natively handles recurrent
    # state / episode_start passing via model.predict)
    mean_reward, std_reward = evaluate_policy(
        model, test_env, n_eval_episodes=n_episodes, deterministic=True, render=False
    )

    print(f"\nResults over {n_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return mean_reward, std_reward


def main():
    parser = argparse.ArgumentParser(
        description="Train and test UDR recurrent policies on Hopper environments"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Mode: train, test, or both",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="RecurrentPPO",
        choices=["RecurrentPPO", "PPO"],
        help="RecurrentPPO (LSTM) oppure PPO feedforward, per il confronto",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000000, help="Total timesteps for training"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to model for testing"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=50, help="Number of episodes for testing"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix for save_path/results file, to avoid overwriting a previous run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed per rendere il training riproducibile e permettere run multipli confrontabili",
    )
    parser.add_argument(
        "--udr_env",
        type=str,
        default="CustomHopper-source-udr-v0",
        help="variante UDR: -udr5 / -udr / -udr25 / -udr50 (ampiezza del range)",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="VecNormalize su osservazioni e reward (default: attivo)",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=4,
        help="ambienti paralleli (SubprocVecEnv se >1)",
    )

    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    if args.mode in ["train", "both"]:
        # Train the UDR model
        model = train_udr_model(
            algorithm=args.algorithm,
            timesteps=args.timesteps,
            save_path=f"./models/udr{suffix}/",
            tag=args.tag,
            seed=args.seed,
            n_envs=args.n_envs,
            normalize=args.normalize,
            udr_env=args.udr_env,
        )
        model_path = f"./models/udr{suffix}/udr_{args.algorithm}_final"
    else:
        if args.model_path is None:
            raise ValueError("Please provide --model_path for testing mode")
        model_path = args.model_path

    if args.mode in ["test", "both"]:
        # Test on both source and target environments
        print("\n" + "=" * 50)
        print("TESTING PHASE - UDR MODEL")
        print("=" * 50)

        import os
        vecnorm_path = (os.path.join(f"./models/udr{suffix}/", VECNORM_FILE)
                        if args.normalize else None)

        print(f"\n1. Testing on SOURCE environment (CustomHopper-source-v0):")
        mean_source, std_source = test_udr_model(
            model_path, "CustomHopper-source-v0", args.n_episodes, vecnorm_path,
            args.algorithm
        )

        print(f"\n2. Testing on TARGET environment (CustomHopper-target-v0):")
        mean_target, std_target = test_udr_model(
            model_path, "CustomHopper-target-v0", args.n_episodes, vecnorm_path,
            args.algorithm
        )

        print("\n" + "=" * 50)
        print("SUMMARY - UDR MODEL PERFORMANCE")
        print("=" * 50)
        print(f"UDR → Source: {mean_source:.2f} +/- {std_source:.2f}")
        print(f"UDR → Target: {mean_target:.2f} +/- {std_target:.2f}")
        print("=" * 50)

        # Calculate performance difference
        performance_drop = mean_source - mean_target
        drop_percentage = (
            (performance_drop / mean_source) * 100 if mean_source > 0 else 0
        )

        print(f"\nPerformance Analysis:")
        print(f"   • Absolute drop: {performance_drop:.2f} points")
        print(f"   • Relative drop: {drop_percentage:.1f}%")

        # Save results to JSON file
        results = {
            "training_method": "Uniform Domain Randomization (UDR)",
            "training_env": args.udr_env,
            "algorithm": args.algorithm,
            "timesteps": args.timesteps,
            "n_test_episodes": args.n_episodes,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "seed": args.seed,
                "n_envs": args.n_envs,
                "normalize": args.normalize,
                # None per PPO: serve a summarize_results.py per non mettere
                # nello stesso gruppo run ricorrenti e feedforward
                "lstm_hidden_size": 128 if args.algorithm == "RecurrentPPO" else None,
            },
            "randomization_config": {
                "enabled": True,
                "method": "uniform",
                "parameters": ["thigh_mass", "leg_mass", "foot_mass"],
            },
            "results": {
                "source_to_source": {
                    "mean": float(mean_source),
                    "std": float(std_source),
                },
                "source_to_target": {
                    "mean": float(mean_target),
                    "std": float(std_target),
                },
                "performance_drop": {
                    "absolute": float(performance_drop),
                    "percentage": float(drop_percentage),
                },
            },
        }

        # Save to JSON file
        results_file = f"./results/udr_{args.algorithm}{suffix}_results.json"
        import os

        os.makedirs("./results", exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nResults saved to: {results_file}")

        # il confronto con le baseline lo fa summarize_results.py, che media
        # sui seed: confrontare due singoli run non dice niente, la varianza
        # tra seed e' piu' grande della differenza tra i metodi
        print("\nPer il confronto con le baseline: python summarize_results.py")

if __name__ == "__main__":
    main()