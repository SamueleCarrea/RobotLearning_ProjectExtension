"""Baseline senza domain randomization: allena su source oppure su target
e valuta su entrambi. Serve per il confronto con la policy UDR.
"""

import argparse
import json
import os
from datetime import datetime

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env.custom_hopper import *

VECNORM_FILE = "vecnormalize.pkl"

# stesse due varianti di train_udr.py, cosi' il confronto ricorrente vs
# feedforward si puo' fare sia con che senza randomizzazione
ALGOS = {"RecurrentPPO": RecurrentPPO, "PPO": PPO}
POLICIES = {"RecurrentPPO": "MlpLstmPolicy", "PPO": "MlpPolicy"}


def _make_env(env_id, seed):
    def _init():
        env = Monitor(gym.make(env_id))
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _init


def _wrap(venv, training, normalize):
    if not normalize:
        return venv
    return VecNormalize(venv, training=training, norm_obs=True,
                        norm_reward=training, clip_obs=10.0)


def train_baseline(env_id, timesteps, save_path, tag, seed, n_envs, normalize,
                   algorithm="RecurrentPPO"):
    os.makedirs(save_path, exist_ok=True)
    cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    train_env = _wrap(cls([_make_env(env_id, seed + i) for i in range(n_envs)]),
                      True, normalize)
    eval_env = _wrap(cls([_make_env(env_id, seed + 1000 + i)
                          for i in range(min(n_envs, 4))]), False, normalize)

    print("=" * 50)
    print(f"Baseline (no randomization) su {env_id}")
    print(f"algoritmo={algorithm}  n_envs={n_envs}  normalize={normalize}  seed={seed}")
    print("=" * 50)

    n_steps = max(2048 // n_envs, 64)
    batch_size = 64
    while (n_steps * n_envs) % batch_size != 0:
        batch_size -= 1

    if algorithm == "RecurrentPPO":
        policy_kwargs = dict(lstm_hidden_size=128, n_lstm_layers=1,
                             shared_lstm=False, enable_critic_lstm=True)
    else:
        # net_arch di default [64, 64], la stessa che MlpLstmPolicy usa attorno
        # all'LSTM: l'unica differenza tra i due e' la ricorrenza
        policy_kwargs = None

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
        tensorboard_log=f"./tensorboard/baseline_{algorithm}{'_' + tag if tag else ''}/",
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[
            EvalCallback(eval_env, best_model_save_path=save_path, log_path=save_path,
                         eval_freq=max(10000 // n_envs, 1), deterministic=True,
                         n_eval_episodes=10),
            CheckpointCallback(save_freq=max(50000 // n_envs, 1), save_path=save_path,
                               name_prefix=f"baseline_{algorithm}"),
        ],
        progress_bar=True,
    )

    model_path = os.path.join(save_path, f"baseline_{algorithm}_final")
    model.save(model_path)
    if normalize:
        train_env.save(os.path.join(save_path, VECNORM_FILE))
    print(f"\nModello salvato in {model_path}")
    train_env.close()
    eval_env.close()
    return model_path


def test_model(model_path, env_id, n_episodes, seed, vecnorm_path,
               algorithm="RecurrentPPO"):
    model = ALGOS[algorithm].load(model_path, device="cpu")
    venv = DummyVecEnv([_make_env(env_id, seed)])
    if vecnorm_path is not None:
        if not os.path.exists(vecnorm_path):
            raise FileNotFoundError(
                f"{vecnorm_path} non esiste. Usa --no-normalize se la policy "
                "e' stata allenata senza VecNormalize."
            )
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    mean_r, std_r = evaluate_policy(model, venv, n_eval_episodes=n_episodes,
                                    deterministic=True)
    print(f"  {env_id}: {mean_r:.2f} +/- {std_r:.2f}")
    venv.close()
    return float(mean_r), float(std_r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0",
                        choices=["CustomHopper-source-v0", "CustomHopper-target-v0"])
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--algorithm", choices=["RecurrentPPO", "PPO"],
                        default="RecurrentPPO",
                        help="RecurrentPPO (LSTM) oppure PPO feedforward")
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    save_path = f"./models/baseline{suffix}/"

    if args.mode in ("train", "both"):
        model_path = train_baseline(args.env, args.timesteps, save_path, args.tag,
                                    args.seed, args.n_envs, args.normalize,
                                    args.algorithm)
    else:
        if args.model_path is None:
            raise ValueError("In modalita' test serve --model_path")
        model_path = args.model_path

    if args.mode not in ("test", "both"):
        return

    vecnorm_dir = os.path.dirname(model_path) or save_path
    vecnorm_path = os.path.join(vecnorm_dir, VECNORM_FILE) if args.normalize else None

    print("\nValutazione:")
    mean_s, std_s = test_model(model_path, "CustomHopper-source-v0",
                               args.n_episodes, args.seed + 500, vecnorm_path,
                               args.algorithm)
    mean_t, std_t = test_model(model_path, "CustomHopper-target-v0",
                               args.n_episodes, args.seed + 500, vecnorm_path,
                               args.algorithm)

    kind = "source" if args.env.endswith("source-v0") else "target"
    results = {
        "training_method": f"Baseline no randomization ({kind})",
        "training_env": args.env,
        "algorithm": args.algorithm,
        "timesteps": args.timesteps,
        "n_test_episodes": args.n_episodes,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"seed": args.seed, "n_envs": args.n_envs,
                   "normalize": args.normalize,
                   # None per PPO, altrimenti summarize_results.py mette nello
                   # stesso gruppo run ricorrenti e feedforward
                   "lstm_hidden_size": 128 if args.algorithm == "RecurrentPPO" else None},
        "results": {
            "source_to_source": {"mean": mean_s, "std": std_s},
            "source_to_target": {"mean": mean_t, "std": std_t},
        },
    }
    os.makedirs("./results", exist_ok=True)
    out = f"./results/{kind}_{args.algorithm}{suffix}_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nRisultati salvati in {out}")


if __name__ == "__main__":
    main()