"""Policy oracolo: PPO feedforward che riceve anche le masse vere in input,
quindi non deve inferirle. Serve come upper bound rispetto alla RecurrentPPO.

Nota: source e target differiscono solo per la massa del torso, che non viene
randomizzata. Con --oracle_masses links (default) l'oracolo riceve solo
thigh/leg/foot, cioe' nessuna informazione sul gap; con all riceve anche il
torso.
"""

import argparse
import json
import os
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env.custom_hopper import *
from oracle_env import OracleWrapper

VECNORM_FILE = "vecnormalize.pkl"


def _make_env(env_id, seed, masses):
    def _init():
        env = Monitor(OracleWrapper(gym.make(env_id), masses=masses))
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _init


def _build(env_id, n_envs, seed, training, normalize, masses):
    cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = cls([_make_env(env_id, seed + i, masses) for i in range(n_envs)])
    if not normalize:
        return venv
    return VecNormalize(venv, training=training, norm_obs=True,
                        norm_reward=training, clip_obs=10.0)


def train_oracle(train_env_id, timesteps, save_path, tag, seed, n_envs, normalize,
                 masses):
    os.makedirs(save_path, exist_ok=True)
    train_env = _build(train_env_id, n_envs, seed, True, normalize, masses)
    eval_env = _build(train_env_id, min(n_envs, 4), seed + 1000, False, normalize, masses)

    print("=" * 50)
    print("Oracle PPO (masse vere in osservazione)")
    print(f"env={train_env_id}  masses={masses}  obs={train_env.observation_space.shape}")
    print(f"n_envs={n_envs}  normalize={normalize}  seed={seed}")
    print("=" * 50)

    n_steps = max(2048 // n_envs, 64)
    batch_size = 64
    while (n_steps * n_envs) % batch_size != 0:
        batch_size -= 1

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu",
        seed=seed,
        tensorboard_log=f"./tensorboard/oracle{'_' + tag if tag else ''}/",
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[
            EvalCallback(eval_env, best_model_save_path=save_path, log_path=save_path,
                         eval_freq=max(10000 // n_envs, 1), deterministic=True,
                         n_eval_episodes=10),
            CheckpointCallback(save_freq=max(50000 // n_envs, 1), save_path=save_path,
                               name_prefix="oracle_PPO"),
        ],
        progress_bar=True,
    )

    model_path = os.path.join(save_path, "oracle_PPO_final")
    model.save(model_path)
    if normalize:
        train_env.save(os.path.join(save_path, VECNORM_FILE))
    print(f"\nModello salvato in {model_path}")
    train_env.close()
    eval_env.close()
    return model_path


def test_oracle(model_path, env_id, n_episodes, seed, vecnorm_path, masses):
    model = PPO.load(model_path, device="cpu")
    venv = DummyVecEnv([_make_env(env_id, seed, masses)])
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
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--train_env", type=str, default="CustomHopper-source-udr-v0")
    parser.add_argument("--oracle_masses", choices=["links", "all"], default="links")
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    save_path = f"./models/oracle{suffix}/"

    if args.mode in ("train", "both"):
        model_path = train_oracle(args.train_env, args.timesteps, save_path, args.tag,
                                  args.seed, args.n_envs, args.normalize,
                                  args.oracle_masses)
    else:
        if args.model_path is None:
            raise ValueError("In modalita' test serve --model_path")
        model_path = args.model_path

    if args.mode not in ("test", "both"):
        return

    vecnorm_dir = os.path.dirname(model_path) or save_path
    vecnorm_path = os.path.join(vecnorm_dir, VECNORM_FILE) if args.normalize else None

    print("\nValutazione oracolo:")
    mean_s, std_s = test_oracle(model_path, "CustomHopper-source-v0", args.n_episodes,
                                args.seed + 500, vecnorm_path, args.oracle_masses)
    mean_t, std_t = test_oracle(model_path, "CustomHopper-target-v0", args.n_episodes,
                                args.seed + 500, vecnorm_path, args.oracle_masses)

    results = {
        "training_method": f"Oracle ({args.oracle_masses} masses)",
        "training_env": args.train_env,
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "n_test_episodes": args.n_episodes,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"seed": args.seed, "n_envs": args.n_envs,
                   "normalize": args.normalize, "oracle_masses": args.oracle_masses},
        "results": {
            "source_to_source": {"mean": mean_s, "std": std_s},
            "source_to_target": {"mean": mean_t, "std": std_t},
        },
    }
    os.makedirs("./results", exist_ok=True)
    out = f"./results/oracle_PPO{suffix}_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nRisultati salvati in {out}")


if __name__ == "__main__":
    main()
