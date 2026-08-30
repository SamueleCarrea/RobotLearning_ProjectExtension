"""Valuta un checkpoint su ambienti in cui variano thigh/leg/foot, cioe' proprio
i parametri che la UDR randomizza, tenendo il torso fisso al valore source.

Serve a testare l'ipotesi strutturale del progetto: la policy ricorrente non
aiuta sul transfer source->target perche' li' cambia il torso, che non e' mai
stato randomizzato e quindi non e' inferibile. Se l'ipotesi e' giusta, su
ambienti in cui cambia cio' che E' stato randomizzato il ricorrente dovrebbe
recuperare terreno sul feedforward.
"""

import argparse
import json
import os

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.custom_hopper import *  # noqa: F401,F403

ALGOS = {"RecurrentPPO": RecurrentPPO, "PPO": PPO}
VECNORM_FILE = "vecnormalize.pkl"


def make_scaled_env(env_id, scale, seed, vecnorm_path, which="all"):
    """Env senza randomizzazione con thigh/leg/foot moltiplicati per `scale`.

    Il torso resta quello dell'env di base (source o target): il fattore agisce
    solo sui link che la UDR randomizza.
    """
    def _init():
        return Monitor(gym.make(env_id))

    venv = DummyVecEnv([_init])
    if vecnorm_path is not None:
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    base = venv.envs[0].unwrapped
    # original_masses = [torso, thigh, leg, foot] PRIMA dello shift source
    masses = base.get_parameters().copy()   # torso qui e' gia' quello giusto
    idx = {"all": [1, 2, 3], "thigh": [1], "leg": [2], "foot": [3]}[which]
    for i in idx:
        masses[i] = base.original_masses[i] * scale
    base.set_parameters(masses)

    venv.seed(seed)
    return venv, masses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--algorithm", choices=["RecurrentPPO", "PPO"],
                   default="RecurrentPPO")
    p.add_argument("--vecnormalize", type=str, default=None,
                   help="default: vecnormalize.pkl accanto al checkpoint")
    p.add_argument("--no-vecnormalize", dest="disable_vecnorm",
                   action="store_true")
    p.add_argument("--env_id", type=str, default="CustomHopper-source-v0")
    p.add_argument("--which", choices=["all", "thigh", "leg", "foot"],
                   default="all", help="quali link scalare")
    p.add_argument("--scales", type=float, nargs="+",
                   default=[0.7, 0.85, 1.0, 1.15, 1.3],
                   help="0.85 e 1.15 sono i bordi del range UDR standard (15%%), "
                        "0.7 e 1.3 sono fuori distribuzione")
    p.add_argument("--n_episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=500)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    vecnorm = None
    if not args.disable_vecnorm:
        vecnorm = args.vecnormalize or os.path.join(
            os.path.dirname(args.checkpoint), VECNORM_FILE)
        if not os.path.exists(vecnorm):
            raise FileNotFoundError(
                f"{vecnorm} non esiste. Usa --no-vecnormalize se la policy e' "
                "stata allenata senza, oppure passa --vecnormalize.")

    model = ALGOS[args.algorithm].load(args.checkpoint, device="cpu")
    print(f"checkpoint: {args.checkpoint}  ({args.algorithm})")
    print(f"env base:   {args.env_id}   link scalati: {args.which}")
    print("=" * 62)

    rows = []
    for scale in args.scales:
        venv, masses = make_scaled_env(args.env_id, scale, args.seed,
                                       vecnorm, args.which)
        mean_r, std_r = evaluate_policy(model, venv,
                                        n_eval_episodes=args.n_episodes,
                                        deterministic=True)
        venv.close()
        print(f"  scale={scale:<5.2f}  masse={np.round(masses, 2)}  "
              f"reward={mean_r:7.1f} +/- {std_r:5.1f}")
        rows.append({"scale": float(scale),
                     "masses": [float(m) for m in masses],
                     "mean": float(mean_r), "std": float(std_r)})

    out = args.out or (
        f"./results/crossmass_{args.algorithm}_{args.which}_"
        f"{os.path.basename(os.path.dirname(args.checkpoint))}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "algorithm": args.algorithm,
                   "env_id": args.env_id, "which": args.which,
                   "n_episodes": args.n_episodes, "results": rows}, f, indent=4)
    print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()