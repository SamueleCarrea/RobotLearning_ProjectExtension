"""Rivaluta i checkpoint `best_model.zip` di tutte le cartelle in models/.

Perche': EvalCallback salva il checkpoint migliore secondo il reward
sull'ambiente di TRAINING (mai sul target), quindi e' un criterio di selezione
legittimo anche in ottica sim-to-real, dove il dominio reale non e' accessibile.
Il checkpoint `_final` invece e' semplicemente quello dove e' finito il budget:
con una curva che oscilla, come quella di RecurrentPPO, e' rumore.

Avere entrambe le tabelle serve a mostrare che le conclusioni non dipendono
dalla scelta del checkpoint.

NOTA: vecnormalize.pkl viene salvato a fine training, quindi corrisponde a
`_final`. Usandolo con best_model c'e' un piccolo disallineamento nelle
statistiche di normalizzazione: e' un limite noto, va dichiarato.
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.custom_hopper import *  # noqa: F401,F403

VECNORM_FILE = "vecnormalize.pkl"


def infer_meta(folder):
    """Ricava metodo, env di training e seed dal nome della cartella."""
    name = os.path.basename(folder.rstrip("/"))
    seed = None
    m = re.search(r"_s(\d+)$", name)
    if m:
        seed = int(m.group(1))

    if name.startswith("oracle"):
        kind = "all" if "_all_" in name else "links"
        return f"Oracle ({kind} masses) [best ckpt]", "CustomHopper-source-udr-v0", seed
    if name.startswith("udr"):
        env = "CustomHopper-source-udr-v0"
        for pct in ("5", "25", "50"):
            if f"udr{pct}_" in name:
                env = f"CustomHopper-source-udr{pct}-v0"
        return "Uniform Domain Randomization (UDR) [best ckpt]", env, seed
    if name.startswith("baseline"):
        if "target" in name:
            return ("Baseline no randomization (target) [best ckpt]",
                    "CustomHopper-target-v0", seed)
        return ("Baseline no randomization (source) [best ckpt]",
                "CustomHopper-source-v0", seed)
    return None, None, seed


def infer_algorithm(folder):
    """best_model.zip non dice da quale algoritmo viene: NON si puo' scoprire
    provando a caricare, perche' un caricamento con la classe sbagliata puo'
    non sollevare eccezione e produrre pesi mescolati (e' il bug che ha dato
    i reward ~60 sulle varianti udr5/udr25). Si deduce dal checkpoint finale
    che sta nella stessa cartella, il cui nome incorpora l'algoritmo.
    """
    finals = glob.glob(os.path.join(folder, "*_final.zip"))
    if not finals:
        raise RuntimeError(f"nessun *_final.zip in {folder}, non posso "
                           "dedurre l'algoritmo in modo affidabile")
    name = os.path.basename(finals[0])
    if "RecurrentPPO" in name:
        return RecurrentPPO, "RecurrentPPO"
    if "PPO" in name:
        return PPO, "PPO"
    raise RuntimeError(f"nome non riconosciuto: {name}")


def load_any(path, folder):
    cls, algo = infer_algorithm(folder)
    return cls.load(path, device="cpu"), algo


def evaluate(model, env_id, n_episodes, vecnorm_path, seed):
    venv = DummyVecEnv([lambda: Monitor(gym.make(env_id))])
    if vecnorm_path is not None and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    venv.seed(seed)
    mean_r, std_r = evaluate_policy(model, venv, n_eval_episodes=n_episodes,
                                    deterministic=True)
    venv.close()
    return float(mean_r), float(std_r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="./models")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--eval_seed", type=int, default=500)
    p.add_argument("--skip", nargs="*", default=["smoke"],
                   help="sottostringhe: le cartelle che le contengono si saltano")
    args = p.parse_args()

    folders = sorted(glob.glob(os.path.join(args.models_dir, "*/")))
    os.makedirs(args.results_dir, exist_ok=True)
    done, skipped = 0, []

    for folder in folders:
        name = os.path.basename(folder.rstrip("/"))
        if any(s in name for s in args.skip):
            skipped.append((name, "escluso da --skip"))
            continue
        best = os.path.join(folder, "best_model.zip")
        if not os.path.exists(best):
            skipped.append((name, "nessun best_model.zip"))
            continue

        method, train_env, seed = infer_meta(folder)
        if method is None:
            skipped.append((name, "nome non riconosciuto"))
            continue

        try:
            model, algo = load_any(best, folder)
        except Exception as exc:
            skipped.append((name, f"load fallito: {exc}"))
            continue

        vecnorm = os.path.join(folder, VECNORM_FILE)
        try:
            mean_s, std_s = evaluate(model, "CustomHopper-source-v0",
                                     args.n_episodes, vecnorm, args.eval_seed)
            mean_t, std_t = evaluate(model, "CustomHopper-target-v0",
                                     args.n_episodes, vecnorm, args.eval_seed)
        except Exception as exc:
            # tipico dell'oracolo: si aspetta osservazioni estese con le masse
            skipped.append((name, f"eval fallita: {type(exc).__name__}"))
            continue

        is_rec = algo == "RecurrentPPO"
        out = {
            "training_method": method,
            "training_env": train_env,
            "algorithm": algo,
            "checkpoint": "best_model",
            "timesteps": 1000000,
            "n_test_episodes": args.n_episodes,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {"seed": seed, "n_envs": 8, "normalize": True,
                       "lstm_hidden_size": 128 if is_rec else None},
            "results": {
                "source_to_source": {"mean": mean_s, "std": std_s},
                "source_to_target": {"mean": mean_t, "std": std_t},
            },
        }
        path = os.path.join(args.results_dir, f"best_{name}_results.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=4)
        print(f"{name:28s} {algo:13s} src={mean_s:7.1f}  tgt={mean_t:7.1f}")
        done += 1

    print(f"\nValutati {done} best_model.")
    if skipped:
        print("Saltati:")
        for n, why in skipped:
            print(f"  {n:28s} {why}")
    print("\nOra: python summarize_results.py --markdown results/summary.md")


if __name__ == "__main__":
    main()