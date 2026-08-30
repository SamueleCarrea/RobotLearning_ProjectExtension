"""Fa girare un episodio con rendering a schermo (vedi l'hopper saltare
dal vivo) e stampa quando avviene il primo contatto del piede col terreno,
per verificare a occhio l'ipotesi: il foot rivela la sua massa subito
al primo impatto, il thigh richiede piu' salti per essere identificato.

Uso:
    python watch_episode.py --checkpoint models/udr/udr_RecurrentPPO_final.zip
    python watch_episode.py --checkpoint ... --no-render   # solo log, niente finestra
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.custom_hopper import *  # noqa: F401,F403


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="se omesso lo cerca accanto al checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true",
                         help="Disattiva la finestra grafica, utile su server/senza display.")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env = gym.make("CustomHopper-source-udr-v0", render_mode=render_mode)

    # se la policy e' stata allenata con VecNormalize servono le stesse stats
    if args.vecnormalize is None and args.checkpoint is not None:
        guess = os.path.join(os.path.dirname(args.checkpoint), "vecnormalize.pkl")
        args.vecnormalize = guess if os.path.exists(guess) else None
    normalizer = None
    if args.vecnormalize is not None:
        normalizer = VecNormalize.load(args.vecnormalize,
                                       DummyVecEnv([lambda: gym.make("CustomHopper-source-udr-v0")]))
        normalizer.training = False
        print(f"VecNormalize caricato da: {args.vecnormalize}")

    if args.checkpoint is not None:
        model = RecurrentPPO.load(args.checkpoint, device="cpu")
    else:
        print("ATTENZIONE: nessun checkpoint, pesi random (solo per testare lo script).")
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, seed=args.seed, device="cpu")

    def prep(o):
        return normalizer.normalize_obs(o) if normalizer is not None else o

    obs, _ = env.reset(seed=args.seed)
    true_masses = env.unwrapped.get_parameters()[1:4]
    print(f"Masse vere in questo episodio: thigh={true_masses[0]:.3f}  "
          f"leg={true_masses[1]:.3f}  foot={true_masses[2]:.3f}")

    lstm_states = None
    episode_start = True
    t = 0
    done = False
    first_contact_t = None
    n_contacts_so_far = 0
    hops = 0
    was_in_contact = False

    while not done:
        action, lstm_states = model.predict(
            prep(obs), state=lstm_states, episode_start=[episode_start], deterministic=True
        )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_start = False

        # ncon = numero di contatti attivi in questo istante di simulazione
        in_contact = env.unwrapped.data.ncon > 0
        if in_contact:
            n_contacts_so_far += 1
            if first_contact_t is None:
                first_contact_t = t
        if was_in_contact and not in_contact:
            hops += 1  # transizione contatto -> aria = un salto completato
        was_in_contact = in_contact

        if t % 20 == 0:
            print(f"  t={t:4d}  z_torso={obs[0]:+.3f}  in_contatto={in_contact}  "
                  f"salti_completati={hops}")

        t += 1

    env.close()
    print(f"\nEpisodio finito dopo {t} step.")
    print(f"Primo contatto col suolo: step {first_contact_t}")
    print(f"Numero totale di step in contatto: {n_contacts_so_far}")
    print(f"Numero di salti completati: {hops}")
    print("\nSe first_contact_t e' molto piccolo (es. <20), conferma che il "
          "foot rivela la sua massa quasi subito. Se il numero di salti "
          "cresce lentamente lungo l'episodio, conferma che il thigh ha "
          "bisogno di piu' cicli di salto per essere identificato.")


if __name__ == "__main__":
    main()