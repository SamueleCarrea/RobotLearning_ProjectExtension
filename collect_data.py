"""Raccoglie coppie (hidden state LSTM, masse vere) facendo rollout della policy
UDR su CustomHopper-source-udr-v0. Output: un .npz per il probe.

Se la policy e' stata allenata con VecNormalize bisogna passare anche le stesse
statistiche, altrimenti gli hidden state raccolti non sono quelli veri.
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.custom_hopper import *  # noqa: F401,F403  registra CustomHopper-*-v0

ENV_ID = "CustomHopper-source-udr-v0"


def collect(checkpoint, vecnormalize, n_episodes, stride, use_cell_state,
            deterministic, seed, env_id):
    venv = DummyVecEnv([lambda: gym.make(env_id)])
    if vecnormalize is not None:
        venv = VecNormalize.load(vecnormalize, venv)
        venv.training = False
        venv.norm_reward = False
        print(f"VecNormalize caricato da: {vecnormalize}")

    if checkpoint is not None:
        model = RecurrentPPO.load(checkpoint, device="cpu")
        print(f"Checkpoint caricato: {checkpoint}")
    else:
        model = RecurrentPPO("MlpLstmPolicy", venv, verbose=0, seed=seed, device="cpu")
        print("ATTENZIONE: nessun checkpoint, pesi random (solo per testare la pipeline).")

    hidden_size = model.policy.lstm_actor.hidden_size
    print(f"lstm_hidden_size della policy: {hidden_size}  "
          f"(feature del probe: {hidden_size * (2 if use_cell_state else 1)})")

    # serve per leggere le masse campionate al reset
    base_env = venv.envs[0].unwrapped

    hidden_states, targets, t_idx, ep_idx, ep_len = [], [], [], [], []

    for ep in range(n_episodes):
        venv.seed(seed + ep)
        obs = venv.reset()
        # thigh, leg, foot (il torso non e' randomizzato)
        true_masses = base_env.get_parameters()[1:4].copy()

        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        t, done = 0, False
        rows_this_ep = []

        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start,
                deterministic=deterministic,
            )

            if t % stride == 0:
                h = lstm_states[0].reshape(-1)
                feat = np.concatenate([h, lstm_states[1].reshape(-1)]) if use_cell_state else h
                hidden_states.append(feat)
                targets.append(true_masses)
                t_idx.append(t)
                ep_idx.append(ep)
                rows_this_ep.append(len(hidden_states) - 1)

            obs, reward, dones, infos = venv.step(action)
            done = bool(dones[0])
            episode_start = dones
            t += 1

        ep_len.extend([t] * len(rows_this_ep))

        if (ep + 1) % 50 == 0:
            print(f"  episodio {ep + 1}/{n_episodes}, lunghezza {t} step")

    venv.close()

    return {
        "X": np.array(hidden_states, dtype=np.float32),
        "y": np.array(targets, dtype=np.float32),
        "t": np.array(t_idx, dtype=np.int32),
        "episode": np.array(ep_idx, dtype=np.int32),
        "episode_length": np.array(ep_len, dtype=np.int32),
        # metadata, per sapere da che policy viene il dataset
        "meta_checkpoint": np.array(str(checkpoint)),
        "meta_vecnormalize": np.array(str(vecnormalize)),
        "meta_env_id": np.array(env_id),
        "meta_lstm_hidden_size": np.array(hidden_size),
        "meta_use_cell_state": np.array(use_cell_state),
        "meta_deterministic": np.array(deterministic),
        "meta_seed": np.array(seed),
        "meta_stride": np.array(stride),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path al .zip di RecurrentPPO. Se omesso, pesi random.")
    p.add_argument("--vecnormalize", type=str, default=None,
                   help="Path a vecnormalize.pkl salvato accanto al checkpoint.")
    p.add_argument("--env_id", type=str, default=ENV_ID)
    p.add_argument("--episodes", type=int, default=450)
    p.add_argument("--stride", type=int, default=5,
                   help="Campiona lo hidden state ogni N step.")
    p.add_argument("--use_cell_state", action="store_true",
                   help="Concatena anche il cell state c oltre a h.")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="probe_dataset.npz")
    args = p.parse_args()

    if args.checkpoint is not None and args.vecnormalize is None:
        guess = os.path.join(os.path.dirname(args.checkpoint), "vecnormalize.pkl")
        if os.path.exists(guess):
            args.vecnormalize = guess
            print(f"vecnormalize.pkl trovato accanto al checkpoint: {guess}")
        else:
            print("\n" + "!" * 70)
            print("ATTENZIONE: nessun vecnormalize.pkl accanto al checkpoint.")
            print("Se la policy e' stata allenata con VecNormalize il dataset")
            print("non e' valido. Passa --vecnormalize a mano.")
            print("!" * 70 + "\n")

    data = collect(args.checkpoint, args.vecnormalize, args.episodes, args.stride,
                   args.use_cell_state, args.deterministic, args.seed, args.env_id)

    np.savez(args.out, **data)
    print(f"\nDataset salvato in {args.out}")
    print(f"  X: {data['X'].shape}   y: {data['y'].shape}   "
          f"episodi: {len(np.unique(data['episode']))}")


if __name__ == "__main__":
    main()
