"""Controllo negativo per il probe: raccoglie hidden state da una LSTM NON
addestrata, nello stesso formato di collect_data.py.

Motivazione: una rete ricorrente con pesi casuali, guidata da osservazioni
reali, produce comunque uno stato interno che riflette la dinamica (e' il
principio del reservoir computing). Un probe puo' decodificare parecchio da
li' senza che la rete abbia imparato nulla. Se il probe recupera molto anche
su questa baseline, il risultato principale non dimostra che la codifica sia
appresa.

Due modalita':
  random_lstm    la policy ADDESTRATA sceglie le azioni, una LSTM casuale
                 osserva le stesse osservazioni e produce lo stato che
                 finisce nel dataset. Isola "codifica appresa" da
                 "traiettorie informative": e' il controllo forte.
  random_policy  tutto casuale, azioni comprese. Piu' semplice ma le
                 traiettorie sono diverse (episodi molto piu' corti), quindi
                 il confronto e' meno pulito.
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.custom_hopper import *  # noqa: F401,F403

ENV_ID = "CustomHopper-source-udr-v0"


def build_random_recurrent(venv, hidden_size, seed):
    return RecurrentPPO(
        "MlpLstmPolicy", venv, verbose=0, seed=seed, device="cpu",
        policy_kwargs=dict(lstm_hidden_size=hidden_size, n_lstm_layers=1,
                           shared_lstm=False, enable_critic_lstm=True),
    )


def collect(mode, checkpoint, vecnormalize, n_episodes, stride, use_cell_state,
            deterministic, seed, env_id, hidden_size, encoder_seed):
    venv = DummyVecEnv([lambda: gym.make(env_id)])
    if vecnormalize is not None:
        venv = VecNormalize.load(vecnormalize, venv)
        venv.training = False
        venv.norm_reward = False
        print(f"VecNormalize caricato da: {vecnormalize}")

    actor = None
    if mode == "random_lstm":
        if checkpoint is None:
            raise ValueError("--mode random_lstm richiede --checkpoint")
        actor = RecurrentPPO.load(checkpoint, device="cpu")
        hidden_size = actor.policy.lstm_actor.hidden_size
        print(f"Policy che agisce: {checkpoint} (hidden {hidden_size})")

    # la rete casuale: e' sempre lei a fornire le feature del probe.
    # encoder_seed e' separato da seed apposta: cosi' si possono generare piu'
    # inizializzazioni casuali sulle STESSE traiettorie, e si vede quanto vale
    # la banda del reservoir invece di fidarsi di una sola estrazione
    encoder = build_random_recurrent(venv, hidden_size, encoder_seed)
    print(f"Encoder LSTM con pesi CASUALI (encoder_seed={encoder_seed}), "
          f"hidden {hidden_size}, "
          f"feature del probe: {hidden_size * (2 if use_cell_state else 1)}")
    if mode == "random_policy":
        actor = encoder
        print("Le azioni le sceglie la stessa rete casuale.")

    base_env = venv.envs[0].unwrapped
    hidden_states, targets, t_idx, ep_idx, ep_len = [], [], [], [], []

    for ep in range(n_episodes):
        venv.seed(seed + ep)
        obs = venv.reset()
        true_masses = base_env.get_parameters()[1:4].copy()

        act_states, enc_states = None, None
        episode_start = np.ones((1,), dtype=bool)
        t, done = 0, False
        rows_this_ep = []

        while not done:
            action, act_states = actor.predict(
                obs, state=act_states, episode_start=episode_start,
                deterministic=deterministic)

            if mode == "random_policy":
                enc_states = act_states
            else:
                # stessa osservazione, encoder casuale: l'azione si scarta
                _, enc_states = encoder.predict(
                    obs, state=enc_states, episode_start=episode_start,
                    deterministic=True)

            if t % stride == 0:
                h = enc_states[0].reshape(-1)
                feat = (np.concatenate([h, enc_states[1].reshape(-1)])
                        if use_cell_state else h)
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
        "meta_checkpoint": np.array(f"CONTROL[{mode}] actor={checkpoint}"),
        "meta_vecnormalize": np.array(str(vecnormalize)),
        "meta_env_id": np.array(env_id),
        "meta_lstm_hidden_size": np.array(hidden_size),
        "meta_use_cell_state": np.array(use_cell_state),
        "meta_deterministic": np.array(deterministic),
        "meta_seed": np.array(seed),
        "meta_encoder_seed": np.array(encoder_seed),
        "meta_stride": np.array(stride),
        "meta_control_mode": np.array(mode),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["random_lstm", "random_policy"],
                   default="random_lstm")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="policy addestrata che sceglie le azioni (random_lstm)")
    p.add_argument("--vecnormalize", type=str, default=None)
    p.add_argument("--env_id", type=str, default=ENV_ID)
    p.add_argument("--episodes", type=int, default=450)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--use_cell_state", action="store_true")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--seed", type=int, default=0,
                   help="seed degli episodi: tenerlo uguale a quello del "
                        "dataset vero, cosi' le traiettorie coincidono")
    p.add_argument("--encoder_seed", type=int, default=None,
                   help="seed dei pesi casuali dell'encoder. Default: seed+999. "
                        "Variare SOLO questo per stimare la banda del reservoir.")
    p.add_argument("--lstm_hidden_size", type=int, default=128,
                   help="usato solo in random_policy; in random_lstm si prende "
                        "dal checkpoint per avere lo stesso numero di feature")
    p.add_argument("--out", type=str, default="probe_dataset_control.npz")
    args = p.parse_args()

    if args.checkpoint is not None and args.vecnormalize is None:
        guess = os.path.join(os.path.dirname(args.checkpoint), "vecnormalize.pkl")
        if os.path.exists(guess):
            args.vecnormalize = guess
            print(f"vecnormalize.pkl trovato accanto al checkpoint: {guess}")

    encoder_seed = (args.encoder_seed if args.encoder_seed is not None
                    else args.seed + 999)

    data = collect(args.mode, args.checkpoint, args.vecnormalize, args.episodes,
                   args.stride, args.use_cell_state, args.deterministic,
                   args.seed, args.env_id, args.lstm_hidden_size, encoder_seed)
    np.savez(args.out, **data)
    print(f"\nDataset di controllo salvato in {args.out}")
    print(f"  X: {data['X'].shape}   y: {data['y'].shape}   "
          f"episodi: {len(np.unique(data['episode']))}   "
          f"lunghezza media: {data['episode_length'].mean():.0f} step")


if __name__ == "__main__":
    main()