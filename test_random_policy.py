"""Policy random sull'Hopper, script di partenza dell'esercizio."""

import argparse

import gymnasium as gym
import numpy as np

from env.custom_hopper import *  # noqa: F401,F403


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CustomHopper-source-v0")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = gym.make(args.env, render_mode="human" if args.render else None)

    print("State space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Masse dei link:", env.unwrapped.get_parameters())

    returns = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done, total, steps = False, 0.0, 0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            total += reward
            steps += 1
        returns.append(total)
        print(f"  episodio {ep + 1}: return={total:.2f}  lunghezza={steps}")

    env.close()
    print(f"\nReturn medio della policy random: {np.mean(returns):.2f} "
          f"+/- {np.std(returns):.2f}")


if __name__ == "__main__":
    main()
