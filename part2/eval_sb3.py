import argparse
import json
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import panda_gym  # noqa: F401 - required so Panda envs are registered


def evaluate(model_path: str, algo: str, n_episodes: int, deterministic: bool, render: bool, env_type: str, json_out: str = None) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    render_mode = "human" if render else "rgb_array"
    base_env = gym.make("PandaPush-v3", render_mode=render_mode, type=env_type, reward_type="dense")
    env = DummyVecEnv([lambda: base_env])

    # Reload the training-time obs normalization
    vec_norm_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    algo_class = PPO if algo == "ppo" else SAC
    model = algo_class.load(model_path, env=env)

    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            episode_return += float(reward[0])

        episode_returns.append(episode_return)

        info = infos[0]
        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    success_rate = float(np.mean(successes)) if successes else None
    if success_rate is not None:
        print(f"Success rate: {success_rate:.2%}")

    if json_out:
        out = {
            "model_path": model_path,
            "algo": algo,
            "env_type": env_type,
            "n_episodes": n_episodes,
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "success_rate": success_rate,
        }
        out_dir = os.path.dirname(json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"JSON saved: {json_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC or PPO on PandaPush-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a trained model zip file",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="Algorithm used for the saved model",
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=500, 
        help="Number of eval episodes"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with a window (render_mode='human')",
    )
    parser.add_argument(
        "--env-type",
        type=str, default="target",
        choices=["source", "target"],
        help="Type of environment to evaluate on (default: target)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="If set, dump the evaluation summary as JSON to this path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        algo=args.algo,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
        json_out=args.json_out,
    )
