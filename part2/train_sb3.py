import argparse
import os
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # wrap env to normalize obs and reward
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC or PPO on PandaPush-v3")
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--sampling-strategy", "--strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=".",
        help="Directory to save the final model",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=".",
        help="Directory to save periodic checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint .zip to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--mass-range",
        nargs=2,
        type=float,
        default=[0.5, 6.0],
        help="Min and Max mass for UDR, or global limits for ADR",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique run identifier (used for checkpoint/model naming)",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()

    def make_env():
        e = gym.make(
            "PandaPush-v3",
            render_mode="rgb_array",
            type=args.env_type,
            reward_type="dense",
        )
        return RandomizationWrapper(
            e,
            mass_range=tuple(args.mass_range),
            mode=args.sampling_strategy,
            seed=args.seed,
        )

    env = DummyVecEnv([make_env])

    # Use the provided run_id if any, otherwise build one
    if args.run_id:
        run_name = args.run_id
    else:
        run_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_seed{args.seed}"
    ckpt_path = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    algo_class = PPO if args.algo == "ppo" else SAC
    batch_size = 256 if args.algo == "ppo" else 1024

    # Resume from checkpoint if provided, otherwise start fresh
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[train] Resuming from {args.resume_from}")
        model = algo_class.load(args.resume_from, env=env)
    else:
        print("[train] Starting fresh")
        # Normalize obs and reward: without this SAC/PPO converge slower
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        model = algo_class("MultiInputPolicy", env, verbose=1, seed=args.seed, batch_size=batch_size)

    # Save a checkpoint every 25k steps so that resume can pick up vec stats and (for SAC) the replay buffer
    ckpt_cb = CheckpointCallback(
        save_freq=max(25_000 // env.num_envs, 1),
        save_path=ckpt_path,
        name_prefix="ckpt",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, callback=ckpt_cb, progress_bar=True)

    save_name = os.path.join(args.model_dir, run_name)
    model.save(save_name)
    env.save(save_name + "_vecnormalize.pkl")
    print(f"[train] Final model saved to {save_name}")


if __name__ == "__main__":
    main()
