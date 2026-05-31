import argparse
from collections import deque
import os
import time

import torch
import wandb
import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on PandaPush-v3")
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env_type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="msi",
        help="device used (default: msi)", #msi is francesco's laptop, for fast training. remember to change
    )

    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=1024, help="Rollout buffer size per env")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size for PPO updates")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per PPO update")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.97, help="GAE lambda for advantage estimation")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient for exploration")
    parser.add_argument("--target_kl", type=float, default=0.05, help="Limit the KL divergence between updates")
    parser.add_argument("--use_sde", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=False, help="Use generalized State-Dependent Exploration")
    parser.add_argument("--sde_sample_freq", type=int, default=1, help="Sample frequency for gSDE")
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Skip saving the final model and VecNormalize stats",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging (all wandb calls become no-ops)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=100,
        help="Maximum number of steps per episode (default: 100, panda-gym default is 50)",
    )
    parser.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="Disable periodic checkpointing during training",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "msi":
        device = "cpu"
        num_envs = 16  # Efficient parallelization for 20-core CPU
        
        # Optimize CPU threads for parallel environments to avoid contention
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
    else:
        device = args.device
        num_envs = 4  # Default fallback

    def make_env(rank: int, seed: int):
        def _init():
            e = gym.make(
                "PandaPush-v3",
                render_mode=None,
                type=args.env_type,
                reward_type="dense",
                max_episode_steps=args.max_episode_steps,
            )
            e.reset(seed=seed + rank)
            return Monitor(e)
        return _init

    env = SubprocVecEnv([make_env(i, args.seed) for i in range(num_envs)])
    env = VecNormalize(env, training=True)

    #TODO: add randomization wrapper here
    #TODO: create model and train it

    config = {
        "algo": "PPO",
        "policy": "MultiInputPolicy",
        "env_type": args.env_type,
        "sampling_strategy": args.sampling_strategy,
        "total_timesteps": args.timesteps,
        "seed": args.seed,
        "device": device,
        "num_envs": num_envs,
        # PPO hyperparameters (CLI defaults, overridden by sweep agent)
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "clip_range": args.clip_range,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
        "target_kl": args.target_kl,
        "use_sde": args.use_sde,
        "sde_sample_freq": args.sde_sample_freq,
        # environment hyperparameters
        "max_episode_steps": args.max_episode_steps,
    }
    run = wandb.init(
        project="faiml-group64-ppo",
        config=config,
        sync_tensorboard=True,
        save_code=True,
        mode="disabled" if args.no_wandb else "online",
    )

    # Read from wandb.config so sweep agent can override CLI defaults
    cfg = wandb.config

    save_name = f"ppo_push_{cfg.sampling_strategy}_{cfg.env_type}_seed{cfg.seed}_{cfg.total_timesteps // 1000}k_{run.id}"
    run.name = save_name

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        device=device,
        verbose=1,
        seed=args.seed,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        clip_range=cfg.clip_range,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        target_kl=cfg.target_kl,
        use_sde=cfg.use_sde,
        sde_sample_freq=cfg.sde_sample_freq,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    )

    t_start = time.time()

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=WandbCallback(
            verbose=2,
        ),
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=save_name,
    )

    elapsed = time.time() - t_start
    steps_per_sec = cfg.total_timesteps / elapsed
    minutes = elapsed / 60

    print(f"\n{'='*50}")
    print(f"Training complete.")
    print(f"Total timesteps:  {cfg.total_timesteps:,}")
    print(f"Elapsed time:     {minutes:.1f} min ({elapsed:.0f} s)")
    print(f"Throughput:       {steps_per_sec:.0f} steps/s")
    print(f"{'='*50}\n")

    wandb.log({
        "timing/elapsed_seconds": elapsed,
        "timing/elapsed_minutes": minutes,
        "timing/steps_per_second": steps_per_sec,
    })

    run.finish()

    if not args.no_save:
        os.makedirs("models", exist_ok=True)
        save_path = os.path.join("models", save_name)
        model.save(save_path)
        env.save(save_path + "_vecnormalize.pkl")
        print(f"Model saved to {save_path}.zip")
    else:
        print("Skipping model save (--no_save).")


if __name__ == "__main__":
    main()