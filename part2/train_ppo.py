import argparse
import os
import re
import time

import torch
import gymnasium as gym
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on PandaPush-v3")
    parser.add_argument("--env-type", type=str, default="source", choices=["source", "target"],
                        help="PandaPush environment type")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Number of training timesteps")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments (SubprocVecEnv)")
    parser.add_argument("--model-dir", type=str, default=".", help="Directory to save the final model")
    parser.add_argument("--ckpt-dir", type=str, default=".", help="Directory to save periodic checkpoints")
    parser.add_argument("--tb-dir", type=str, default=None, help="TensorBoard log directory (disabled if unset)")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint .zip to resume from")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--run-id", type=str, default=None, help="Unique run identifier (checkpoint/model naming)")
    parser.add_argument("--ckpt-freq", type=int, default=25_000,
                        help="Save a checkpoint every N steps (0 to disable)")
    parser.add_argument("--device", type=str, default="auto", help="Torch device (auto/cpu/cuda)")
    parser.add_argument("--eval-target", action="store_true", help="Periodically evaluate on the deployment env")
    parser.add_argument("--eval-env-type", type=str, default="target", choices=["source", "target"])
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="faiml-group64-ppo",
                        help="W&B project name")
    parser.add_argument("--max-episode-steps", type=int, default=100,
                        help="Maximum number of steps per episode (default: 100, panda-gym default is 50)")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout buffer size per env")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size for PPO updates")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per PPO update")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.97, help="GAE lambda for advantage estimation")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient for exploration")
    parser.add_argument("--target-kl", type=float, default=0.05, help="Limit the KL divergence between updates")
    parser.add_argument("--use-sde", action="store_true", help="Use generalized State-Dependent Exploration")
    parser.add_argument("--sde-sample-freq", type=int, default=1, help="Sample frequency for gSDE")
    parser.add_argument("--net-arch", type=str, default="256,256",
                        help='Comma-separated hidden layer sizes, e.g. "256,256"')

    return parser.parse_args()


def make_env(env_type: str, seed: int, rank: int, max_episode_steps: int):
    """Factory that returns a thunk for SubprocVecEnv."""
    def _init():
        e = gym.make(
            "PandaPush-v3",
            render_mode=None,
            type=env_type,
            reward_type="dense",
            max_episode_steps=max_episode_steps,
        )
        e.reset(seed=seed + rank)
        return Monitor(e)
    return _init


def main() -> None:
    args = parse_args()

    # Optimise CPU threads for parallel envs (avoids contention with SubprocVecEnv)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    num_envs = args.n_envs

    # ── Run name ──────────────────────────────────────────────────────────
    if args.run_id:
        run_name = args.run_id
    else:
        run_name = f"ppo_push_{args.env_type}_seed{args.seed}"

    # ── Optional W&B ──────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        import wandb
        if args.tb_dir is None:
            args.tb_dir = os.path.join("runs", "tb")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=f"ppo_{args.env_type}",
            tags=["ppo", args.env_type, f"seed{args.seed}"],
            config={
                "algo": "PPO",
                "env_type": args.env_type,
                "seed": args.seed,
                "timesteps": args.timesteps,
                "num_envs": num_envs,
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
                "max_episode_steps": args.max_episode_steps,
                "net_arch": args.net_arch,
            },
            sync_tensorboard=True,
        )

    # ── Environment ───────────────────────────────────────────────────────
    env = SubprocVecEnv([
        make_env(args.env_type, args.seed, i, args.max_episode_steps)
        for i in range(num_envs)
    ])

    # ── Checkpoint dirs ───────────────────────────────────────────────────
    ckpt_path = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ── Parse network architecture ────────────────────────────────────────
    net_arch_list = [int(x) for x in args.net_arch.split(",")]
    policy_kwargs = dict(net_arch=dict(pi=net_arch_list, vf=net_arch_list))

    # ── Resume or fresh start ─────────────────────────────────────────────
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[train_ppo] Resuming from {args.resume_from}")
        basename = os.path.basename(args.resume_from)
        m = re.match(r"(.+)_(\d+)_steps\.zip$", basename)
        if m:
            prefix, step = m.group(1), m.group(2)
            ckpt_dir_resume = os.path.dirname(args.resume_from)
            vec_norm_path = os.path.join(ckpt_dir_resume, f"{prefix}_vecnormalize_{step}_steps.pkl")
        else:
            vec_norm_path = args.resume_from.replace(".zip", "_vecnormalize.pkl")

        if not os.path.exists(vec_norm_path):
            raise FileNotFoundError(
                f"VecNormalize stats missing next to checkpoint: {vec_norm_path}. "
                "Without them obs end up in the wrong scale."
            )
        env = VecNormalize.load(vec_norm_path, env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(args.resume_from, env=env, tensorboard_log=args.tb_dir)

        remaining = max(0, args.timesteps - model.num_timesteps)
        reset_num = False
    else:
        print("[train_ppo] Starting fresh")
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            device=args.device,
            verbose=1,
            seed=args.seed,
            tensorboard_log=args.tb_dir,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            use_sde=args.use_sde,
            sde_sample_freq=args.sde_sample_freq,
            policy_kwargs=policy_kwargs,
        )
        remaining = args.timesteps
        reset_num = True

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = []
    if args.ckpt_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.ckpt_freq // num_envs, 1),
                save_path=ckpt_path,
                name_prefix="ckpt",
                save_replay_buffer=False,
                save_vecnormalize=True,
                verbose=1,
            )
        )

    if args.eval_target:
        eval_env = SubprocVecEnv([
            make_env(args.eval_env_type, args.seed + 10_000, 0, args.max_episode_steps)
        ])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=ckpt_path,
                eval_freq=max(args.eval_freq // num_envs, 1),
                n_eval_episodes=50,
                deterministic=True,
                render=False,
                verbose=1,
            )
        )

    if args.wandb:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(verbose=1))

    # ── Train ─────────────────────────────────────────────────────────────
    t_start = time.time()

    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_num,
        tb_log_name=run_name,
    )

    elapsed = time.time() - t_start
    steps_per_sec = remaining / elapsed if elapsed > 0 else 0
    minutes = elapsed / 60

    print(f"\n{'='*50}")
    print(f"Training complete.")
    print(f"Total timesteps:  {remaining:,}")
    print(f"Elapsed time:     {minutes:.1f} min ({elapsed:.0f} s)")
    print(f"Throughput:       {steps_per_sec:.0f} steps/s")
    print(f"{'='*50}\n")

    # ── Save ──────────────────────────────────────────────────────────────
    save_name_path = os.path.join(args.model_dir, run_name)
    model.save(save_name_path)
    env.save(save_name_path + "_vecnormalize.pkl")
    print(f"[train_ppo] Final model saved to {save_name_path}")

    if wandb_run is not None:
        import wandb
        wandb.log({
            "timing/elapsed_seconds": elapsed,
            "timing/elapsed_minutes": minutes,
            "timing/steps_per_second": steps_per_sec,
        })
        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(save_name_path + ".zip")
        artifact.add_file(save_name_path + "_vecnormalize.pkl")
        wandb_run.log_artifact(artifact)
        wandb_run.finish()


if __name__ == "__main__":
    main()