import argparse
import os
import re

import gymnasium as gym
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # wrap env to normalize obs and reward
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC or PPO on PandaPush-v3")
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"])
    parser.add_argument(
        "--sampling-strategy", "--strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument("--env-type", type=str, default="source", choices=["source", "target"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--model-dir", type=str, default=".", help="Directory to save the final model")
    parser.add_argument("--ckpt-dir", type=str, default=".", help="Directory to save periodic checkpoints")
    parser.add_argument("--tb-dir", type=str, default=None, help="TensorBoard log directory (disabled if unset)")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint .zip to resume from")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--mass-range",
        nargs=2,
        type=float,
        default=[0.5, 8.0],
        help="Min and Max mass for UDR, or global limits for ADR",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Unique run identifier (checkpoint/model naming)")
    parser.add_argument("--ckpt-freq", type=int, default=25_000, help="Save a checkpoint every N steps (0 to disable)")
    parser.add_argument("--device", type=str, default="auto", help="Torch device (auto/cpu/cuda)")
    parser.add_argument("--eval-target", action="store_true", help="Periodically evaluate on the deployment env")
    parser.add_argument("--eval-env-type", type=str, default="target", choices=["source", "target"])
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="faiml-group64-pandapush")
    # Optional hyperparameter overrides; left as None they keep the SB3 defaults (baseline runs unchanged)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--ent-coef", type=str, default=None, help='"auto" or a float')
    parser.add_argument("--net-arch", type=str, default=None, help='e.g. "256,256"')
    parser.add_argument("--tau", type=float, default=None, help="SAC only")
    parser.add_argument("--train-freq", type=int, default=None, help="SAC only")
    parser.add_argument("--gradient-steps", type=int, default=None, help="SAC only")
    args, _ = parser.parse_known_args()
    return args


def make_env(env_type, mass_range, strategy, seed):
    e = gym.make("PandaPush-v3", render_mode="rgb_array", type=env_type, reward_type="dense")
    return RandomizationWrapper(e, mass_range=tuple(mass_range), mode=strategy, seed=seed)


def main() -> None:
    args = parse_args()

    if args.run_id:
        run_name = args.run_id
    else:
        run_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_seed{args.seed}"

    # Optional W&B run; sync_tensorboard also mirrors the local SB3 tensorboard scalars.
    wandb_run = None
    if args.wandb:
        import wandb
        if args.tb_dir is None:
            args.tb_dir = os.path.join("runs", "tb")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=f"{args.algo}_{args.env_type}_{args.sampling_strategy}",
            tags=[args.algo, args.env_type, args.sampling_strategy, f"seed{args.seed}"],
            config={
                "algo": args.algo,
                "env_type": args.env_type,
                "strategy": args.sampling_strategy,
                "seed": args.seed,
                "timesteps": args.timesteps,
                "mass_range": args.mass_range,
            },
            sync_tensorboard=True,
        )

    env = DummyVecEnv([lambda: make_env(args.env_type, args.mass_range, args.sampling_strategy, args.seed)])

    ckpt_path = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    algo_class = PPO if args.algo == "ppo" else SAC
    batch_size = args.batch_size if args.batch_size is not None else (256 if args.algo == "ppo" else 1024)

    # Resume from checkpoint if provided, otherwise start fresh
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[train] Resuming from {args.resume_from}")
        basename = os.path.basename(args.resume_from)
        m = re.match(r"(.+)_(\d+)_steps\.zip$", basename)
        if m:
            prefix, step = m.group(1), m.group(2)
            ckpt_dir = os.path.dirname(args.resume_from)
            vec_norm_path = os.path.join(ckpt_dir, f"{prefix}_vecnormalize_{step}_steps.pkl")
            replay_buffer_path = os.path.join(ckpt_dir, f"{prefix}_replay_buffer_{step}_steps.pkl")
        else:
            vec_norm_path = args.resume_from.replace(".zip", "_vecnormalize.pkl")
            replay_buffer_path = args.resume_from.replace(".zip", "_replay_buffer.pkl")

        if not os.path.exists(vec_norm_path):
            raise FileNotFoundError(
                f"VecNormalize stats missing next to checkpoint: {vec_norm_path}. "
                "Without them obs end up in the wrong scale."
            )
        env = VecNormalize.load(vec_norm_path, env)
        env.training = True
        env.norm_reward = True
        model = algo_class.load(args.resume_from, env=env, tensorboard_log=args.tb_dir)

        if args.algo == "sac":
            if not os.path.exists(replay_buffer_path):
                raise FileNotFoundError(
                    f"Replay buffer missing next to checkpoint: {replay_buffer_path}. "
                    "SAC needs it on resume or the policy collapses in a few k steps."
                )
            model.load_replay_buffer(replay_buffer_path)
            print(f"[train] Replay buffer loaded ({model.replay_buffer.size()} transitions)")

        remaining = max(0, args.timesteps - model.num_timesteps)
        reset_num = False
    else:
        print("[train] Starting fresh")
        # Normalize obs and reward: without this SAC/PPO converge slower
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        kwargs = dict(verbose=1, seed=args.seed, batch_size=batch_size,
                      device=args.device, tensorboard_log=args.tb_dir)
        if args.lr is not None:
            kwargs["learning_rate"] = args.lr
        if args.gamma is not None:
            kwargs["gamma"] = args.gamma
        if args.ent_coef is not None:
            kwargs["ent_coef"] = args.ent_coef if args.ent_coef == "auto" else float(args.ent_coef)
        if args.net_arch is not None:
            kwargs["policy_kwargs"] = dict(net_arch=[int(x) for x in args.net_arch.split(",")])
        if args.algo == "sac":
            if args.tau is not None:
                kwargs["tau"] = args.tau
            if args.train_freq is not None:
                kwargs["train_freq"] = args.train_freq
            if args.gradient_steps is not None:
                kwargs["gradient_steps"] = args.gradient_steps
        model = algo_class("MultiInputPolicy", env, **kwargs)
        remaining = args.timesteps
        reset_num = True

    # Periodic checkpoints
    callbacks = []
    if args.ckpt_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.ckpt_freq // env.num_envs, 1),
                save_path=ckpt_path,
                name_prefix="ckpt",
                save_replay_buffer=True,
                save_vecnormalize=True,
                verbose=1,
            )
        )

    # Held-out eval on the deployment env; EvalCallback syncs the VecNormalize stats, so this is real transfer.
    if args.eval_target:
        eval_env = DummyVecEnv([lambda: make_env(args.eval_env_type, args.mass_range, "none", args.seed + 10_000)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=ckpt_path,
                eval_freq=max(args.eval_freq // env.num_envs, 1),
                n_eval_episodes=50,
                deterministic=True,
                render=False,
                verbose=1,
            )
        )

    if args.wandb:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(verbose=1))

    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_num,
        tb_log_name=run_name,
    )

    save_name = os.path.join(args.model_dir, run_name)
    model.save(save_name)
    env.save(save_name + "_vecnormalize.pkl")
    print(f"[train] Final model saved to {save_name}")

    if wandb_run is not None:
        import wandb
        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(save_name + ".zip")
        artifact.add_file(save_name + "_vecnormalize.pkl")
        wandb_run.log_artifact(artifact)
        wandb_run.finish()


if __name__ == "__main__":
    main()
