import os
import time
import argparse
import gymnasium as gym
import numpy as np
import torch
import random

from agent import Agent, Policy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train Actor-Critic on Hopper-v4")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for both actor and critic")

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--project", type=str, default="hopper-faiml", help="W&B project name")

    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--sigma-floor", type=float, default=0.1, help="Minimum policy std added after softplus")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy coefficient")

    parser.add_argument("--lr-scheduler", action="store_true", help="Enable linear LR decay scheduler")
    parser.add_argument("--lr-decay-start", type=int, default=10000, help="Episode at which LR starts decaying")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate at end of training")

    args = parser.parse_args()

    NUM_RUNS = args.runs
    NUM_EPISODES = args.episodes

    os.makedirs('part1/models', exist_ok=True)

    print(f"\n{'='*40}")
    print(f" ACTOR-CRITIC TRAINING")
    print(f" Episodes: {NUM_EPISODES} | Runs: {NUM_RUNS} | Learning Rate: {args.lr}")

    print(
        f" LR Scheduler: {args.lr_scheduler} | "
        f"Decay Start: {args.lr_decay_start} | "
        f"Min LR: {args.min_lr}"
    )
    print(f"{'='*40}")

    for run in range(1, NUM_RUNS + 1):
        print(f"\n--- Run {run}/{NUM_RUNS} ---")

        if args.wandb:
            import wandb
            wandb.init(
                entity="s355100-politecnico-di-torino",
                project=args.project,
                group="Actor-Critic",
                name=f"AC_run_{run}_lr_{args.lr}_entropy_{args.entropy_coef}",
                config={
                    "algorithm": "Actor-Critic",
                    "learning_rate": args.lr,
                    "episodes": NUM_EPISODES,
                    "run_id": run,

                    "seed": args.seed + run,

                    "gae_lambda": args.gae_lambda,
                    "sigma_floor": args.sigma_floor,

                    "entropy_coef": args.entropy_coef,

                    "lr_scheduler": args.lr_scheduler,
                    "lr_decay_start": args.lr_decay_start,
                    "min_lr": args.min_lr,
                },
                reinit=True
            )

        seed = args.seed + run
        set_seed(seed)
        if args.wandb: 
            wandb.config.update({"seed": seed})

        env = gym.make('Hopper-v4')
        policy = Policy(env.observation_space.shape[0], env.action_space.shape[0], sigma_floor=args.sigma_floor)
        agent = Agent(policy, lr=args.lr, gae_lambda=args.gae_lambda, entropy_coef=args.entropy_coef)
        
        scheduler = None
        if args.lr_scheduler:
            min_lr_ratio = args.min_lr / args.lr  # ratio at the end of training
            decay_start = args.lr_decay_start
            decay_episodes = max(NUM_EPISODES - decay_start, 1)

            def lr_lambda(episode):
                if episode < decay_start:
                    return 1.0  # constant LR
                # linear decay from 1.0 down to min_lr_ratio
                progress = (episode - decay_start) / decay_episodes
                return max(1.0 - progress * (1.0 - min_lr_ratio), min_lr_ratio)

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                agent.optimizer, lr_lambda=lr_lambda
            )

        rewards_log = []
        lengths_log = []
        actor_losses_log = []
        critic_losses_log = []

        # [NEW]
        diagnostics_log = []
        
        best_avg_reward = -float('inf')
        start_time = time.time()

        for episode in range(NUM_EPISODES):

            if episode == 0:
                state, _ = env.reset(seed=seed)
            else:
                state, _ = env.reset()

            episode_reward = 0
            step_count = 0

            while True:
                action, action_log_prob = agent.get_action(state)
                numpy_action = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
                next_state, reward, terminated, truncated, _ = env.step(numpy_action)
                done = terminated or truncated
                agent.store_outcome(state, next_state, action_log_prob, reward, done)
                state = next_state
                episode_reward += reward
                step_count += 1
                if done:
                    break

            actor_loss, critic_loss, diagnostics = agent.update_policy(actor_critic=True)
            rewards_log.append(episode_reward)
            lengths_log.append(step_count)
            actor_losses_log.append(actor_loss)
            critic_losses_log.append(critic_loss)
            
            # [NEW]
            diagnostics_log.append(diagnostics)

            avg100 = np.mean(rewards_log[-100:])

            if scheduler is not None:
                scheduler.step()
            current_lr = agent.optimizer.param_groups[0]['lr']


            if episode >= 100 and avg100 > best_avg_reward:
                best_avg_reward = avg100
                best_model_path = f"part1/models/policy_actor_critic_run_{run}_best.pth"
                torch.save(agent.policy.state_dict(), best_model_path)

            if args.wandb:
                wandb.log({
                    "episode": episode,
                    "reward": episode_reward,
                    "length": step_count,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "avg_reward_100": avg100,
                    "best_avg_reward": best_avg_reward,

                    # [NEW] additional actor critic diagnostics
                    "value_mean": diagnostics["value_mean"],
                    "value_std": diagnostics["value_std"],
                    "raw_adv_mean": diagnostics["raw_adv_mean"],
                    "raw_adv_std": diagnostics["raw_adv_std"],
                    "raw_adv_min": diagnostics["raw_adv_min"],
                    "raw_adv_max": diagnostics["raw_adv_max"],
                    "norm_adv_mean": diagnostics["norm_adv_mean"],
                    "norm_adv_std": diagnostics["norm_adv_std"],
                    "norm_adv_min": diagnostics["norm_adv_min"],
                    "norm_adv_max": diagnostics["norm_adv_max"],
                    "sigma_mean": diagnostics["sigma_mean"],
                    "sigma_min": diagnostics["sigma_min"],
                    "sigma_max": diagnostics["sigma_max"],

                    "current_lr": current_lr,
                    "entropy": diagnostics["entropy"],
                })

            if episode % 100 == 0:
                print(
                    f"Actor-Critic | Run {run} | Episode {episode:4d} | "
                    f"Reward: {episode_reward:8.2f} | Avg100: {avg100:8.2f} | Best: {best_avg_reward:8.2f} | "
                    f"LR: {current_lr:.2e}"
                )

        elapsed = time.time() - start_time
        print(f"Training time (actor-critic, run={run}): {elapsed/60:.1f} min")

        model_path = f"part1/models/policy_actor_critic_run_{run}.pth"
        torch.save(agent.policy.state_dict(), model_path)

        np.save(f"part1/models/rewards_actor_critic_run_{run}.npy", np.array(rewards_log))
        np.save(f"part1/models/lengths_actor_critic_run_{run}.npy", np.array(lengths_log))
        np.save(f"part1/models/actor_losses_actor_critic_run_{run}.npy", np.array(actor_losses_log))
        np.save(f"part1/models/critic_losses_actor_critic_run_{run}.npy", np.array(critic_losses_log))
        
        # [NEW]
        np.save(f"part1/models/diagnostics_actor_critic_run_{run}.npy", np.array(diagnostics_log, dtype=object))

        np.save(f"part1/models/time_actor_critic_run_{run}.npy", np.array([elapsed]))
        env.close()

        if args.wandb:
            wandb.finish()

if __name__ == '__main__':
    main()
