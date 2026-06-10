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
    parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")                              # specify it when runing
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs")                                       # specify it when runing
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for both actor and critic")

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")                                 # specify it when runing
    parser.add_argument("--project", type=str, default="hopper_REINFORCE_Actor_Critic", help="W&B project name")

    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--sigma-floor", type=float, default=0.1, help="Minimum policy std added after softplus")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")                                 # specify it when runing

    parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy coefficient")

    parser.add_argument("--lr-scheduler", action="store_true", help="Enable triggered LR decay")                                # specify it when runing
    parser.add_argument("--lr-trigger-avg-reward", type=float, default=1000.0)
    parser.add_argument("--lr-trigger-avg-length", type=float, default=500.0)
    parser.add_argument("--lr-trigger-best-reward", type=float, default=1500.0)
    parser.add_argument("--lr-trigger-episode", type=int, default=45000)
    parser.add_argument("--min-lr", type=float, default=1e-5)

    # [NEW]
    parser.add_argument("--output-dir", type=str, default="part1/models", help="Directory where models/results are saved")      # specify it when runing
    parser.add_argument("--run-tag", type=str, default="actor_critic", help="Tag used in saved filenames")                      # specify it when runing

    args = parser.parse_args()

    NUM_RUNS = args.runs
    NUM_EPISODES = args.episodes

    # os.makedirs('part1/models', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Run tag: {args.run_tag}")

    print(f"\n{'='*40}")
    print(f" ACTOR-CRITIC TRAINING")
    print(f" Episodes: {NUM_EPISODES} | Runs: {NUM_RUNS} | Learning Rate: {args.lr}")

    print(
    f" LR Scheduler: {args.lr_scheduler} | "
        f"Trigger avg_reward_100: {args.lr_trigger_avg_reward} | "
        f"Trigger avg_length_100: {args.lr_trigger_avg_length} | "
        f"Trigger best_avg_reward: {args.lr_trigger_best_reward} | "
        f"Trigger episode: {args.lr_trigger_episode} | "
        f"Min LR: {args.min_lr}"
    )

    print(f"{'='*40}")

    for run in range(1, NUM_RUNS + 1):
        print(f"\n--- Run {run}/{NUM_RUNS} ---")
        seed = args.seed + run - 1
        if args.wandb:
            import wandb
            wandb.init(
                entity="s355100-politecnico-di-torino",
                project=args.project,
                group=f"AC_seed_{seed}",
                name=f"AC_run_{run}_lr_{args.lr}_entropy_{args.entropy_coef}_seed_{seed}",
                config={
                    "algorithm": "Actor-Critic",
                    "learning_rate": args.lr,
                    "episodes": NUM_EPISODES,
                    "run_id": run,

                    "seed": args.seed + run - 1,

                    "gae_lambda": args.gae_lambda,
                    "sigma_floor": args.sigma_floor,

                    "entropy_coef": args.entropy_coef,

                    "lr_scheduler": args.lr_scheduler,
                    "lr_trigger_avg_reward": args.lr_trigger_avg_reward,
                    "lr_trigger_avg_length": args.lr_trigger_avg_length,
                    "lr_trigger_best_reward": args.lr_trigger_best_reward,
                    "lr_trigger_episode": args.lr_trigger_episode,
                    "min_lr": args.min_lr,
                },
                reinit=True
            )
            

        set_seed(seed)
        if args.wandb: 
            wandb.config.update({"seed": seed})

        env = gym.make('Hopper-v4')
        policy = Policy(env.observation_space.shape[0], env.action_space.shape[0], sigma_floor=args.sigma_floor)
        agent = Agent(policy, lr=args.lr, gae_lambda=args.gae_lambda, entropy_coef=args.entropy_coef)
        
        lr_decay_started = False
        lr_decay_start_episode = None
        initial_lr = args.lr

        rewards_log = []
        lengths_log = []
        actor_losses_log = []
        critic_losses_log = []

        # [NEW]
        diagnostics_log = []
        
        best_avg_reward = -float('inf')
        best_episode_reward = -float('inf')
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
            best_episode_reward = max(best_episode_reward, episode_reward)
            lengths_log.append(step_count)
            actor_losses_log.append(actor_loss)
            critic_losses_log.append(critic_loss)
            
            # [NEW]
            diagnostics_log.append(diagnostics)

            avg_reward_100 = np.mean(rewards_log[-100:])
            avg_length_100 = np.mean(lengths_log[-100:])
            reward_std_100 = np.std(rewards_log[-100:])

            if episode >= 100 and avg_reward_100 > best_avg_reward:
                best_avg_reward = avg_reward_100
                best_model_path = f"{args.output_dir}/policy_{args.run_tag}_seed_{seed}_run_{run}_best.pth"
                torch.save(agent.policy.state_dict(), best_model_path)


            if args.lr_scheduler:
                trigger_by_reward_and_length = (
                    avg_reward_100 >= args.lr_trigger_avg_reward and
                    avg_length_100 >= args.lr_trigger_avg_length
                )

                trigger_by_best_reward = best_avg_reward >= args.lr_trigger_best_reward
                trigger_by_episode = episode >= args.lr_trigger_episode

                if (not lr_decay_started) and (
                    trigger_by_reward_and_length or trigger_by_best_reward or trigger_by_episode
                ):
                    lr_decay_started = True
                    lr_decay_start_episode = episode
                    print(
                        f"[LR DECAY STARTED] Episode {episode} | "
                        f"AvgReward100: {avg_reward_100:.2f} | "
                        f"AvgLength100: {avg_length_100:.2f} | "
                        f"Best: {best_avg_reward:.2f}"
                    )

                if lr_decay_started:
                    progress = (episode - lr_decay_start_episode) / max(NUM_EPISODES - lr_decay_start_episode, 1)
                    progress = min(max(progress, 0.0), 1.0)

                    current_lr = initial_lr - progress * (initial_lr - args.min_lr)

                    for param_group in agent.optimizer.param_groups:
                        param_group["lr"] = current_lr
                else:
                    current_lr = initial_lr
            else:
                current_lr = agent.optimizer.param_groups[0]["lr"]

            if args.wandb:
                wandb.log({
                    "episode": episode,
                    "reward": episode_reward,
                    "best_episode_reward": best_episode_reward,
                    "length": step_count,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "avg_reward_100": avg_reward_100,
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

                    "avg_length_100": avg_length_100,
                    "reward_std_100": reward_std_100,
                    "lr_decay_started": int(lr_decay_started),
                })

            if episode % 100 == 0:
                print(
                    f"Actor-Critic | Run {run} | Episode {episode:4d} | "
                    f"Reward: {episode_reward:8.2f} | AvgReward100: {avg_reward_100:8.2f} | "
                    f"BestEpisode: {best_episode_reward:8.2f} | "
                    f"Len100: {avg_length_100:7.2f} | StdReward100: {reward_std_100:7.2f} | "
                    f"BestReward: {best_avg_reward:8.2f} | LR: {current_lr:.2e} | "
                    f"Decay: {lr_decay_started}"
                )

        elapsed = time.time() - start_time
        print(f"Training time (actor-critic, run={run}): {elapsed/60:.1f} min")

        model_path = f"{args.output_dir}/policy_{args.run_tag}_seed_{seed}_run_{run}.pth"
        torch.save(agent.policy.state_dict(), model_path)

        np.save(f"{args.output_dir}/rewards_{args.run_tag}_seed_{seed}_run_{run}.npy", np.array(rewards_log))
        np.save(f"{args.output_dir}/lengths_{args.run_tag}_seed_{seed}_run_{run}.npy", np.array(lengths_log))
        np.save(f"{args.output_dir}/actor_losses_{args.run_tag}_seed_{seed}_run_{run}.npy", np.array(actor_losses_log))
        np.save(f"{args.output_dir}/critic_losses_{args.run_tag}_seed_{seed}_run_{run}.npy", np.array(critic_losses_log))
        np.save(f"{args.output_dir}/diagnostics_{args.run_tag}_seed_{seed}_run_{run}.npy", np.array(diagnostics_log, dtype=object))
        np.save(f"{args.output_dir}/time_{args.run_tag}_seed_{seed}_run_{run}.npy", np.array([elapsed]))
        env.close()

        if args.wandb:
            wandb.finish()

if __name__ == '__main__':
    main()
