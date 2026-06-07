import os
import time
import argparse
import gymnasium as gym
import numpy as np
import torch

from agent import Agent, Policy

def main():
    parser = argparse.ArgumentParser(description="Train REINFORCE on Hopper-v4")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the policy")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--project", type=str, default="hopper-ac-gae-sigmaFloor-entropy-weirdLR", help="W&B project name")
    parser.add_argument("--output-dir", type=str, default="part1/models")
    parser.add_argument("--run-tag", type=str, default="reinforce")
    args = parser.parse_args()

    baselines = [0.0, 20.0]
    NUM_RUNS = args.runs
    NUM_EPISODES = args.episodes

    os.makedirs(args.output_dir, exist_ok=True)

    for baseline in baselines:
        print(f"\n{'='*40}")
        print(f" START TRAINING BASELINE: {baseline}")
        print(f" Episodes: {NUM_EPISODES} | Runs: {NUM_RUNS} | Learning Rate: {args.lr}")
        print(f"{'='*40}")

        for run in range(1, NUM_RUNS + 1):
            print(f"\n--- Run {run}/{NUM_RUNS} ---")

            if args.wandb:
                import wandb
                wandb.init(
                    project=args.project,
                    group=f"REINFORCE_b_{baseline}",
                    name=f"REINFORCE_b_{baseline}_run_{run}_lr_{args.lr}",
                    config={
                        "algorithm": "REINFORCE",
                        "baseline": baseline,
                        "learning_rate": args.lr,
                        "episodes": NUM_EPISODES,
                        "run_id": run,
                        "seed": 42 + run
                    },
                    reinit=True
                )

            seed = 42 + run
            torch.manual_seed(seed)
            np.random.seed(seed)

            env = gym.make('Hopper-v4')
            policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
            agent = Agent(policy, lr=args.lr)

            rewards_log = []
            lengths_log = []
            losses_log = []
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

                loss = agent.update_policy(baseline)
                rewards_log.append(episode_reward)
                best_episode_reward = max(best_episode_reward, episode_reward)
                lengths_log.append(step_count)
                losses_log.append(loss)

                avg100 = np.mean(rewards_log[-100:])

                if episode >= 100 and avg100 > best_avg_reward:
                    best_avg_reward = avg100
                    best_model_path = (
                        f"{args.output_dir}/policy_{args.run_tag}"
                        f"_baseline_{baseline}_run_{run}_best.pth"
                        )
                    torch.save(agent.policy.state_dict(), best_model_path)

                if args.wandb:
                    wandb.log({
                        "episode": episode,
                        "reward": episode_reward,
                        "length": step_count,
                        "loss": loss,
                        "avg_reward_100": avg100,
                        "best_avg_reward": best_avg_reward,
                        "best_episode_reward": best_episode_reward,
                    })

                if episode % 100 == 0:
                    print(
                        f"Baseline {baseline} | Run {run} | Episode {episode:4d} | "
                        f"Reward: {episode_reward:8.2f} | "
                        f"BestEpisode: {best_episode_reward:8.2f} | "
                        f"Avg100: {avg100:8.2f} | "
                        f"BestAvg100: {best_avg_reward:8.2f}"
                    )

            elapsed = time.time() - start_time
            print(f"Training time (baseline={baseline}, run={run}): {elapsed/60:.1f} min")

            # Save files including the run number in the filename
            model_path = (
                f"{args.output_dir}/policy_{args.run_tag}"
                f"_baseline_{baseline}_run_{run}.pth"
                )
            torch.save(agent.policy.state_dict(), model_path)

            np.save(
                f"{args.output_dir}/rewards_{args.run_tag}_baseline_{baseline}_run_{run}.npy",
                np.array(rewards_log)
            )

            np.save(
                f"{args.output_dir}/lengths_{args.run_tag}_baseline_{baseline}_run_{run}.npy",
                np.array(lengths_log)
            )

            np.save(
                f"{args.output_dir}/losses_{args.run_tag}_baseline_{baseline}_run_{run}.npy",
                np.array(losses_log)
            )

            np.save(
                f"{args.output_dir}/time_{args.run_tag}_baseline_{baseline}_run_{run}.npy",
                np.array([elapsed])
            )

            np.save(
                f"{args.output_dir}/best_episode_reward_{args.run_tag}_baseline_{baseline}_run_{run}.npy",
                np.array([best_episode_reward])
            )

            env.close()

            if args.wandb:
                wandb.finish()

if __name__ == '__main__':
    main()