import os
import time
import gymnasium as gym
import numpy as np
import torch

from agent import Agent, Policy

def main():
    baselines = [0.0, 20.0]
    NUM_RUNS = 3          
    NUM_EPISODES = 5000 
    
    os.makedirs('part1/models', exist_ok=True)

    for baseline in baselines:
        print(f"\n{'='*40}")
        print(f" INIZIO TRAINING BASELINE: {baseline}")
        print(f"{'='*40}")
        
        for run in range(1, NUM_RUNS + 1):
            print(f"\n--- Run {run}/{NUM_RUNS} ---")
            
            
            seed = 42 + run
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            env = gym.make('Hopper-v4')
            policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
            agent = Agent(policy)

            rewards_log = []
            lengths_log =[]
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

                agent.update_policy(baseline)
                rewards_log.append(episode_reward)
                lengths_log.append(step_count)

                if episode % 100 == 0:
                    avg100 = np.mean(rewards_log[-100:])
                    print(f"Baseline {baseline} | Run {run} | Episode {episode:4d} | "
                          f"Reward: {episode_reward:8.2f} | Avg100: {avg100:8.2f}")

            elapsed = time.time() - start_time
            print(f"Training time (baseline={baseline}, run={run}): {elapsed/60:.1f} min")

            # 2. Salva i file includendo il numero della run nel nome
            model_path = f"part1/models/policy_baseline_{baseline}_run_{run}.pth"
            torch.save(agent.policy.state_dict(), model_path)
            
            np.save(f"part1/models/rewards_baseline_{baseline}_run_{run}.npy", np.array(rewards_log))
            np.save(f"part1/models/lengths_baseline_{baseline}_run_{run}.npy", np.array(lengths_log))
            
            env.close()

if __name__ == '__main__':
    main()