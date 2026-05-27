import os
import gymnasium as gym
import numpy as np
import torch

from agent import Agent, Policy

N_TEST_EPISODES = 50
NUM_RUNS = 3


def eval_model(env, model_path, n_episodes):
    """Carica un modello e valutalo per n_episodes episodi. Ritorna array di returns."""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = Policy(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    policy.eval()

    agent = Agent(policy)
    returns = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        while True:
            action, _ = agent.get_action(state, evaluation=True)
            numpy_action = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
            state, reward, terminated, truncated, _ = env.step(numpy_action)
            ep_reward += reward
            if terminated or truncated:
                break
        returns.append(ep_reward)

    return np.array(returns)


def main():
    env = gym.make('Hopper-v4')

    for baseline in [0.0, 20.0]:
        print(f"\n{'='*50}")
        print(f" BASELINE: {baseline}")
        print(f"{'='*50}")

        all_returns = []

        for run in range(1, NUM_RUNS + 1):
            best_model = f"part1/models/policy_baseline_{baseline}_run_{run}_best.pth"
            standard_model = f"part1/models/policy_baseline_{baseline}_run_{run}.pth"
            model_path = best_model if os.path.exists(best_model) else standard_model
            
            print(f"\n  Run {run}/{NUM_RUNS} — {model_path}")

            try:
                returns = eval_model(env, model_path, N_TEST_EPISODES)
            except FileNotFoundError:
                print(f"    [SKIP] File non trovato — esegui prima train.py")
                continue

            np.save(f"part1/models/test_returns_baseline_{baseline}_run_{run}.npy", returns)

            print(f"    Episodi : {N_TEST_EPISODES}")
            print(f"    Mean    : {np.mean(returns):.2f}")
            print(f"    Std     : {np.std(returns):.2f}")
            print(f"    Min     : {np.min(returns):.2f}")
            print(f"    Max     : {np.max(returns):.2f}")

            all_returns.append(returns)

        if all_returns:
            all_returns = np.concatenate(all_returns)
            print(f"\n  --- Aggregato ({len(all_returns)} episodi totali) ---")
            print(f"    Mean    : {np.mean(all_returns):.2f}")
            print(f"    Std     : {np.std(all_returns):.2f}")
            print(f"    Min     : {np.min(all_returns):.2f}")
            print(f"    Max     : {np.max(all_returns):.2f}")
            np.save(f"part1/models/test_returns_baseline_{baseline}_all.npy", all_returns)

    # --- Actor-Critic models ---
    print(f"\n{'='*50}")
    print(f" ACTOR-CRITIC")
    print(f"{'='*50}")

    all_returns = []

    for run in range(1, NUM_RUNS + 1):
        best_model = f"part1/models/policy_actor_critic_run_{run}_best.pth"
        standard_model = f"part1/models/policy_actor_critic_run_{run}.pth"
        model_path = best_model if os.path.exists(best_model) else standard_model
        
        print(f"\n  Run {run}/{NUM_RUNS} — {model_path}")

        try:
            returns = eval_model(env, model_path, N_TEST_EPISODES)
        except FileNotFoundError:
            print(f"    [SKIP] File non trovato — esegui prima train_ac.py")
            continue

        np.save(f"part1/models/test_returns_actor_critic_run_{run}.npy", returns)

        print(f"    Episodi : {N_TEST_EPISODES}")
        print(f"    Mean    : {np.mean(returns):.2f}")
        print(f"    Std     : {np.std(returns):.2f}")
        print(f"    Min     : {np.min(returns):.2f}")
        print(f"    Max     : {np.max(returns):.2f}")

        all_returns.append(returns)

    if all_returns:
        all_returns = np.concatenate(all_returns)
        print(f"\n  --- Aggregato ({len(all_returns)} episodi totali) ---")
        print(f"    Mean    : {np.mean(all_returns):.2f}")
        print(f"    Std     : {np.std(all_returns):.2f}")
        print(f"    Min     : {np.min(all_returns):.2f}")
        print(f"    Max     : {np.max(all_returns):.2f}")
        np.save(f"part1/models/test_returns_actor_critic_all.npy", all_returns)

    env.close()


if __name__ == '__main__':
    main()
