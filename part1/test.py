import os
import gymnasium as gym
import numpy as np
import torch

from agent import Agent, Policy

# ── Config ───────────────────────────────────────────────────────────────────
N_TEST_EPISODES = 50   # episodes per seed
OUTPUT_DIR      = "part1/models"   # where test_returns_*.npy are saved

# ── Model paths for every seed of every algorithm ────────────────────────────
# Each entry: (algorithm_label, model_path)
# We prefer the _best.pth checkpoint; fall back to the regular one.
_BASE = "part1"

_MODEL_SPECS = {
    "REINFORCE (b=0)": [
        (
            f"{_BASE}/models_reinforce_70k_filippo_2026-06-07/"
            "policy_reinforce_70k_filippo_2026-06-07_baseline_0.0_run_1_best.pth",
            f"{_BASE}/models_reinforce_70k_filippo_2026-06-07/"
            "policy_reinforce_70k_filippo_2026-06-07_baseline_0.0_run_1.pth",
        ),
        (
            f"{_BASE}/models_reinforce_extra_seeds_2026-06-07/"
            "policy_reinforce_2026-06-07_baseline_0.0_seed_67_run_1_best.pth",
            f"{_BASE}/models_reinforce_extra_seeds_2026-06-07/"
            "policy_reinforce_2026-06-07_baseline_0.0_seed_67_run_1.pth",
        ),
        (
            f"{_BASE}/models_reinforce_extra_seeds_2026-06-07/"
            "policy_reinforce_2026-06-07_baseline_0.0_seed_128_run_1_best.pth",
            f"{_BASE}/models_reinforce_extra_seeds_2026-06-07/"
            "policy_reinforce_2026-06-07_baseline_0.0_seed_128_run_1.pth",
        ),
    ],
    "REINFORCE (b=20)": [
        (
            f"{_BASE}/models_reinforce_70k_filippo_2026-06-07/"
            "policy_reinforce_70k_filippo_2026-06-07_baseline_20.0_run_1_best.pth",
            f"{_BASE}/models_reinforce_70k_filippo_2026-06-07/"
            "policy_reinforce_70k_filippo_2026-06-07_baseline_20.0_run_1.pth",
        ),
        (
            f"{_BASE}/models_reinforce_extra_seeds_2026-06-07/"
            "policy_reinforce_2026-06-07_baseline_20.0_seed_67_run_1_best.pth",
            f"{_BASE}/models_reinforce_extra_seeds_2026-06-07/"
            "policy_reinforce_2026-06-07_baseline_20.0_seed_67_run_1.pth",
        ),
        (
            f"{_BASE}/models_reinforce_extra_seeds_2026-06-07/"
            "policy_reinforce_2026-06-07_baseline_20.0_seed_128_run_1_best.pth",
            f"{_BASE}/models_reinforce_extra_seeds_2026-06-07/"
            "policy_reinforce_2026-06-07_baseline_20.0_seed_128_run_1.pth",
        ),
    ],
    "Actor-Critic": [
        (
            f"{_BASE}/models_ac_70k_filippo_2026-06-07/"
            "policy_ac_70k_filippo_2026-06-07_run_1_best.pth",
            f"{_BASE}/models_ac_70k_filippo_2026-06-07/"
            "policy_ac_70k_filippo_2026-06-07_run_1.pth",
        ),
        (
            f"{_BASE}/models_ac_extra_seeds_2026-06-07/"
            "policy_ac_2026-06-07_seed_67_run_1_best.pth",
            f"{_BASE}/models_ac_extra_seeds_2026-06-07/"
            "policy_ac_2026-06-07_seed_67_run_1.pth",
        ),
        (
            f"{_BASE}/models_ac_extra_seeds_2026-06-07/"
            "policy_ac_2026-06-07_seed_128_run_1_best.pth",
            f"{_BASE}/models_ac_extra_seeds_2026-06-07/"
            "policy_ac_2026-06-07_seed_128_run_1.pth",
        ),
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def pick_path(best, fallback):
    """Return the best checkpoint path if it exists, otherwise the fallback."""
    if os.path.exists(best):
        return best
    if os.path.exists(fallback):
        print(f"  [WARN] _best not found, falling back to: {fallback}")
        return fallback
    return None


def eval_model(env, model_path, n_episodes):
    """Load a policy checkpoint and evaluate it deterministically."""
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = gym.make("Hopper-v4")

    print(f"\n{'='*60}")
    print(f" PART 1 EVALUATION — {N_TEST_EPISODES} deterministic episodes per seed")
    print(f"{'='*60}")

    summary = {}

    for algo_name, seed_pairs in _MODEL_SPECS.items():
        print(f"\n{'-'*60}")
        print(f" {algo_name}")
        print(f"{'-'*60}")

        algo_all_returns = []

        for seed_idx, (best_path, fallback_path) in enumerate(seed_pairs, start=1):
            model_path = pick_path(best_path, fallback_path)

            if model_path is None:
                print(f"  [SKIP] Seed {seed_idx}: no model file found.")
                print(f"         best:     {best_path}")
                print(f"         fallback: {fallback_path}")
                continue

            print(f"\n  Seed {seed_idx}/{len(seed_pairs)} — {model_path}")

            returns = eval_model(env, model_path, N_TEST_EPISODES)

            # save per-seed results
            tag = algo_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            save_path = os.path.join(OUTPUT_DIR, f"test_returns_{tag}_seed{seed_idx}.npy")
            np.save(save_path, returns)

            print(f"    Episodes : {N_TEST_EPISODES}")
            print(f"    Mean     : {np.mean(returns):.2f}")
            print(f"    Std      : {np.std(returns):.2f}")
            print(f"    Min      : {np.min(returns):.2f}")
            print(f"    Max      : {np.max(returns):.2f}")

            algo_all_returns.append(returns)

        if algo_all_returns:
            pooled = np.concatenate(algo_all_returns)
            print(f"  -- Aggregated ({len(pooled)} episodes across {len(algo_all_returns)} seeds) --")
            print(f"    Mean : {np.mean(pooled):.2f}")
            print(f"    Std  : {np.std(pooled):.2f}")
            print(f"    Min  : {np.min(pooled):.2f}")
            print(f"    Max  : {np.max(pooled):.2f}")

            tag = algo_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            np.save(os.path.join(OUTPUT_DIR, f"test_returns_{tag}_all.npy"), pooled)

            summary[algo_name] = {
                "mean": np.mean(pooled),
                "std":  np.std(pooled),
                "min":  np.min(pooled),
                "max":  np.max(pooled),
                "n":    len(pooled),
            }

    env.close()

    # ── Final summary table ──────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f" FINAL SUMMARY (pooled over all seeds)")
    print(f"{'='*60}")
    print(f"{'Algorithm':<22} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-'*60}")
    for algo, s in summary.items():
        print(f"{algo:<22} {s['n']:>4} {s['mean']:>8.2f} {s['std']:>8.2f} {s['min']:>8.2f} {s['max']:>8.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
