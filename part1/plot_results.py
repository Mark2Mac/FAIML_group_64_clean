"""
Plot script for Part 1 — generates all figures and tables for the report.
Compares REINFORCE (b=0, b=20) vs Actor-Critic across 3 runs.

Usage:
    cd part1
    python plot_results.py

Output: part1/figures/*.pdf  (vector graphics, ready for LaTeX)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ──────────────────────────────────────────────────────────────────
NUM_RUNS = 3
SMOOTH_WINDOW = 100          # rolling average window for learning curves
FIGURES_DIR = "part1/figures"
MODELS_DIR = "part1/models"

ALGORITHMS = {
    "REINFORCE (b=0)":   {"prefix": "baseline_0.0",     "color": "#e74c3c", "ls": "-"},
    "REINFORCE (b=20)":  {"prefix": "baseline_20.0",    "color": "#3498db", "ls": "-"},
    "Actor-Critic":      {"prefix": "actor_critic",      "color": "#2ecc71", "ls": "-"},
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_runs(prefix, kind="rewards"):
    """Load NUM_RUNS .npy arrays, return list (may be shorter if files missing)."""
    runs = []
    for r in range(1, NUM_RUNS + 1):
        path = os.path.join(MODELS_DIR, f"{kind}_{prefix}_run_{r}.npy")
        if os.path.exists(path):
            runs.append(np.load(path))
        else:
            print(f"  [WARN] Missing: {path}")
    return runs


def smooth(data, window):
    """Simple rolling mean; pads the start with the expanding mean."""
    out = np.empty_like(data, dtype=float)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        out[i] = np.mean(data[start:i + 1])
    return out


def runs_to_matrix(runs):
    """Stack runs (possibly different lengths) into (n_runs, max_len) with NaN padding."""
    max_len = max(len(r) for r in runs)
    mat = np.full((len(runs), max_len), np.nan)
    for i, r in enumerate(runs):
        mat[i, :len(r)] = r
    return mat


# ── 1. Learning Curves (reward) ────────────────────────────────────────────
def plot_learning_curves():
    fig, ax = plt.subplots(figsize=(8, 4))

    for name, cfg in ALGORITHMS.items():
        runs = load_runs(cfg["prefix"], "rewards")
        if not runs:
            continue
        mat = runs_to_matrix(runs)
        # Smooth each run individually, then compute stats
        smoothed = np.array([smooth(mat[i], SMOOTH_WINDOW) for i in range(len(runs))])
        mean = np.nanmean(smoothed, axis=0)
        std  = np.nanstd(smoothed, axis=0)
        episodes = np.arange(1, len(mean) + 1)

        ax.plot(episodes, mean, label=name, color=cfg["color"], ls=cfg["ls"], lw=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=cfg["color"])

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Return (rolling avg, w={SMOOTH_WINDOW})")
    ax.set_title("Learning Curves — REINFORCE vs Actor-Critic")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    fig.savefig(os.path.join(FIGURES_DIR, "learning_curves.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "learning_curves.png"))
    print("  [OK] learning_curves.pdf")
    plt.close(fig)


# ── 2. Episode Length Curves ────────────────────────────────────────────────
def plot_episode_lengths():
    fig, ax = plt.subplots(figsize=(8, 4))

    for name, cfg in ALGORITHMS.items():
        runs = load_runs(cfg["prefix"], "lengths")
        if not runs:
            continue
        mat = runs_to_matrix(runs)
        smoothed = np.array([smooth(mat[i], SMOOTH_WINDOW) for i in range(len(runs))])
        mean = np.nanmean(smoothed, axis=0)
        std  = np.nanstd(smoothed, axis=0)
        episodes = np.arange(1, len(mean) + 1)

        ax.plot(episodes, mean, label=name, color=cfg["color"], ls=cfg["ls"], lw=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=cfg["color"])

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Episode Length (rolling avg, w={SMOOTH_WINDOW})")
    ax.set_title("Episode Lengths — REINFORCE vs Actor-Critic")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    fig.savefig(os.path.join(FIGURES_DIR, "episode_lengths.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "episode_lengths.png"))
    print("  [OK] episode_lengths.pdf")
    plt.close(fig)


# ── 3. Test Returns — Box Plot ──────────────────────────────────────────────
def plot_test_returns():
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels, colors = [], [], []

    for name, cfg in ALGORITHMS.items():
        runs = load_runs(cfg["prefix"], "test_returns")
        if not runs:
            continue
        all_returns = np.concatenate(runs)
        data.append(all_returns)
        labels.append(name)
        colors.append(cfg["color"])

    if not data:
        print("  [SKIP] No test returns found — run test.py first")
        return

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=6))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel("Test Return (50 episodes)")
    ax.set_title("Test Performance — REINFORCE vs Actor-Critic on Hopper-v4")
    fig.savefig(os.path.join(FIGURES_DIR, "test_returns_boxplot.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "test_returns_boxplot.png"))
    print("  [OK] test_returns_boxplot.pdf")
    plt.close(fig)


# ── 5. Losses ───────────────────────────────────────────────────────────────
def plot_losses():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # REINFORCE Losses
    for name, cfg in ALGORITHMS.items():
        if "REINFORCE" not in name:
            continue
        runs = load_runs(cfg["prefix"], "losses")
        if not runs:
            continue
        mat = runs_to_matrix(runs)
        smoothed = np.array([smooth(mat[i], SMOOTH_WINDOW) for i in range(len(runs))])
        mean = np.nanmean(smoothed, axis=0)
        std  = np.nanstd(smoothed, axis=0)
        episodes = np.arange(1, len(mean) + 1)
        ax1.plot(episodes, mean, label=name, color=cfg["color"], lw=2)
        ax1.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=cfg["color"])
        
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss (rolling avg)")
    ax1.set_title("REINFORCE Loss")
    ax1.legend()
    
    # Actor-Critic Losses
    ac_name = "Actor-Critic"
    ac_cfg = ALGORITHMS[ac_name]
    actor_runs = load_runs(ac_cfg["prefix"], "actor_losses")
    critic_runs = load_runs(ac_cfg["prefix"], "critic_losses")
    
    if actor_runs and critic_runs:
        mat_actor = runs_to_matrix(actor_runs)
        mat_critic = runs_to_matrix(critic_runs)
        sm_actor = np.array([smooth(mat_actor[i], SMOOTH_WINDOW) for i in range(len(actor_runs))])
        sm_critic = np.array([smooth(mat_critic[i], SMOOTH_WINDOW) for i in range(len(critic_runs))])
        
        m_a, s_a = np.nanmean(sm_actor, axis=0), np.nanstd(sm_actor, axis=0)
        m_c, s_c = np.nanmean(sm_critic, axis=0), np.nanstd(sm_critic, axis=0)
        ep = np.arange(1, len(m_a) + 1)
        
        ax2.plot(ep, m_a, label="Actor Loss", color="#9b59b6", lw=2)
        ax2.fill_between(ep, m_a - s_a, m_a + s_a, alpha=0.15, color="#9b59b6")
        ax2.plot(ep, m_c, label="Critic Loss", color="#e67e22", lw=2)
        ax2.fill_between(ep, m_c - s_c, m_c + s_c, alpha=0.15, color="#e67e22")
        
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss (rolling avg)")
    ax2.set_title("Actor-Critic Losses")
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "losses.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "losses.png"))
    print("  [OK] losses.pdf")
    plt.close(fig)

# ── 6. Training Time Bar Chart ─────────────────────────────────────────────
def plot_time_bar():
    fig, ax = plt.subplots(figsize=(6, 4))
    algo_names, means, stds, bar_colors = [], [], [], []

    for name, cfg in ALGORITHMS.items():
        runs = load_runs(cfg["prefix"], "time")
        if not runs:
            continue
        all_times = np.concatenate(runs) / 60.0  # to minutes
        algo_names.append(name)
        means.append(np.mean(all_times))
        stds.append(np.std(all_times))
        bar_colors.append(cfg["color"])

    if not algo_names:
        return

    x = np.arange(len(algo_names))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.7,
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names)
    ax.set_ylabel("Training Time (minutes)")
    ax.set_title("Training Time Comparison")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{m:.1f}m", ha='center', va='center', fontweight='bold', color='white', fontsize=10)

    fig.savefig(os.path.join(FIGURES_DIR, "training_time_bar.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "training_time_bar.png"))
    print("  [OK] training_time_bar.pdf")
    plt.close(fig)


# ── 7. Summary Table (printed to console & saved as txt) ───────────────────
def print_summary_table():
    header = f"{'Algorithm':<22} | {'Train Mean':>10} | {'Train Std':>10} | {'Test Mean':>10} | {'Test Std':>10} | {'Ep. Len Mean':>12} | {'Time (min)':>10}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for name, cfg in ALGORITHMS.items():
        train_runs = load_runs(cfg["prefix"], "rewards")
        test_runs  = load_runs(cfg["prefix"], "test_returns")
        len_runs   = load_runs(cfg["prefix"], "lengths")
        time_runs  = load_runs(cfg["prefix"], "time")

        # Last 500 episodes of training (converged performance)
        if train_runs:
            tail = np.concatenate([r[-500:] for r in train_runs])
            tr_mean, tr_std = np.mean(tail), np.std(tail)
        else:
            tr_mean, tr_std = float('nan'), float('nan')

        if test_runs:
            all_test = np.concatenate(test_runs)
            te_mean, te_std = np.mean(all_test), np.std(all_test)
        else:
            te_mean, te_std = float('nan'), float('nan')

        if len_runs:
            tail_len = np.concatenate([r[-500:] for r in len_runs])
            le_mean = np.mean(tail_len)
        else:
            le_mean = float('nan')

        if time_runs:
            time_mean = np.mean(np.concatenate(time_runs)) / 60.0 # to minutes
        else:
            time_mean = float('nan')

        lines.append(f"{name:<22} | {tr_mean:>10.2f} | {tr_std:>10.2f} | {te_mean:>10.2f} | {te_std:>10.2f} | {le_mean:>12.1f} | {time_mean:>10.1f}")

    lines.append(sep)
    table = "\n".join(lines)
    print(table)

    with open(os.path.join(FIGURES_DIR, "summary_table.txt"), "w") as f:
        f.write(table + "\n")
    print("  [OK] summary_table.txt")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"\nGenerating figures in {FIGURES_DIR}/\n")

    plot_learning_curves()
    plot_episode_lengths()
    plot_test_returns()
    plot_losses()
    plot_time_bar()
    print()
    print_summary_table()

    print(f"\nDone. All figures saved in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()