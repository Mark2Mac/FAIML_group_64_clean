"""
Plot script for Part 1 — generates all figures and tables for the report.
Compares REINFORCE (b=0, b=20) vs Actor-Critic across 3 runs.

Data is spread across four directories (date-stamped 2026-06-07):
  - models_reinforce_70k_filippo_2026-06-07   → REINFORCE run 1 (default seed)
  - models_reinforce_extra_seeds_2026-06-07   → REINFORCE runs 2 & 3 (seed 67, seed 128)
  - models_ac_70k_filippo_2026-06-07          → AC run 1 (default seed)
  - models_ac_extra_seeds_2026-06-07          → AC runs 2 & 3 (seed 67, seed 128)

Usage (run from project root):
    python part1/plot_results.py

Output: part1/figures/*.pdf  (vector graphics, ready for LaTeX)
        part1/figures/*.png  (raster copies)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ──────────────────────────────────────────────────────────────────
SMOOTH_WINDOW = 100          # rolling-average window (episodes)
FIGURES_DIR   = "part1/figures"

# Base directory (relative to project root, where this script is called from)
_BASE = "part1"

# ── Explicit file paths for every run (kind = "rewards" | "lengths" | ...) ─
# Each entry: (directory, filename_stem)
# The loader will build: <dir>/<kind>_<stem>.npy
_RUN_SPECS = {
    "REINFORCE (b=0)": [
        (_BASE + "/models_reinforce_70k_filippo_2026-06-07",
         "reinforce_70k_filippo_2026-06-07_baseline_0.0_run_1"),
        (_BASE + "/models_reinforce_extra_seeds_2026-06-07",
         "reinforce_2026-06-07_baseline_0.0_seed_67_run_1"),
        (_BASE + "/models_reinforce_extra_seeds_2026-06-07",
         "reinforce_2026-06-07_baseline_0.0_seed_128_run_1"),
    ],
    "REINFORCE (b=20)": [
        (_BASE + "/models_reinforce_70k_filippo_2026-06-07",
         "reinforce_70k_filippo_2026-06-07_baseline_20.0_run_1"),
        (_BASE + "/models_reinforce_extra_seeds_2026-06-07",
         "reinforce_2026-06-07_baseline_20.0_seed_67_run_1"),
        (_BASE + "/models_reinforce_extra_seeds_2026-06-07",
         "reinforce_2026-06-07_baseline_20.0_seed_128_run_1"),
    ],
    "Actor-Critic": [
        (_BASE + "/models_ac_70k_filippo_2026-06-07",
         "ac_70k_filippo_2026-06-07_run_1"),
        (_BASE + "/models_ac_extra_seeds_2026-06-07",
         "ac_2026-06-07_seed_67_run_1"),
        (_BASE + "/models_ac_extra_seeds_2026-06-07",
         "ac_2026-06-07_seed_128_run_1"),
    ],
}

ALGORITHMS = {
    "REINFORCE (b=0)":  {"color": "#e74c3c", "ls": "-"},
    "REINFORCE (b=20)": {"color": "#3498db", "ls": "-"},
    "Actor-Critic":     {"color": "#2ecc71", "ls": "-"},
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.1,
})


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_runs(algo_name, kind="rewards"):
    """Load all runs for algo_name, return list of 1-D numpy arrays."""
    specs = _RUN_SPECS.get(algo_name, [])
    runs = []
    for directory, stem in specs:
        path = os.path.join(directory, f"{kind}_{stem}.npy")
        if os.path.exists(path):
            runs.append(np.load(path))
        else:
            print(f"  [WARN] Missing: {path}")
    return runs


def load_runs_ac_losses(kind):
    """Shortcut for Actor-Critic actor_losses / critic_losses."""
    return load_runs("Actor-Critic", kind)


def smooth(data, window):
    """Simple rolling mean; expanding mean at the start."""
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


def _plot_mean_std(ax, runs, color, label, ls="-"):
    """Smooth each run, then plot mean ± std band."""
    mat      = runs_to_matrix(runs)
    smoothed = np.array([smooth(mat[i], SMOOTH_WINDOW) for i in range(len(runs))])
    mean     = np.nanmean(smoothed, axis=0)
    std      = np.nanstd(smoothed,  axis=0)
    episodes = np.arange(1, len(mean) + 1)
    ax.plot(episodes, mean, label=label, color=color, ls=ls, lw=2)
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=color)
    return mean, std, episodes


# ── Experiments plot (Giuseppe) ───────────────────────────────────────────────
def plot_experiments_reward():
    """
    Avg-reward-100 (rolling mean over 100 episodes) ± std across 3 seeds,
    for all three algorithms — the main 'Experiments' figure.
    Saved as experiments_reward.{pdf,png}.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, cfg in ALGORITHMS.items():
        runs = load_runs(name, "rewards")
        if not runs:
            print(f"  [SKIP] No reward data for {name}")
            continue
        _plot_mean_std(ax, runs, cfg["color"], name, cfg["ls"])

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Average Return (rolling window = {SMOOTH_WINDOW} ep.)", fontsize=12)
    ax.set_title("Training Performance: Average Reward ± Std over Seeds", fontsize=13, pad=10)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.85)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10_000))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "experiments_reward.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "experiments_reward.png"))
    print("  [OK] experiments_reward.pdf / .png")
    plt.close(fig)


# ── 1. Learning Curves (reward) ───────────────────────────────────────────────
def plot_learning_curves():
    fig, ax = plt.subplots(figsize=(8, 4))

    for name, cfg in ALGORITHMS.items():
        runs = load_runs(name, "rewards")
        if not runs:
            continue
        _plot_mean_std(ax, runs, cfg["color"], name, cfg["ls"])

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Return (rolling avg, w={SMOOTH_WINDOW})")
    ax.set_title("Learning Curves — REINFORCE vs Actor-Critic")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10_000))
    fig.savefig(os.path.join(FIGURES_DIR, "learning_curves.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "learning_curves.png"))
    print("  [OK] learning_curves.pdf")
    plt.close(fig)


# ── 2. Episode Length Curves ──────────────────────────────────────────────────
def plot_episode_lengths():
    fig, ax = plt.subplots(figsize=(8, 4))

    for name, cfg in ALGORITHMS.items():
        runs = load_runs(name, "lengths")
        if not runs:
            continue
        _plot_mean_std(ax, runs, cfg["color"], name, cfg["ls"])

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Episode Length (rolling avg, w={SMOOTH_WINDOW})")
    ax.set_title("Episode Lengths — REINFORCE vs Actor-Critic")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10_000))
    fig.savefig(os.path.join(FIGURES_DIR, "episode_lengths.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "episode_lengths.png"))
    print("  [OK] episode_lengths.pdf")
    plt.close(fig)


# ── 3. Test Returns — Box Plot ────────────────────────────────────────────────
def plot_test_returns():
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels, colors = [], [], []

    for name, cfg in ALGORITHMS.items():
        runs = load_runs(name, "test_returns")
        if not runs:
            continue
        all_returns = np.concatenate(runs)
        data.append(all_returns)
        labels.append(name)
        colors.append(cfg["color"])

    if not data:
        print("  [SKIP] No test returns found — run test.py first")
        plt.close(fig)
        return

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markeredgecolor='black',
                                  markerfacecolor='white', markersize=6))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel("Test Return (50 episodes)")
    ax.set_title("Test Performance — REINFORCE vs Actor-Critic on Hopper-v4")
    fig.savefig(os.path.join(FIGURES_DIR, "test_returns_boxplot.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "test_returns_boxplot.png"))
    print("  [OK] test_returns_boxplot.pdf")
    plt.close(fig)


# ── 4. Losses ─────────────────────────────────────────────────────────────────
def plot_losses():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # REINFORCE losses
    for name, cfg in ALGORITHMS.items():
        if "REINFORCE" not in name:
            continue
        runs = load_runs(name, "losses")
        if not runs:
            continue
        _plot_mean_std(ax1, runs, cfg["color"], name)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss (rolling avg)")
    ax1.set_title("REINFORCE Loss")
    ax1.legend()

    # Actor-Critic actor + critic losses
    actor_runs  = load_runs_ac_losses("actor_losses")
    critic_runs = load_runs_ac_losses("critic_losses")

    if actor_runs:
        _plot_mean_std(ax2, actor_runs, "#9b59b6", "Actor Loss")
    if critic_runs:
        _plot_mean_std(ax3, critic_runs, "#e67e22", "Critic Loss")

    ax2.set_xlabel("Episode");  ax2.set_ylabel("Loss (rolling avg)")
    ax2.set_title("Actor-Critic Actor Loss");  ax2.legend()
    ax3.set_xlabel("Episode");  ax3.set_ylabel("Loss (rolling avg)")
    ax3.set_title("Actor-Critic Critic Loss"); ax3.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "losses.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "losses.png"))
    print("  [OK] losses.pdf")
    plt.close(fig)


# ── 5. Training Time Bar Chart ────────────────────────────────────────────────
def plot_time_bar():
    fig, ax = plt.subplots(figsize=(6, 4))
    algo_names, means, stds, bar_colors = [], [], [], []

    for name, cfg in ALGORITHMS.items():
        runs = load_runs(name, "time")
        if not runs:
            continue
        all_times = np.concatenate(runs) / 60.0   # seconds → minutes
        algo_names.append(name)
        means.append(np.mean(all_times))
        stds.append(np.std(all_times))
        bar_colors.append(cfg["color"])

    if not algo_names:
        plt.close(fig)
        return

    x    = np.arange(len(algo_names))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.7,
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names)
    ax.set_ylabel("Training Time (minutes)")
    ax.set_title("Training Time Comparison")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{m:.1f}m", ha='center', va='center',
                fontweight='bold', color='white', fontsize=10)

    fig.savefig(os.path.join(FIGURES_DIR, "training_time_bar.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "training_time_bar.png"))
    print("  [OK] training_time_bar.pdf")
    plt.close(fig)


# ── 6. Summary Table ──────────────────────────────────────────────────────────
def print_summary_table():
    header = (f"{'Algorithm':<22} | {'Train Mean':>10} | {'Train Std':>10} |"
              f" {'Test Mean':>10} | {'Test Std':>10} | {'Ep. Len Mean':>12} | {'Time (min)':>10}")
    sep   = "-" * len(header)
    lines = [sep, header, sep]

    for name, cfg in ALGORITHMS.items():
        train_runs = load_runs(name, "rewards")
        test_runs  = load_runs(name, "test_returns")
        len_runs   = load_runs(name, "lengths")
        time_runs  = load_runs(name, "time")

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

        le_mean = (np.mean(np.concatenate([r[-500:] for r in len_runs]))
                   if len_runs else float('nan'))

        time_mean = (np.mean(np.concatenate(time_runs)) / 60.0
                     if time_runs else float('nan'))

        lines.append(
            f"{name:<22} | {tr_mean:>10.2f} | {tr_std:>10.2f} |"
            f" {te_mean:>10.2f} | {te_std:>10.2f} | {le_mean:>12.1f} | {time_mean:>10.1f}"
        )

    lines.append(sep)
    table = "\n".join(lines)
    print(table)

    with open(os.path.join(FIGURES_DIR, "summary_table.txt"), "w") as f:
        f.write(table + "\n")
    print("  [OK] summary_table.txt")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"\nGenerating figures in {FIGURES_DIR}/\n")

    # Primary experiments plot (avg reward ± std, window=100)
    plot_experiments_reward()

    # Supporting figures
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