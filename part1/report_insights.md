# FAIML RL Project - Report Insights (Part 1)

This document contains all quantitative results, key insights, environment specifications, and mathematical references for Part 1 of the CVPR report. All numbers are final (50,000 episodes, 3 independent runs per algorithm, 50 test episodes per run = 150 total).

## 0. Hopper Environment Specs (Task 1)

*   **State Space:** Continuous, 11-dimensional vector (`Box(-inf, inf, (11,), float64)`). Contains torso z-position, pitch angle, joint angles (thigh, leg, foot) and their respective velocities. Horizontal x-position is excluded to keep the policy position-invariant.
*   **Action Space:** Continuous, 3-dimensional vector (`Box(-1.0, 1.0, (3,), float32)`). Represents the continuous torque values applied to the three joints (thigh, leg, foot).
*   **Link Masses (Source/Target are identical in Part 1):**
    *   Torso: ~3.67 kg (exactly `3.66519143` kg)
    *   Thigh: ~4.06 kg (exactly `4.05789051` kg)
    *   Leg: ~2.78 kg (exactly `2.78135670` kg)
    *   Foot: ~5.32 kg (exactly `5.31557477` kg)

---

## 1. Hyperparameters & Evaluation Protocol

### Common Hyperparameters (all algorithms)

| Hyperparameter | Value | Justification |
|---|---|---|
| Learning rate | $3 \times 10^{-4}$ | Selected after preliminary sweeps at $10^{-3}$ and $7 \times 10^{-4}$, which exhibited excessive gradient variance and unstable convergence. The lower rate of $3 \times 10^{-4}$ provided a favorable trade-off between convergence speed and training stability. |
| Discount factor $\gamma$ | 0.99 | Standard value for continuous control tasks with long horizons [1]. |
| Hidden layers | 2 × 64 (tanh) | Compact architecture suitable for the 11-dim state space; orthogonal initialization with gain $\sqrt{2}$, policy head gain $0.01$ for near-zero initial actions. |
| Initial $\sigma$ | 0.5 | Learned per-action std via `softplus`, initialized at 0.5 to ensure sufficient initial exploration without overly noisy actions. |
| Training episodes | 50,000 | Sufficient for convergence assessment and to observe long-term stability/collapse phenomena. |
| Independent runs | 3 | Seeds 43, 44, 45 — enough to estimate inter-run variance. |
| Optimizer | Adam | Default $\beta_1=0.9$, $\beta_2=0.999$. |

### REINFORCE-Specific

| Parameter | Value |
|---|---|
| Baseline $b$ | 0 (no baseline) and 20 (constant) |
| Policy gradient loss | $\mathcal{L} = -\sum_t (G_t - b) \log \pi_\theta(a_t | s_t)$ |
| Update frequency | Per-episode (full Monte Carlo trajectory) |

### Actor-Critic-Specific

| Parameter | Value |
|---|---|
| Critic architecture | Separate 2×64 network, output $V(s) \in \mathbb{R}$ |
| TD target | $y_t = r_t + \gamma V(s_{t+1}) (1 - d_t)$ |
| Advantage normalization | $\hat{A} = \frac{A - \mu_A}{\sigma_A + 10^{-8}}$ (batch-level) |
| Loss | $\mathcal{L} = \mathcal{L}_{\text{actor}} + \mathcal{L}_{\text{critic}} = -\mathbb{E}[\hat{A} \log \pi] + \text{MSE}(V, y)$ |

### Evaluation Protocol

*   **Test episodes:** 50 per run (deterministic policy, $\mu$ of Gaussian)
*   **Total test data:** 150 episodes per algorithm (3 runs × 50)
*   **Metrics:** Mean return ± std, episode length, wall-clock training time

---

## 2. Methodology (Custom Implementation Details)

When describing algorithm implementations, focus on these non-trivial technical design choices:

*   **The Variance Problem (REINFORCE):** Cite **Sutton & Barto [1, Sec. 13.3]**. REINFORCE utilizes *Monte Carlo returns* ($G_t$). Since $G_t$ accumulates all rewards until the end of the episode, the trajectory is subject to high stochasticity, leading to high variance in the policy gradient estimates.

*   **The Constant Baseline (b=0 vs b=20):** Reference **[1, Sec. 13.4]**. Subtracting any action-independent baseline $b(s)$ from $G_t$ preserves the gradient's unbiasedness ($\sum_a \nabla \pi(a|s) \, b(s) = 0$). The variance of the gradient estimator is minimized when $b \approx \mathbb{E}[G_t]$. For the Hopper environment, typical returns lie in the range 600–1100. Therefore:
    *   $b = 0$: the vanilla estimator, where $(G_t - 0) = G_t$ is used directly.
    *   $b = 20$: subtracting 20 from returns of magnitude $\sim 700$ reduces them by only $\sim 3\%$, yielding a negligible variance reduction. This is expected by design: the purpose is to show that a poorly chosen constant baseline offers no tangible benefit, motivating the need for a *learned* baseline (i.e., the Critic in Actor-Critic).

*   **Actor-Critic & Advantage Normalization:**
    *   Cite **[1, Sec. 13.5]**. Instead of $G_t$, the Actor-Critic bootstraps using the 1-step TD-target ($r_{t+1} + \gamma V(s_{t+1})$). This drastically reduces variance but introduces *bias* (due to initial Critic approximation errors).
    *   **Custom Design:** Batch-level Advantage Standardization ($\hat{A} = \frac{A - \mu}{\sigma + \epsilon}$). Centering gradients (mean 0) and scaling them (unit variance) stabilizes the training process and prevents overly aggressive policy updates. However, this introduces a failure mode when advantage variance collapses (see Discussion).

---

## 3. Experimental Results

### 3.1 Summary Table

| Algorithm | Test Mean ± Std | Ep. Length | Time (min) |
|---|---|---|---|
| REINFORCE ($b=0$) | **1097.8 ± 722.0** | 255.7 | 208.2 |
| REINFORCE ($b=20$) | **1060.1 ± 829.6** | 224.0 | 533.8 |
| Actor-Critic | **939.7 ± 184.1** | 80.3 | 322.7 |

*Note: Test stats are pooled across 150 deterministic episodes (3 independent runs × 50 episodes). For all algorithms, the model evaluated during test time was the best checkpoint saved during training (based on the highest 100-episode rolling average). Episode length refers to the overall training average. Time is mean wall-clock per run.*

### 3.2 Per-Run Test Breakdown

| Algorithm | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| REINFORCE ($b=0$) | 1222.1 ± 794.9 | 771.7 ± 483.2 | 1299.6 ± 732.1 |
| REINFORCE ($b=20$) | 493.5 ± 540.5 | 792.1 ± 596.2 | 1894.7 ± 572.7 |
| Actor-Critic | 760.0 ± 78.5 | 1147.7 ± 20.2 | 911.5 ± 136.9 |

### 3.3 Actor-Critic Loss Values (during collapse phase)

| Run | Actor Loss | Critic Loss |
|---|---|---|
| 1 | −0.003 ± 0.151 | 21.0 ± 53.6 |
| 2 | −0.014 ± 0.143 | 8.0 ± 12.9 |
| 3 | −0.008 ± 0.136 | 14.4 ± 17.4 |

*Actor loss centered near zero confirms advantage normalization is working. However, the critic loss variance remains high, reflecting the instability that ultimately leads to the policy collapse in the second half of training.*

### 3.4 Key Observations

*   **REINFORCE $b=0$ vs $b=20$:** Test means are nearly identical (1097.8 vs 1060.1). As predicted by theory, since $b=20$ is only $\sim 3\%$ of the typical return magnitude, the effect on variance reduction is negligible. The inter-run variance of $b=20$ is even *higher* (test std 829.6 vs 722.0), confirming no meaningful benefit.
*   **REINFORCE vs Actor-Critic (from best checkpoint):** REINFORCE achieves higher test returns (1097.8 vs 939.7) with peak single-episode rewards above 2800, but with massive variance (std 722). Actor-Critic produces far more consistent results (std 184.1, best single run: 1147.7 ± 20.2).
*   **The Actor-Critic Collapse:** Despite achieving a respectable test score from its best checkpoint, the Actor-Critic agent experiences a catastrophic collapse during the second half of training. The final training returns drop to $\sim$160, and the average episode length drops to 80 steps (vs 256 for REINFORCE). The agent learns to fall quickly rather than survive.
*   **Visualizing the Data:** Insert the *Losses* and *Learning Curves* plots. The Actor-Critic loss plot demonstrates how the *Critic Loss* spikes during exploration phases, while the *Actor Loss* remains bounded around zero due to advantage normalization.

---

## 4. Discussion (Critical Analysis)

*   **The Hopper's "Local Minimum" (Bias vs. Variance Tradeoff):**
    *   *Why did REINFORCE hit 2800 while AC stalled at ~1150?* The Hopper environment grants a $+1.0$ "healthy reward" for every step the agent survives. Standing still for 1000 steps yields exactly 1000 reward.
    *   Because of the Critic's *bias*, the Actor-Critic quickly learns that standing still is a perfectly safe strategy ($V(s) \approx 1000$). If the agent attempts a jump (risking a fall and a low episode return of $\sim$200), the Critic computes a massively negative Advantage ($200 - 1000 = -800$), instantly penalizing the exploration.
    *   REINFORCE, being *unbiased* and having high variance, stumbles upon forward-running trajectories by pure chance (reward $\sim$2800). Without a conservative Critic to block risky behavior, it successfully learns a jumping gait.

*   **Why $b=20$ Cannot Help:**
    *   The optimal baseline is $b^* = \mathbb{E}[G_t]$, which for the Hopper lies around 600–1100 depending on training phase. Setting $b=20$ reduces the magnitude of $(G_t - b)$ by only $\sim 2$–$3\%$, which is insufficient to meaningfully reduce the variance of the policy gradient estimator $\text{Var}[\hat{g}] \propto \mathbb{E}[(G_t - b)^2]$. The results confirm this: both $b=0$ and $b=20$ produce statistically indistinguishable performance (test means: 1097.8 vs 1060.1, well within the inter-run variance).
    *   This motivates the Actor-Critic approach, where the baseline $b(s) = V_\phi(s)$ is *learned* and state-dependent, closely tracking $\mathbb{E}[G_t | S_t = s]$ — the theoretically optimal choice [1, Sec. 13.4].

*   **Catastrophic Forgetting and Actor-Critic Collapse:**
    *   The Actor-Critic learning curve shows a complete collapse in the second half of training. This is explained through the custom Advantage Normalization: once the agent masters standing still, every episode return is nearly identical ($\sim$1000). The variance of the Advantages approaches zero ($\sigma \to 0$).
    *   In the standardization formula $\frac{A - \mu}{\sigma + \epsilon}$, dividing by a microscopic $\sigma$ generates exploding gradients that destroy the neural network weights, leading to Catastrophic Forgetting. The final training mean of 160.1 (vs peak $\sim$1000) confirms total policy destruction.

*   **Hyperparameter Selection:**
    *   The learning rate $\alpha = 3 \times 10^{-4}$ was selected after preliminary experiments with $\alpha \in \{10^{-3}, 7 \times 10^{-4}\}$. Higher rates led to excessive gradient variance, manifesting as unstable learning curves with frequent reward collapses. The chosen rate provides adequate convergence speed while maintaining training stability across all three algorithms.

*   **Bridging to Part 2 (Motivating PPO and SAC):**
    *   Use the aforementioned collapse to justify the transition to advanced RL algorithms.
    *   Cite **Schulman et al. [8] (PPO)**: The failure of Vanilla Actor-Critic in continuous control stems from the lack of bounded policy updates. PPO introduces a *Clipped Surrogate Objective* to create a *Trust Region*, preventing exploding gradients from destroying the policy.
    *   Cite **Haarnoja et al. [7] (SAC)**: To solve the agent getting stuck in the safe "standing still" local minimum, SAC introduces *Entropy Maximization*. This mathematically forces the agent to explore, preventing premature convergence to suboptimal safe behaviors.
