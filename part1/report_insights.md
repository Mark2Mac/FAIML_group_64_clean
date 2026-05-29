# FAIML RL Project - Report Insights (Part 1)

This document contains key insights, environment specifications, and mathematical references to be used as a backbone for the CVPR report.

## 0. Hopper Environment Specs (Task 1)

*   **State Space:** Continuous, 11-dimensional vector (`Box(-inf, inf, (11,), float64)`). Contains torso z-position, pitch angle, joint angles (thigh, leg, foot) and their respective velocities. Horizontal x-position is excluded to keep the policy position-invariant.
*   **Action Space:** Continuous, 3-dimensional vector (`Box(-1.0, 1.0, (3,), float32)`). Represents the continuous torque values applied to the three joints (thigh, leg, foot).
*   **Link Masses (Source/Target are identical in Part 1):**
    *   Torso: ~3.67 kg (exactly `3.66519143` kg)
    *   Thigh: ~4.06 kg (exactly `4.05789051` kg)
    *   Leg: ~2.78 kg (exactly `2.78135670` kg)
    *   Foot: ~5.32 kg (exactly `5.31557477` kg)

---

## 2. Methodology (Custom Implementation Details)

When describing algorithm implementations, focus on these non-trivial technical design choices:

*   **The Variance Problem (REINFORCE):** Cite **Sutton & Barto [1, Sec. 13.3]**. Explain that REINFORCE utilizes *Monte Carlo returns* ($G_t$). Since $G_t$ accumulates all rewards until the end of the episode, the trajectory is subject to enormous stochasticity, leading to high variance in gradient estimates.
*   **The Constant Baseline Choice:** Reference **[1, Sec. 13.4]**. Mathematically, subtracting a baseline $b(s)$ does not introduce bias ($\sum_a \nabla \pi(a|s) b(s) = 0$). However, to effectively reduce variance, $b$ must closely approximate $\mathbb{E}[G_t]$. Explain that setting $b=0$ and $b=20$ was a deliberate choice to demonstrate that a static, incorrectly scaled baseline (since Hopper returns are in the 1000-2000 range) offers no tangible variance reduction.
*   **Actor-Critic & Advantage Normalization:** 
    *   Cite **[1, Sec. 13.5]**. Instead of $G_t$, the Actor-Critic bootstraps using the TD-target ($r_{t+1} + \gamma V(s_{t+1})$). This drastically reduces variance but introduces *bias* (due to initial Critic approximation errors).
    *   **Custom Design:** Detail the implementation of batch-level Advantage Standardization ($\hat{A} = \frac{A - \mu}{\sigma + \epsilon}$). Explain the mathematical reasoning: centering gradients (mean 0) and scaling them (variance 1) stabilizes the training process and prevents overly aggressive policy updates.

---

## 3. Experimental Results (What to Highlight)

The quantitative results tell a specific story about the algorithms:

*   **REINFORCE vs Actor-Critic (Test Returns):** Highlight that REINFORCE (b=0) exhibited massive standard deviation ($\sim 722$) with extremely high peak rewards (e.g., $2800+$). Conversely, Actor-Critic demonstrated very low variance ($\sim 184$) but failed to surpass a maximum return of $\sim 1300$.
*   **Visualizing the Data:** Insert the *Losses* and *Learning Curves* plots in the report. The Actor-Critic loss plot perfectly demonstrates how the *Critic Loss* spikes during exploration (falling), while the *Actor Loss* remains bounded around zero due to the advantage normalization.

---

## 4. Discussion (Critical Analysis)

This is the most important section for the final grade. Expand on these concepts:

*   **The Hopper's "Local Minimum" (Bias vs. Variance Tradeoff):**
    *   *Why did REINFORCE hit 2800 while AC stalled at 1150?* The Hopper environment grants a $+1.0$ "healthy reward" for every step the agent survives. Standing still for 1000 steps yields exactly 1000 reward.
    *   Because of the Critic's *bias*, the Actor-Critic quickly learns that standing still is a perfectly safe strategy ($V(s) \approx 1000$). If the agent attempts a jump (risking a fall and a low episode return of 200), the Critic computes a massively negative Advantage ($200 - 1000 = -800$), instantly penalizing the exploration.
    *   REINFORCE, being *unbiased* and having high variance, stumbles upon forward-running trajectories by pure chance (reward 2800). Without a conservative Critic to block risky behavior, it successfully learns a jumping gait.
*   **Catastrophic Forgetting and Actor-Critic Collapse:**
    *   The Actor-Critic learning curve shows a complete collapse in the second half of training. Explain this mathematically through the custom Advantage Normalization: once the agent perfectly masters standing still, every episode return is nearly identical (exactly 1000). The variance of the Advantages approaches zero ($\sigma \to 0$).
    *   In the standardization formula $\frac{A - \mu}{\sigma + \epsilon}$, dividing by a microscopic $\sigma$ generates exploding gradients that destroy the neural network weights, leading to Catastrophic Forgetting.
*   **Bridging to Part 2 (Motivating PPO and SAC):**
    *   Use the aforementioned collapse to justify the transition to advanced RL algorithms.
    *   Cite **Schulman et al. [8] (PPO)**: The failure of Vanilla Actor-Critic in continuous control stems from the lack of bounded policy updates. PPO introduces a *Clipped Surrogate Objective* to create a *Trust Region*, preventing exploding gradients from destroying the policy.
    *   Cite **Haarnoja et al. [7] (SAC)**: To solve the agent getting stuck in the safe "standing still" local minimum, SAC introduces *Entropy Maximization*. This mathematically forces the agent to explore, preventing premature convergence to suboptimal safe behaviors.
