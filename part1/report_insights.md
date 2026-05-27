## Task 1: Hopper Environment Specs

* **State Space:** Continuous, 11-dimensional vector (`Box(-inf, inf, (11,), float64)`).
  * Contains torso z-position, pitch angle, joint angles (thigh, leg, foot) and their respective velocities. Horizontal x-position is excluded to keep the policy position-invariant.
* **Action Space:** Continuous, 3-dimensional vector (`Box(-1.0, 1.0, (3,), float32)`).
  * Represents the continuous torque values applied to the three joints (thigh, leg, foot).
* **Link Masses:**
  * Torso: ~3.67 kg (exactly `3.66519143` kg)
  * Thigh: ~4.06 kg (exactly `4.05789051` kg)
  * Leg: ~2.78 kg (exactly `2.78135670` kg)
  * Foot: ~5.32 kg (exactly `5.31557477` kg)
  * *Note on source vs target:* In Part 1 (Hopper), the environment is identical for train and test. The source/target mass gap (shifting the mass by -4kg) only applies to the Panda-Gym cube in Part 2.

---

## REINFORCE with No Baseline (b = 0.0)

### Raw Results:
* **Mean Test Reward:** **119.51** (overall average of 150 episodes)
  * Run 1: **128.45** | Run 2: **90.73** | Run 3: **139.37** (high variance across seeds)
* **Average Episode Length:** **67.2 steps**
* **Average Train Time:** **6.3 minutes**

### Quick Insights:
* The agent learns to stand and execute some basic forward jumps but easily falls over after 60-70 steps.
* High variance across runs is a known issue of Monte Carlo estimators. Because we update using the return of the entire episode ($G_t$), any single lucky or unlucky action swings the gradients wildly.
* No baseline = noisy updates and slow, unstable convergence.

---

## REINFORCE with Constant Baseline (b = 20.0)

### Raw Results:
* **Mean Test Reward:** **53.32** (overall average)
  * Run 1: **0.61** (complete collapse) | Run 2: **54.72** | Run 3: **104.62**
* **Average Train Time:** **4.5 minutes** (Run 1 finished in just 1.3 mins because the episodes ended instantly)

### Quick Insights:
* **Why Run 1 collapsed:** At the start, the hopper falls instantly and gets tiny rewards (e.g. 5 to 10). Subtracting 20.0 makes the advantage negative ($G_t - 20 < 0$) for almost all actions. The agent gets penalized for everything, panics, and learns to fall down immediately to end the episode.
* **Why Run 3 survived:** By pure chance, a sequence of random actions got a reward $>20$. This gave positive gradients, allowing it to recover and learn to jump.
* **Takeaway:** Constant baselines are dangerous. If set too high, they kill early exploration. Baselines must be dynamic and adaptive.

---

## Actor-Critic (Dynamic Learned Baseline)

We evaluated two distinct variants of the Actor-Critic algorithm to investigate the impact of variance reduction on continuous control benchmarks:

### Variant A: Vanilla Actor-Critic (Without Normalization)
* **Mean Test Reward:** **243.34** (overall average of 150 episodes)
  * Run 1: **196.07** | Run 2: **182.05** | Run 3: **351.90** (Max single episode hit **765.14**!)
* **Average Episode Length:** **164.7 steps** (more than double REINFORCE)
* **Average Train Time:** **9.7 minutes** 
* **Takeaway:** High seed-to-seed variance. The model is highly dependent on initial exploration luck; Run 3 converged well, whereas Run 2 stalled and struggled.

### Variant B: Optimized Actor-Critic (With Advantage Normalization)
* **Mean Test Reward:** **236.95** (overall average of 150 episodes)
  * Run 1: **203.54** | Run 2: **275.77** *(Outstanding +51.4% improvement over Variant A Run 2!)* | Run 3: **231.55** (Max single episode hit **404.91**!)
* **Average Episode Length:** **168.2 steps**
* **Average Train Time:** **11.2 minutes** (Run 2 took 17.5 minutes because the Hopper survived extremely long periods!)
* **Takeaway:** Dramatic reduction in seed-to-seed variance. Standardizing advantages $A(s,a) = \frac{A - \mu}{\sigma + \epsilon}$ across each update centers the gradients and keeps the update steps bounded. The worst-performing seed (Run 2) is pulled up from 182.05 to a robust **275.77**, making the training process highly consistent and repeatable.

### Quick Insights:
* Massive improvement -> The agent actually learns to run stably.
* **Why it works:**
  1. **TD Bootstrapping:** Replacing the Monte Carlo return $G_t$ with a bootstrapped TD target ($r_t + \gamma V(s_{t+1})$) significantly reduces variance.
  2. **Dynamic Baseline:** The Critic $V^\phi(s_t)$ serves as a state-dependent baseline. The advantage represents whether an action did better or worse *relative to the expectation for that specific state*, rather than the whole episode.
* **Why the high standard deviation remains:** Vanilla AC doesn't limit the step size in policy space (no clipping like PPO) and doesn't promote exploration (no entropy term like SAC). One bad tilt in an unseen state causes a massive bad gradient step, leading to sudden policy collapse.

---

## Visual Rendering Analysis: The "Lazy Hopper" Phenomenon

When visualizing our best-trained policy (**Actor-Critic Run 3**), we observe a very distinctive physical behavior: instead of executing powerful, high-altitude jumps forward, the Hopper tends to **stand upright, balance on its single leg, and execute micro-steps or slide forward gently**.

This is a classic, mathematically expected **local minimum** in Reinforcement Learning continuous control tasks, caused by the interplay of two terms in the Hopper's reward function:

1. **The "Healthy Reward" Trap (+1.0 per step):** 
   The environment grants a constant reward of $+1.0$ for every step the Hopper remains upright. If the agent attempts a powerful jump, it risks destabilizing itself, falling, and ending the episode prematurely (thereby losing all future healthy rewards).
2. **The Control Cost Penalty (Torque cost):**
   The reward function penalizes high torques applied to the joints ($-0.001 \times \|u\|^2$). Since aggressive jumping requires massive forces, it incurs a substantial penalty.

### Conclusion of Visual Behavior:
The agent converges to a highly conservative **"survival policy"** (stalling around a reward of 200–500 over several hundred steps). It has learned that the safest, most cost-effective way to maximize cumulative reward is to prioritize standing still and balancing rather than risking a fall by jumping. 
To achieve aggressive jumping behavior (scores of 2,000+), one would need advanced exploration mechanisms (like Soft Actor-Critic's entropy maximization), step-size trust regions (like PPO), or a much longer training budget to overcome this strong local minimum.

---

## Dynamic Checkpointing & Stabilization Upgrades (Latest Modifications)

We introduced key codebase modifications to solve training instability and capture peak agent performance:

### 1. Bounded Policy Variance ($\sigma$ Floor)
* **What we did:** In `agent.py`, we added a mathematical floor of `+ 1e-3` to the policy's standard deviation parameter $\sigma$:
  $$\sigma = \text{Softplus}(\sigma_{\text{param}}) + 1e-3$$
* **Quick Insight:** In continuous action spaces, standard deviation parameterization can collapse toward zero during training. When $\sigma \to 0$, the log-probability calculation ($\log(\sigma)$) encounters numerical underflow or divides by zero, causing `NaN` gradients and complete policy collapse. Adding this tiny floor successfully stabilized both REINFORCE and Actor-Critic training.

### 2. Best Model Checkpointing (`_best.pth`)
* **What we did:** Modified `train.py` and `train_ac.py` to calculate a rolling average of the last 100 episodes (`avg100`). The network weights are automatically saved to `models/policy_..._best.pth` whenever `avg100` exceeds the historical maximum.
* **Quick Insight:** RL training on continuous tasks (especially without clipping/trust regions) is highly oscillatory. Saving only the "final episode" model risks saving an unstable, collapsed policy. Best Model Checkpointing ensures we capture the agent at its peak stability.

---

## Locomotion Breakthrough: Escaping the "Lazy Hopper" Minimum

In our latest Actor-Critic Run 1 (running with learning rate $0.0007$), we observed a fascinating training trajectory:

### 1. The Trajectory & The "Average Reward Drop" Paradox
* **The Stand-Still Phase (Episodes 14,200 - 15,500):** The agent quickly locked onto the "Lazy Hopper" policy, standing perfectly upright and achieving a highly stable reward of exactly $\approx 1000$ (maximizing survival bonus with zero control cost).
* **The Exploration Dip (Episodes 15,700 - 17,200):** As training progressed, the agent began exploring forward-leaning and dynamic jumping actions. Because the Hopper was sbilanciando (leaning) to push forward, it fell over repeatedly after only 200–300 steps. These early falls dropped the individual episode rewards to ~200-300, pulling the rolling average (`Avg100`) down from $1000$ to **~222**.
* **Report Writer Insight:** This crollo (drop) in average reward is **not a policy collapse**; it is the mathematical consequence of exploration. In order to transition from a safe, stationary posture (local minimum) to active locomotion, the agent must destabilize itself, which temporarily increases the failure rate before stable coordination is learned.

### 2. Physical Breakthrough (Testing `policy_actor_critic_run_1_best.pth`)
When testing this new best checkpoint (saved when the agent successfully combined forward velocity with long-term survival), we confirmed the agent successfully **escaped the stationary local minimum** and learned an active jumping gait:
* **Test Episode 1: 1000 steps (Full Survival) | Cumulative Reward: 1129.74**
* **Test Episode 2: 872 steps | Cumulative Reward: 1031.72**
* **Test Episode 3: 778 steps | Cumulative Reward: 940.63**

### Quick Insights for the Report:
* **Proof of Forward Movement:** Because the max survival bonus over 1000 steps is 1000 (1.0 per step), achieving **1129.74** reward and surviving the full 1000 steps is **mathematical proof of positive forward velocity** ($v_x > 0$), as the velocity rewards outweighed the control penalties!
* **Comparison:** While the previous "best" model stood perfectly still to get 1000, this new best model actually hops forward, covering distance while maintaining balance for most of the episode.

