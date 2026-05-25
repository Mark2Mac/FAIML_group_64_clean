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
 
