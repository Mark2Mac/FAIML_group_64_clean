
import gymnasium as gym
from collections import deque
import random

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment.
    """
    def __init__(
        self,
        env,
        mass_range=(0.5, 6.0),
        mode="none",
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range

        self.mass_min_limit, self.mass_max_limit = mass_range
        self.last_sample_type = "none"

        if self.mode == "adr":
            # ADR starts at source domain mass (1.0 kg) and expands adaptively
            self.phi_low = 1.0
            self.phi_high = 1.0

            self.buffer_low = deque(maxlen=20)
            self.buffer_high = deque(maxlen=20)
            self.ep_reward = 0.0

            self.threshold_expand = -4.0
            self.threshold_contract = -12.0
            self.step_size = 0.1
            self.min_gap = 0.05


    def _sample_mass(self):

        if self.mode == "none":
            self.last_sample_type = "none"
            return None

        elif self.mode == "udr":
            self.last_sample_type = "interior"
            return random.uniform(self.mass_min_limit, self.mass_max_limit)

        elif self.mode == "adr":
            # 50% boundary sampling, 50% interior (AutoDR)
            if random.random() < 0.5:
                if random.random() < 0.5:
                    self.last_sample_type = "low"
                    return self.phi_low
                else:
                    self.last_sample_type = "high"
                    return self.phi_high
            else:
                self.last_sample_type = "interior"
                return random.uniform(self.phi_low, self.phi_high)
        else:
            raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented yet.")

    # --- Step & ADR Logic ---

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if self.mode == "adr":
            self.ep_reward += float(reward)
            if done:
                self._update_adr_boundaries()

        return obs, reward, terminated, truncated, info

    def _update_adr_boundaries(self):
        """Update ADR boundaries at the end of an episode."""
        if self.last_sample_type == "low":
            self.buffer_low.append(self.ep_reward)
            self.phi_low = self._compute_new_boundary(
                buffer=self.buffer_low,
                current_phi=self.phi_low,
                is_lower_bound=True,
            )

        elif self.last_sample_type == "high":
            self.buffer_high.append(self.ep_reward)
            self.phi_high = self._compute_new_boundary(
                buffer=self.buffer_high,
                current_phi=self.phi_high,
                is_lower_bound=False,
            )

    def _compute_new_boundary(self, buffer, current_phi, is_lower_bound):
        """Compute updated phi value based on average return in the buffer."""
        if len(buffer) < buffer.maxlen:
            return current_phi

        avg_return = sum(buffer) / len(buffer)
        buffer.clear()

        if avg_return > self.threshold_expand:
            if is_lower_bound:
                return max(self.mass_min_limit, current_phi - self.step_size)
            else:
                return min(self.mass_max_limit, current_phi + self.step_size)

        elif avg_return < self.threshold_contract:
            if is_lower_bound:
                return min(self.phi_high - self.min_gap, current_phi + self.step_size)
            else:
                return max(self.phi_low + self.min_gap, current_phi - self.step_size)

        return current_phi

    # --- Reset ---

    def reset(self, **kwargs):

        if self.mode == "adr":
            self.ep_reward = 0.0

        new_mass = self._sample_mass()

        if new_mass is not None:

            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

        return super().reset(**kwargs)
