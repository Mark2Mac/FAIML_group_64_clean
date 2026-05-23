
import gymnasium as gym

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment.
    """
    def __init__(
        self,
        env,
        mass_range=(1.0, 1.0),
        mode="none",
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range

        # global limits
        self.mass_min_limit, self.mass_max_limit = mass_range

        if self.mode == "adr":
            from collections import deque
            self.phi_low = 0.9
            self.phi_high = 1.1
            self.buffer_low = deque(maxlen=10)
            self.buffer_high = deque(maxlen=10)
            self.ep_reward = 0.0

    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_mass(self):

        if self.mode == "none":
            self.mass_min = self.mass_min_limit
            self.mass_max = self.mass_max_limit
            self.last_sample_type = "none"
            return None
        elif self.mode == "udr":
            import random
            self.mass_min = self.mass_min_limit
            self.mass_max = self.mass_max_limit
            self.last_sample_type = "interior"
            return random.uniform(self.mass_min, self.mass_max)
        elif self.mode == "adr":
            import random
            # prob 50% di campionare agli estremi
            if random.random() < 0.5:
                if random.random() < 0.5:
                    self.last_sample_type = "low"
                    self.mass_min = self.phi_low
                    self.mass_max = self.phi_low
                    return self.phi_low
                else:
                    self.last_sample_type = "high"
                    self.mass_min = self.phi_high
                    self.mass_max = self.phi_high
                    return self.phi_high
            else:
                self.last_sample_type = "interior"
                self.mass_min = self.phi_low
                self.mass_max = self.phi_high
                return random.uniform(self.phi_low, self.phi_high)
        else:
            raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented yet.")

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        # Update ADR boundaries
        if self.mode == "adr":
            self.ep_reward += float(reward)
            if done:
                if self.last_sample_type == "low":
                    self.buffer_low.append(self.ep_reward)
                    if len(self.buffer_low) == self.buffer_low.maxlen:
                        avg = sum(self.buffer_low) / len(self.buffer_low)
                        if avg > 50.0:
                            self.phi_low = max(self.mass_min_limit, self.phi_low - 0.1)
                        elif avg < -50.0:
                            self.phi_low = min(self.phi_high - 0.05, self.phi_low + 0.1)
                        self.buffer_low.clear()
                elif self.last_sample_type == "high":
                    self.buffer_high.append(self.ep_reward)
                    if len(self.buffer_high) == self.buffer_high.maxlen:
                        avg = sum(self.buffer_high) / len(self.buffer_high)
                        if avg > 50.0:
                            self.phi_high = min(self.mass_max_limit, self.phi_high + 0.1)
                        elif avg < -50.0:
                            self.phi_high = max(self.phi_low + 0.05, self.phi_high - 0.1)
                        self.buffer_high.clear()

        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------

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

            print(
                f"[{self.mode}] mass={new_mass:.2f} "
                f"range=[{self.mass_min:.2f},{self.mass_max:.2f}] "
                f"type={self.last_sample_type}"
            )

        return super().reset(**kwargs)
