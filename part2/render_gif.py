"""Render a trained SAC/PPO PandaPush policy to an animated GIF (or MP4).

Reuses the model/VecNormalize loading convention of eval_sb3.py.
Frames are captured from the raw panda-gym env via render_mode="rgb_array".

Example:
    python render_gif.py --model models/sac_target_none_seed2.zip --algo sac \
        --env-type target --episodes 3 --out ../report/imgs/sac_target.gif
"""
import argparse
import os
import pickle
import sys

import numpy.core  # noqa: F401
# Models/normalizers were saved under numpy 2.x (module path numpy._core);
# alias them to numpy.core so they load under the numpy<2 required by pybullet.
for _m in list(sys.modules):
    if _m == "numpy.core" or _m.startswith("numpy.core."):
        sys.modules["numpy._core" + _m[len("numpy.core"):]] = sys.modules[_m]

import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import panda_gym  # noqa: F401 - registers the Panda envs


class _NoState:  # absorbs any numpy.random RNG; unused in deterministic eval
    def __init__(self, *a, **k):
        pass
    def __setstate__(self, state):
        pass
    def __reduce__(self):
        return (object, ())
def _stub(*a, **k):
    return _NoState()
class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy.random"):
            return _stub  # pkl saved under numpy 2.x; skip incompatible RNG state
        return super().find_class(module, name)


def _load_vecnormalize(path, venv):
    with open(path, "rb") as f:
        vn = _TolerantUnpickler(f).load()
    vn.set_venv(venv)
    vn.training = False
    vn.norm_reward = False
    return vn


def render(model_path, algo, env_type, episodes, deterministic, out, fps):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    base_env = gym.make("PandaPush-v3", render_mode="rgb_array", type=env_type,
                        reward_type="dense")
    env = DummyVecEnv([lambda: base_env])
    vec_norm_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vec_norm_path):
        env = _load_vecnormalize(vec_norm_path, env)

    raw = env.venv.envs[0] if isinstance(env, VecNormalize) else env.envs[0]
    # Override pickled objects that embed a numpy-2.x RNG (gym spaces, schedules)
    # so loading works under numpy<2; spaces are taken from the live env instead.
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = (PPO if algo == "ppo" else SAC).load(
        model_path, env=env, custom_objects=custom_objects)

    frames = []
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, infos = env.step(action)
            frame = raw.render()
            if frame is not None:
                frames.append(frame)
            done = bool(dones[0])
        print(f"episode {ep}/{episodes} captured ({len(frames)} frames total)")

    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if out.lower().endswith(".mp4"):
        imageio.mimsave(out, frames, fps=fps)
    else:
        imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"saved {len(frames)} frames -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    ap.add_argument("--env-type", choices=["source", "target"], default="target")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--out", default="rollout.gif", help=".gif or .mp4")
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    render(a.model, a.algo, a.env_type, a.episodes, a.deterministic, a.out, a.fps)
