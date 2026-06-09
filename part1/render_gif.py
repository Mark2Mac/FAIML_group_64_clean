"""Render a trained Hopper policy (REINFORCE / Actor-Critic) to an animated GIF.

Uses the Policy/Agent classes from agent.py. Forces an offscreen MuJoCo backend
so it works headless. Frames are subsampled and downscaled to keep the GIF small.

Example:
    python render_gif.py --model models_ac_70k_filippo_2026-06-07/policy_ac_70k_filippo_2026-06-07_run_1_best.pth \
        --episodes 3 --out ../assets/hopper.gif
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless offscreen rendering

import gymnasium as gym
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image

from agent import Agent, Policy


def render(model_path, episodes, out, width, height, stride, max_frames, fps):
    env = gym.make("Hopper-v4", render_mode="rgb_array", width=width, height=height)
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
    policy.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    policy.eval()
    agent = Agent(policy)

    frames = []
    for ep in range(episodes):
        state, _ = env.reset(seed=ep)
        step = 0
        while True:
            action, _ = agent.get_action(state, evaluation=True)
            action = action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
            state, _, term, trunc, _ = env.step(action)
            if step % stride == 0:
                frames.append(env.render())
            step += 1
            if term or trunc:
                break
        print(f"episode {ep + 1}/{episodes} -> {len(frames)} frames so far")
    env.close()

    frames = frames[:max_frames]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    imageio.mimsave(out, [np.asarray(Image.fromarray(f)) for f in frames], fps=fps, loop=0)
    print(f"saved {len(frames)} frames -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="../assets/hopper.gif")
    ap.add_argument("--width", type=int, default=720, help="render width")
    ap.add_argument("--height", type=int, default=480, help="render height")
    ap.add_argument("--stride", type=int, default=4, help="keep one frame every N steps")
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()
    render(a.model, a.episodes, a.out, a.width, a.height, a.stride, a.max_frames, a.fps)
