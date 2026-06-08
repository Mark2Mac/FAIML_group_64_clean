# FAIML RL project — group 64

Our code for the RL project: Hopper control in part 1, and the sim-to-real
push task (panda-gym) with domain randomization in part 2.

## Setup

```
pip install -r requirements.txt
```

Quick check that things work: `python part1/test_random_policy.py`.

Part 2 uses a local copy of panda-gym, install it from the folder:
`pip install -e part2/panda-gym`.

## What's where

- `part1/` — REINFORCE, REINFORCE + baseline and actor-critic on Hopper.
  Training in `train.py` and `train_ac.py`, testing in `test.py`.
- `part2/` — PPO and SAC on the push task, with the UDR/ADR randomization
  living in `rand_wrapper.py`. Training in `train_sb3.py`, eval in `eval_sb3.py`.
- `report/` — LaTeX source and the compiled `main.pdf`.

## Trainings, logs and model weights

We ran a lot of trainings, so the heavy stuff stays outside the repo:

- training curves on Weights & Biases: https://wandb.ai/s355100-politecnico-di-torino
- full models, tensorboard logs and extra figures on Drive:
  https://drive.google.com/drive/folders/1E1y1AwZ2oIPeDL7Y4RPE5VPml3dOItwC

The small Hopper policies (`.pth`) are already in `part1/`. The final SAC models
are in `part2/models/` — each one needs its `vecnormalize.pkl` next to it,
otherwise evaluation gives wrong numbers.
