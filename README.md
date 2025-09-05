# CLIP-PPO Research Project

This repository contains implementations of [cleanrl's PPO](https://github.com/vwxyzjn/cleanrl) (Proximal Policy Optimization) and CLIP-PPO for robustness evaluation on MiniGrid environments.

## Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for CLIP processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AlexanderBurkhart/CLIP-PPO
cd clip-ppo
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Standard PPO

Run standard PPO on MiniGrid environments:

```bash
# Basic PPO training
python minigrid_experiments/ppo/ppo_minigrid.py

# PPO with visual disturbances
python minigrid_experiments/ppo/ppo_minigrid.py --apply_disturbances --disturbance_severity MODERATE

# Custom hyperparameters
python minigrid_experiments/ppo/ppo_minigrid.py --total_timesteps 1000000 --learning_rate 3e-4 --env_id MiniGrid-DoorKey-6x6-v0
```

### CLIP-PPO

Run CLIP-enhanced PPO with semantic alignment:

```bash
# Basic CLIP-PPO training
python minigrid_experiments/clip_ppo/clip_ppo_minigrid.py --clip_lambda 0.00001

# CLIP-PPO with visual disturbances
python minigrid_experiments/clip_ppo/clip_ppo_minigrid.py --clip_lambda 0.00001 --apply_disturbances --disturbance_severity HARD

# Enable verbose debugging
python minigrid_experiments/clip_ppo/clip_ppo_minigrid.py --clip_lambda 0.00001 --verbose

# Disable CLIP alignment (pure PPO baseline)
python minigrid_experiments/clip_ppo/clip_ppo_minigrid.py --clip_lambda 0.0
```

## Key Parameters

### Common Parameters
- `--env_id`: MiniGrid environment (default: `MiniGrid-Empty-16x16-v0`)
- `--total_timesteps`: Training duration (default: 250,000)
- `--num_envs`: Number of parallel environments (default: 8)
- `--learning_rate`: Optimizer learning rate (default: 2.5e-4)

### Disturbance Parameters
- `--apply_disturbances`: Enable visual disturbances during training
- `--disturbance_severity`: Severity level (`MILD`, `MODERATE`, `HARD`, `SEVERE`)

### CLIP-PPO Specific Parameters
- `--clip_lambda`: CLIP alignment loss coefficient (default: 0.00001)
- `--clip_model`: CLIP model variant (default: "ViT-B/32")
- `--verbose`: Enable detailed loss debugging output

### Model Saving Parameters
- `--save_model`: Enable model checkpointing (default: True)
- `--save_freq`: Save frequency in timesteps (default: 100,000)
- `--model_path`: Checkpoint directory (default: "checkpoints")

### Tracking Parameters
- `--track`: Enable Weights & Biases tracking
- `--wandb_project_name`: W&B project name (default: "cleanRL")
- `--capture_video`: Record agent performance videos (default: True)

## Disturbance Testing

Test visual disturbances on sample images:

```bash
python minigrid_experiments/disturbances_test.py --image_path your_image.png --severity MODERATE
```

## Disturbance Severity Levels

| Severity | Gaussian Noise σ | Blur σ | Contrast Range | Cutout % |
|----------|------------------|--------|----------------|----------|
| MILD     | 0.08            | 1.0    | (0.75, 1.25)   | 10%      |
| MODERATE | 0.12            | 2.0    | (0.70, 1.30)   | 17%      |
| HARD     | 0.13            | 2.1    | (0.69, 1.31)   | 18%      |
| SEVERE   | 0.26            | 3.0    | (0.60, 1.40)   | 25%      |

## Output Structure

Training generates the following outputs:

- **TensorBoard logs**: `runs/`
- **Videos**: `videos/minigrid/ppo/` or `videos/minigrid/clip_ppo/`
- **Checkpoints**: `checkpoints/`
