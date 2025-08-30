# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a CLIP-PPO research repository for ICLR that implements PPO (Proximal Policy Optimization) algorithms enhanced with CLIP-based semantic regularization. The codebase contains two main implementations:

- `minigrid_experiments/ppo/ppo_minigrid.py`: Standard PPO implementation for MiniGrid environments
- `minigrid_experiments/clip_ppo/clip_ppo_minigrid.py`: CLIP-enhanced PPO implementation with semantic regularization

## Project Design Context

### Research Objective
CLIP-PPO augments standard PPO with CLIP-based semantic regularization to improve robustness to visual disturbances. The method adds a loss term that aligns the agent's latent state representation with CLIP text embeddings of ground-truth symbolic environment descriptions, reducing overfitting to spurious pixels.

### Target Environments
1. **MiniGrid**: Development and testing (any suitable task for quick iteration)
2. **Procgen**: Final results (train on 200 levels, test on 10k unseen levels)
   - Starpilot: sensitivity to blur/contrast
   - CoinRun: semantic regularities across unseen layouts
3. **Atari-100k**: Final results
   - Breakout: object interaction, ball/brick color sensitivity
   - Pong: robustness on simpler scenes
   - Seaquest: exploration + clutter, background shift brittleness

### CLIP-PPO Technical Design
The CLIP integration follows this procedure:
1. **Expose PPO latent**: Extract hidden state from PPO network
2. **CLIP embedding generation**: Load CLIP model and generate text embeddings
3. **Alignment loss**: Add InfoNCE loss (L_clip) between PPO latents and CLIP embeddings
4. **Combined objective**: L_total = L_ppo + λ*L_clip

#### InfoNCE Loss Calculation
```python
# L2 Normalize
z = torch.nn.functional.normalize(z, dim=-1)  # PPO latent
c = torch.nn.functional.normalize(c, dim=-1)  # CLIP embedding

# Compute similarity scores
logit_scale = torch.clamp(model.logit_scale.exp(), max=100)
logits_z2c = logit_scale * (z @ c.T)
logits_c2z = logit_scale * (c @ z.T)
targets = torch.arange(z.size(0), device=z.device)

# Cross entropy
loss_z = torch.nn.functional.cross_entropy(logits_z2c, targets)
loss_c = torch.nn.functional.cross_entropy(logits_c2z, targets)
L_clip = 0.5 * (loss_z + loss_c)
```

### Planned System Components
- **PPO Agent**: CleanRL-based implementation
- **CLIP-PPO Extension**: Semantic alignment loss integration
- **Disturbance Wrapper**: Visual perturbations for robustness testing
- **Secondary Agents**: PPG (Procgen), Rainbow DQN (Atari) for comparison
- **Evaluation Framework**: Comprehensive ablation studies and metrics

### Disturbance Types (for robustness evaluation)
- **Gaussian Noise**: σ=0.08-0.26 across severity levels
- **Gaussian Blur**: σ=1-3 kernel blur
- **Contrast Jitter**: multiplicative factors 0.6-1.4
- **Cutout/Occlusion**: 10-25% area random patches
- **Sticky Actions**: p=0.25 (Atari-specific)
- **Frame Skip Jitter**: Random frame skipping

### Planned Ablations
- PPO + Frozen CLIP Encoder (no alignment loss)
- PPO + Random Encoder (parameter control)
- CLIP-PPO w/o λ (loss ablation)
- CLIP-PPO with different λ values (0.01, 0.1, 1.0)

### Evaluation Metrics
- Average episodic return
- Relative performance drop (clean vs disturbed)
- Generalization gap (Procgen train vs test)
- Linear probe accuracy on latent features
- Representation similarity analysis
- Learning curves and robustness curves

## Environment Setup

The project uses a Python virtual environment located at `.venv/`. To work with this codebase:

```bash
source .venv/bin/activate  # Activate the virtual environment
```

## Running Experiments

### Basic PPO Training

```bash
# Run standard PPO on MiniGrid
python minigrid_experiments/ppo/ppo_minigrid.py

# Run CLIP-PPO variant with semantic regularization
python minigrid_experiments/clip_ppo/clip_ppo_minigrid.py
```

### Key Parameters

Both scripts use `tyro` for command-line arguments. Key parameters include:

**General Parameters:**
- `--env_id`: MiniGrid environment (default: 'MiniGrid-Empty-16x16-v0')
- `--total_timesteps`: Training duration (default: 500,000)
- `--num_envs`: Parallel environments (default: 8)
- `--learning_rate`: Optimizer learning rate (default: 2.5e-4)
- `--track`: Enable Weights & Biases tracking
- `--capture_video`: Record agent performance videos

**Model Saving Parameters:**
- `--save_model`: Enable model checkpointing (default: True)
- `--save_freq`: Save frequency in timesteps (default: 100,000)
- `--model_path`: Checkpoint directory (default: "checkpoints")

**CLIP-PPO Specific Parameters:**
- `--clip_lambda`: CLIP alignment loss coefficient (default: 0.1)
- `--clip_model`: CLIP model variant (default: "ViT-B/32")

### Usage Examples

```bash
# Standard PPO with custom parameters
python minigrid_experiments/ppo/ppo_minigrid.py --env_id MiniGrid-DoorKey-6x6-v0 --total_timesteps 1000000 --save_freq 50000

# CLIP-PPO with different lambda values
python minigrid_experiments/clip_ppo/clip_ppo_minigrid.py --clip_lambda 0.01 --total_timesteps 1000000

# Disable CLIP alignment (pure PPO)
python minigrid_experiments/clip_ppo/clip_ppo_minigrid.py --clip_lambda 0.0

# Enable W&B tracking
python minigrid_experiments/clip_ppo/clip_ppo_minigrid.py --track --wandb_project_name clip-ppo-experiments
```

## Architecture

### Agent Architecture
- **Feature Extractor**: CNN with 3 convolutional layers (32→64→64 filters)
- **Policy Head**: Linear layer outputting action logits
- **Value Head**: Linear layer outputting state values
- **Input Processing**: RGB images (84x84) from MiniGrid environments

### Training Loop
1. **Data Collection**: Parallel environment rollouts using vectorized environments
2. **Advantage Estimation**: GAE (Generalized Advantage Estimation) 
3. **Policy Updates**: Multiple epochs of minibatch SGD with PPO clipping
4. **Logging**: TensorBoard metrics and optional Weights & Biases tracking

## Output Directories

- `runs/`: TensorBoard logs organized by run name format: `{env_id}__{exp_name}__{seed}__{timestamp}`
- `videos/minigrid/ppo/`: Standard PPO performance videos
- `videos/minigrid/clip_ppo/`: CLIP-PPO performance videos  
- `checkpoints/`: Model checkpoints (when `save_model=True`)
  - `{run_name}_step_{global_step}.pt`: Periodic checkpoints
  - `{run_name}_latest.pt`: Latest checkpoint (overwritten)
  - `{run_name}_final.pt`: Final checkpoint at training completion

## Model Checkpointing

Both implementations include comprehensive model saving:

### Checkpoint Contents
- **Agent state dict**: All model weights
- **Optimizer state dict**: For resuming training  
- **Training metadata**: iteration, global_step, hyperparameters
- **Performance data**: Returns for analysis
- **Training status**: Whether training completed

### Shared Utility
The implementations use a shared utility (`minigrid_experiments/utils.py`) to avoid code duplication:
```python
utils.save_checkpoint(agent, optimizer, iteration, global_step, args, checkpoint_path, b_returns, final=False)
```

## Dependencies

Core dependencies (based on imports):
- `torch`: Neural network framework
- `clip`: OpenAI CLIP model (for CLIP-PPO)
- `gymnasium`: Environment interface
- `minigrid`: MiniGrid environments and wrappers
- `numpy`: Numerical computations
- `tyro`: Command-line argument parsing
- `tqdm`: Progress bars
- `tensorboard`: Logging and visualization
- `wandb`: Experiment tracking (optional)

## CLIP-PPO Implementation Details

### Semantic Regularization
- **Text Descriptions**: Generated from MiniGrid's internal state (agent position, objects, etc.)
- **CLIP Model**: Frozen ViT-B/32 for semantic stability
- **Alignment Loss**: InfoNCE loss between PPO latents (512-dim) and CLIP text embeddings
- **Integration**: Added to PPO loss as `L_total = L_ppo + λ*L_clip`

### Key Functions
- `get_symbolic_descriptions()`: Extracts MiniGrid state to text descriptions
- `compute_infonce_loss()`: Calculates CLIP alignment loss
- `Agent.get_latent_representation()`: Exposes PPO hidden states

## Code Conventions

- Uses CleanRL coding style and structure
- CNN architecture designed for 84x84 RGB input → 7x7 feature maps  
- Orthogonal weight initialization with custom standard deviations
- Environment wrappers: `ImgObsWrapper`, `ResizeObservation`, `RecordEpisodeStatistics`
- **Module imports**: Uses `sys.path.insert()` for clean imports without PYTHONPATH requirements
- **Shared utilities**: Common functions in `minigrid_experiments/utils.py`