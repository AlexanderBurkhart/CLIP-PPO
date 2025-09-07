# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
import sys
import tqdm
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import ale_py

# Import shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
import clip_ppo_utils
import checkpoint_utils
from shared.disturbances_gpu import DisturbanceWrapperGPU
from shared.disturbance_types import DisturbanceSeverity

# Import atari wrappers from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# Register ALE environments
gym.register_envs(ale_py)


@dataclass
class AtariClipPPOConfig(clip_ppo_utils.ClipPPOConfig):
    """Atari-specific CLIP-PPO configuration with environment-specific defaults."""
    
    # Atari-specific CLIP defaults
    clip_lambda: float = 0.00001
    """coefficient for CLIP alignment loss"""
    clip_modality: str = "text"
    """CLIP modality to use for alignment (image better for Atari visual scenes)"""
    ablation_mode: clip_ppo_utils.AblationMode = clip_ppo_utils.AblationMode.NONE
    """ablation mode for controlled experiments"""
    
    # Visual disturbance parameters for Atari
    apply_disturbances: bool = False
    """whether to apply visual disturbances during training"""
    disturbance_severity: str = "MODERATE"
    """disturbance severity level: MILD, MODERATE, HARD, SEVERE"""


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    verbose: bool = True
    """enable verbose debug output for losses"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    
    # CLIP-PPO specific config
    clip_config: AtariClipPPOConfig = field(default_factory=AtariClipPPOConfig)
    """Atari-specific CLIP-PPO configuration"""
    
    # Run configuration
    run_name: Optional[str] = None
    """custom run name for experiment tracking (auto-generated if None)"""
    
    # Model saving arguments (Atari-specific)
    save_model: bool = True
    """whether to save model checkpoints"""
    save_freq: int = 100000
    """save model every N timesteps"""
    model_path: str = "checkpoints"
    """directory to save model checkpoints"""
    resume_checkpoint: str = ''
    """path to checkpoint file to resume training from"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/atari/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, ablation_mode=None, clip_model=None):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.clip_model = clip_model
        
        if ablation_mode == clip_ppo_utils.AblationMode.FROZEN_CLIP:
            # Use frozen CLIP visual encoder instead of learned CNN
            # CLIP outputs 512-dim features, so we can directly use them
            self.network = nn.Sequential(
                nn.Identity()  # Placeholder - features come from CLIP
            )
        else:
            # Standard CNN architecture
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )
        
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        
    def _get_features(self, x):
        """Get features from either CNN or frozen CLIP."""
        if self.ablation_mode == clip_ppo_utils.AblationMode.FROZEN_CLIP:
            # Convert frame stack to RGB format for CLIP
            rgb_frames = convert_atari_frames_for_clip(x)  # [batch, 3, 84, 84]
            rgb_frames = rgb_frames.float() / 255.0  # [batch, 3, 84, 84]
            return clip_ppo_utils.get_frozen_clip_features(rgb_frames, self.clip_model)
        else:
            return self.network(x / 255.0)

    def get_value(self, x):
        hidden = self._get_features(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self._get_features(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def get_latent_representation(self, x):
        """Get latent representation for CLIP alignment."""
        return self._get_features(x).detach()


def convert_atari_frames_for_clip(obs_batch):
    """
    Convert Atari frame stack observations to format suitable for CLIP.
    
    Args:
        obs_batch: [batch_size, 4, 84, 84] grayscale frame stack
        
    Returns:
        RGB images [batch_size, 3, 84, 84] suitable for CLIP preprocessing
    """
    # Take the most recent frame from the stack (last channel)
    recent_frame = obs_batch[:, -1, :, :].unsqueeze(1)  # [batch, 1, 84, 84]
    
    # Convert grayscale to RGB by repeating across channels
    rgb_frames = recent_frame.repeat(1, 3, 1, 1)  # [batch, 3, 84, 84]
    
    return rgb_frames


def generate_breakout_descriptions(envs, batch_size: int) -> list:
    """
    Generate descriptive text for Breakout using per-environment RAM state analysis.
    
    Args:
        envs: Vectorized Atari environments 
        batch_size: Number of descriptions to generate (should match num_envs for per-env states)
        
    Returns:
        List of text descriptions, one for each environment's current game state
    """
    result = []
    
    try:
        # Get all unwrapped environments
        all_envs_unwrapped = envs.get_attr('unwrapped')
        num_envs = len(all_envs_unwrapped)
        
        for i in range(batch_size):
            # Use modulo to cycle through environments if batch_size != num_envs
            env_idx = i % num_envs
            env_unwrapped = all_envs_unwrapped[env_idx]
            
            try:
                # Extract Breakout game state from RAM for this specific environment
                ram = env_unwrapped.ale.getRAM()
                
                # Breakout RAM addresses (well-documented)
                score = ram[36] * 100 + ram[37] * 10 + ram[38]  # BCD encoded score
                ball_x = ram[99]  # Ball X position
                ball_y = ram[101]  # Ball Y position  
                paddle_x = ram[72]  # Paddle X position
                lives = ram[57] & 0x7  # Lives remaining
                
                # Generate concise description starting with base info
                ball_paddle_distance = abs(ball_x - paddle_x)
                
                # Start with base description
                description = f"Breakout score {score}, ball ({ball_x},{ball_y}), paddle ({paddle_x}), lives {lives}"
                
                # Add state-specific context
                if ball_paddle_distance < 15 and ball_y > 180:
                    # Critical moment
                    description += ", near paddle danger zone"
                elif ball_paddle_distance < 15:
                    # Ball near paddle, safe
                    description += ", near paddle safe"
                elif ball_y > 180:
                    # Ball in danger zone, away from paddle
                    description += ", ball danger zone"
                else:
                    # Ball hitting bricks above
                    description += ", ball hitting bricks"
                
                result.append(description)
                
            except:
                # Fallback for individual environment if RAM extraction fails
                result.append(f"Breakout: paddle and ball gameplay in progress")
                
    except:
        # Complete fallback if environment access fails
        for i in range(batch_size):
            result.append(f"Breakout: classic arcade brick breaking game")
    
    return result


def generate_pong_descriptions(envs, batch_size: int) -> list:
    """
    Generate descriptive text for Pong using per-environment RAM state analysis.
    
    Args:
        envs: Vectorized Atari environments 
        batch_size: Number of descriptions to generate (should match num_envs for per-env states)
        
    Returns:
        List of text descriptions, one for each environment's current game state
    """
    result = []
    
    try:
        # Get all unwrapped environments
        all_envs_unwrapped = envs.get_attr('unwrapped')
        num_envs = len(all_envs_unwrapped)
        
        for i in range(batch_size):
            # Use modulo to cycle through environments if batch_size != num_envs
            env_idx = i % num_envs
            env_unwrapped = all_envs_unwrapped[env_idx]
            
            try:
                # Extract Pong game state from RAM for this specific environment
                ram = env_unwrapped.ale.getRAM()
                
                # Pong RAM addresses
                player_score = ram[13]  # Player score (right paddle)
                computer_score = ram[14]  # Computer score (left paddle) 
                ball_x = ram[49]  # Ball X position
                ball_y = ram[54]  # Ball Y position
                player_paddle_y = ram[51]  # Player paddle Y position
                computer_paddle_y = ram[50]  # Computer paddle Y position
                
                # Calculate relative positions and distances
                ball_player_distance = abs(ball_y - player_paddle_y)
                ball_computer_distance = abs(ball_y - computer_paddle_y)
                
                # Start with base description
                description = f"Pong score {player_score}-{computer_score}, ball ({ball_x},{ball_y}), player paddle ({player_paddle_y}), computer paddle ({computer_paddle_y})"
                
                # Add contextual state information
                if ball_x > 140:  # Ball near player side (right)
                    if ball_player_distance < 10:
                        description += ", near player paddle"
                    else:
                        description += ", ball approaching player"
                elif ball_x < 20:  # Ball near computer side (left)
                    if ball_computer_distance < 10:
                        description += ", near computer paddle"
                    else:
                        description += ", ball approaching computer"
                else:  # Ball in center area
                    if ball_x > 80:
                        description += ", ball moving toward player"
                    else:
                        description += ", ball moving toward computer"
                
                result.append(description)
                
            except:
                # Fallback for individual environment if RAM extraction fails
                print("WARNING: environment access failed for generating description")
                result.append(f"Pong: player vs computer paddle tennis match")
                
    except:
        # Complete fallback if environment access fails
        print("WARNING: environment access failed for generating description")
        for i in range(batch_size):
            result.append(f"Pong: classic paddle tennis gameplay")
    
    return result


def generate_atari_descriptions(envs, batch_size: int, env_id: str) -> list:
    """
    Generate descriptive text for Atari games using per-environment RAM state analysis.
    
    Args:
        envs: Vectorized Atari environments 
        batch_size: Number of descriptions to generate (should match num_envs for per-env states)
        env_id: Environment ID to determine which game-specific function to use
        
    Returns:
        List of text descriptions, one for each environment's current game state
    """
    if "Breakout" in env_id:
        return generate_breakout_descriptions(envs, batch_size)
    elif "Pong" in env_id:
        return generate_pong_descriptions(envs, batch_size)
    else:
        raise ValueError(f"{env_id} not supported for CLIP PPO")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = args.run_name if args.run_name else f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Log disturbance severity for metrics analysis
    if args.clip_config.apply_disturbances:
        writer.add_text("config/disturbance_severity", args.clip_config.disturbance_severity)
    else:
        writer.add_text("config/disturbance_severity", "CLEAN")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize disturbance wrapper if enabled
    disturber = None
    if args.clip_config.apply_disturbances:
        severity = getattr(DisturbanceSeverity, args.clip_config.disturbance_severity)
        disturber = DisturbanceWrapperGPU(device=device, severity=severity)
        print(f"Disturbances enabled with severity: {args.clip_config.disturbance_severity}")
    else:
        print("Disturbances disabled")

    # CLIP-PPO: Load CLIP model if needed (before creating agent)
    clip_model = None
    if (
        clip_ppo_utils.should_compute_clip_loss(args.clip_config.ablation_mode, args.clip_config.clip_lambda) or
        args.clip_config.ablation_mode == clip_ppo_utils.AblationMode.FROZEN_CLIP
    ):
        clip_model = clip_ppo_utils.load_clip_model(args.clip_config.clip_model, device)

    agent = Agent(envs, args.clip_config.ablation_mode, clip_model).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Create checkpoint directory
    if args.save_model:
        os.makedirs(args.model_path, exist_ok=True)
        checkpoint_path = os.path.join(args.model_path, run_name)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Load checkpoint if specified
    start_iteration = 1
    global_step = 0
    if args.resume_checkpoint:
        start_iteration, global_step = checkpoint_utils.load_checkpoint(
            args.resume_checkpoint, agent, optimizer, device
        )
        start_iteration += 1  # Start from next iteration

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in tqdm.tqdm(range(start_iteration, args.num_iterations + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            
            # Apply disturbances if enabled (GPU batch processing)
            if disturber:
                # Normalize to [0,1] and apply disturbances (already in [B, C, H, W] format)
                next_obs_float = next_obs.float() / 255.0
                disturbed_obs = disturber.apply_disturbances(next_obs_float)
                # Denormalize back to uint8
                next_obs = (disturbed_obs * 255.0).byte()
            
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Handle episode completion logging (Atari format)
            if "episode" in infos:
                episode_info = infos["episode"]
                if episode_info is not None and "_r" in episode_info:
                    completed_episodes = episode_info["_r"]  # Boolean array
                    returns = episode_info["r"]  # Return values
                    lengths = episode_info["l"]  # Episode lengths
                    
                    for env_idx, completed in enumerate(completed_episodes):
                        if not completed:
                            continue
                        episodic_return = float(returns[env_idx])
                        episode_length = int(lengths[env_idx])
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", episode_length, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # CLIP-PPO: Pre-compute CLIP embeddings once per iteration for efficiency
        clip_embeddings = None
        if clip_ppo_utils.should_compute_clip_loss(args.clip_config.ablation_mode, args.clip_config.clip_lambda):
            if args.clip_config.clip_modality == "text":
                # Generate dynamic descriptions based on game metadata
                descriptions = generate_atari_descriptions(envs, b_obs.shape[0], args.env_id)
                clip_embeddings = clip_ppo_utils.generate_clip_embeddings(
                    ablation_mode=args.clip_config.ablation_mode,
                    clip_model=clip_model,
                    modality=args.clip_config.clip_modality,
                    batch_size=b_obs.shape[0],
                    device=device,
                    descriptions=descriptions
                )
            
            elif args.clip_config.clip_modality == "image":
                # Convert Atari frames to RGB format for CLIP
                rgb_images = convert_atari_frames_for_clip(b_obs)
                clip_embeddings = clip_ppo_utils.generate_clip_embeddings(
                    args.clip_config.ablation_mode,
                    clip_model,
                    modality=args.clip_config.clip_modality,
                    batch_size=b_obs.shape[0],
                    device=device,
                    images=rgb_images
                )
            else:
                raise ValueError(f"Invalid CLIP modality: {args.clip_config.clip_modality}. Must be 'image' or 'text'")

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        minibatch_counter = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                
                # CLIP-PPO: Compute alignment loss (applied every _SKIP_LENGTH_CLIP_LOSS minibatch)
                clip_loss = torch.tensor(0.0, device=device)
                if (clip_ppo_utils.should_compute_clip_loss(args.clip_config.ablation_mode, args.clip_config.clip_lambda) 
                    and minibatch_counter % clip_ppo_utils.CLIP_LOSS_FREQUENCY == 0):
                    # Get PPO latent representations
                    ppo_latents = agent.get_latent_representation(b_obs[mb_inds])
                    mb_clip_embeddings = clip_embeddings[mb_inds]
                    
                    # Compute cosine embedding loss
                    clip_loss = clip_ppo_utils.compute_cosine_embedding_loss(ppo_latents, mb_clip_embeddings)

                    # Debug: Print key metrics for first epoch if verbose enabled
                    if args.verbose and start == 0 and epoch == 0:
                        print(f"Iter {iteration}")
                        print(f"Weighted CLIP loss: {args.clip_config.clip_lambda * clip_loss.item():>15.10f}")
                        print(f"PPO loss:           {pg_loss.item():>15.10f}")
                        print(f"Combined loss: {(pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.clip_config.clip_lambda * clip_loss).item():.6f}")
                        print("---")

                # Apply CLIP lambda warmup
                current_clip_lambda = clip_ppo_utils.get_clip_lambda_with_warmup(
                    args.clip_config.clip_lambda, iteration - 1, args.num_iterations
                )

                # Total loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + current_clip_lambda * clip_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                
                minibatch_counter += 1

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/clip_loss", clip_loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # Save model checkpoint
        if args.save_model and global_step % args.save_freq == 0:
            checkpoint_utils.save_checkpoint(agent, optimizer, iteration, global_step, args, checkpoint_path, b_returns)

    # Save final model
    if args.save_model:
        checkpoint_utils.save_checkpoint(agent, optimizer, args.num_iterations, global_step, args, checkpoint_path, 
                       b_returns if 'b_returns' in locals() else None, final=True)

    envs.close()
    writer.close()
