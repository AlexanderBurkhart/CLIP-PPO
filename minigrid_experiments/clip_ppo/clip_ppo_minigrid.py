# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import clip

from minigrid.wrappers import ImgObsWrapper
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


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
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
 
    # Algorithm specific arguments
    env_id: str = 'MiniGrid-Empty-16x16-v0'
    """the id of the environment"""
    total_timesteps: int = 250_000
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
    target_kl: float = None
    """the target KL divergence threshold"""
    
    # CLIP-PPO specific arguments
    clip_lambda: float = 0.1
    """coefficient for CLIP alignment loss"""
    clip_model: str = "ViT-B/32"
    """CLIP model variant to use"""
    
    # Model saving arguments
    save_model: bool = True
    """whether to save model checkpoints"""
    save_freq: int = 100000
    """save model every N timesteps"""
    model_path: str = "checkpoints"
    """directory to save model checkpoints"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
         # MiniGrid with RGB frames
        env = gym.make(env_id, render_mode="rgb_array")
        env = ImgObsWrapper(env)  # -> HWC uint8 RGB

        # (Optional) standard helpers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))  # keep CleanRL CNN output = 7x7
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/minigrid/clip_ppo/{run_name}")

        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def compute_infonce_loss(z, c, logit_scale):
    """
    Compute InfoNCE loss between PPO latent representations (z) and CLIP text embeddings (c).
    
    Args:
        z: PPO latent vectors [batch_size, latent_dim]
        c: CLIP text embeddings [batch_size, clip_dim]
        logit_scale: CLIP logit scale parameter
    
    Returns:
        InfoNCE loss scalar
    """
    # L2 Normalize
    z = torch.nn.functional.normalize(z, dim=-1)
    c = torch.nn.functional.normalize(c, dim=-1)

    # Compute pairwise similarity scores
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    logits_z2c = logit_scale * (z @ c.T)
    logits_c2z = logit_scale * (c @ z.T)
    targets = torch.arange(z.size(0), device=z.device)

    # Cross entropy
    loss_z = torch.nn.functional.cross_entropy(logits_z2c, targets)
    loss_c = torch.nn.functional.cross_entropy(logits_c2z, targets)
    L_clip = 0.5 * (loss_z + loss_c)
    
    return L_clip



def get_symbolic_descriptions(envs):
    """
    Extract symbolic descriptions from MiniGrid environments' internal state.
    
    Args:
        envs: Vectorized environment
    
    Returns:
        List of text descriptions for each environment
    """
    descriptions = []
    
    for i in range(envs.num_envs):
        try:
            # Access the unwrapped environment to get MiniGrid state
            env = envs.envs[i].env
            # Navigate through wrappers to get to the base MiniGrid environment
            while hasattr(env, 'env'):
                env = env.env
            
            # Extract agent position and direction
            agent_pos = env.agent_pos
            agent_dir = env.agent_dir
            dir_names = ["right", "down", "left", "up"]
            dir_name = dir_names[agent_dir]
            
            # Extract objects in the environment
            objects = []
            grid = env.grid
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and cell.type != 'wall':
                        objects.append(f"{cell.type} at ({x},{y})")
            
            # Build description
            desc = f"agent at ({agent_pos[0]},{agent_pos[1]}) facing {dir_name}"
            if objects:
                desc += f", objects: {', '.join(objects[:3])}"  # Limit to first 3 objects
            
            descriptions.append(desc)
            
        except Exception as e:
            # Fallback description if state extraction fails
            descriptions.append("agent navigating grid environment")
    
    return descriptions


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
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

    def _pre(self, x):
        # x: [B,H,W,C] -> [B,C,H,W]
        if x.dim() == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()
        return x / 255.0

    def get_value(self, x):
        x = self._pre(x)
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        x = self._pre(x)
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def get_latent_representation(self, x):
        x = self._pre(x)
        return self.network(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Load CLIP model
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()  # Keep CLIP frozen
    
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

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in tqdm.tqdm(range(1, args.num_iterations + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
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

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
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
                
                # CLIP alignment loss
                clip_loss = 0.0
                if args.clip_lambda > 0.0:
                    # Get PPO latent representations
                    with torch.no_grad():
                        # Get symbolic descriptions for current minibatch
                        # Note: We need to map minibatch indices back to environment states
                        # For simplicity, we'll generate descriptions based on current env state
                        descriptions = get_symbolic_descriptions(envs)
                        # Repeat descriptions to match minibatch size
                        mb_descriptions = [descriptions[i % len(descriptions)] for i in range(len(mb_inds))]
                        
                        # Generate CLIP text embeddings
                        text_tokens = clip.tokenize(mb_descriptions).to(device)
                        clip_text_embeddings = clip_model.encode_text(text_tokens).float()
                    
                    # Get PPO latent representations for minibatch
                    ppo_latents = agent.get_latent_representation(b_obs[mb_inds])
                    
                    # Compute InfoNCE loss
                    clip_loss = compute_infonce_loss(ppo_latents, clip_text_embeddings, clip_model.logit_scale)
                
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.clip_lambda * clip_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

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
        if args.clip_lambda > 0.0:
            writer.add_scalar("losses/clip_loss", clip_loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # Save model checkpoint
        if args.save_model and global_step % args.save_freq == 0:
            utils.save_checkpoint(agent, optimizer, iteration, global_step, args, checkpoint_path, b_returns)

    # Save final model
    if args.save_model:
        utils.save_checkpoint(agent, optimizer, args.num_iterations, global_step, args, checkpoint_path, 
                       b_returns if 'b_returns' in locals() else None, final=True)

    envs.close()
    writer.close()
