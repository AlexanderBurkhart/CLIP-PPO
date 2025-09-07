"""
Shared checkpoint utilities for PPO and CLIP-PPO implementations
"""
import torch
from typing import Optional
from dataclasses import dataclass


def save_checkpoint(
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    global_step: int,
    args: dataclass,
    checkpoint_path: str,
    b_returns: Optional[torch.Tensor] = None,
    final: bool = False,
    extra_models: Optional[dict] = None
) -> None:
    """Save model checkpoint"""
    checkpoint = {
        'iteration': iteration,
        'global_step': global_step,
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'returns': b_returns.cpu().numpy() if b_returns is not None else None,
        'training_complete': final
    }
    
    # Save extra models if provided
    if extra_models:
        for name, model in extra_models.items():
            if model is not None:
                checkpoint[f'{name}_state_dict'] = model.state_dict()
    
    if final:
        filename = f"{checkpoint_path}_final.pt"
        print(f"Final model saved: {filename}")
    else:
        filename = f"{checkpoint_path}_step_{global_step}.pt"
        print(f"Model saved at step {global_step}")
        # Also save as latest
        torch.save(checkpoint, f"{checkpoint_path}_latest.pt")
    
    torch.save(checkpoint, filename)


def load_checkpoint(
    checkpoint_path: str,
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    extra_models: Optional[dict] = None
) -> tuple:
    """Load model checkpoint and return iteration, global_step"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load with weights_only=False to handle numpy arrays in checkpoint
    # This is safe since we trust our own checkpoint files
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model and optimizer states
    agent.load_state_dict(checkpoint['agent_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load extra models if provided
    if extra_models:
        for name, model in extra_models.items():
            if model is not None and f'{name}_state_dict' in checkpoint:
                model.load_state_dict(checkpoint[f'{name}_state_dict'])
    
    iteration = checkpoint['iteration']
    global_step = checkpoint['global_step']
    training_complete = checkpoint.get('training_complete', False)
    
    print(f"Checkpoint loaded: iteration {iteration}, global_step {global_step}")
    if training_complete:
        print("Warning: This was a final checkpoint - training was marked as complete")
    
    return iteration, global_step