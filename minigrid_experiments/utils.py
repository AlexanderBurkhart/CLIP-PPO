"""
Shared utilities for PPO and CLIP-PPO implementations
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
    final: bool = False
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
    
    if final:
        filename = f"{checkpoint_path}_final.pt"
        print(f"Final model saved: {filename}")
    else:
        filename = f"{checkpoint_path}_step_{global_step}.pt"
        print(f"Model saved at step {global_step}")
        # Also save as latest
        torch.save(checkpoint, f"{checkpoint_path}_latest.pt")
    
    torch.save(checkpoint, filename)