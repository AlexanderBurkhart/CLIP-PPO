"""
Compare multiple algorithms across clean and disturbed environments.
"""
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List
import tyro
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'minigrid_experiments'))
from disturbances import DisturbanceSeverity
from calculate_metrics import get_disturbance_severity, compute_robustness_index


@dataclass
class RobustnessConfig:
    """Configuration for robustness analysis across disturbance levels."""
    algorithm_name: str = "PPO"
    """Algorithm name for the plot title"""
    clean_run_path: str = "runs/short_clip_ppo_clean"
    """Path to clean (undisturbed) TensorBoard run directory"""
    disturbance_runs: tuple[str] = ('runs/short_clip_ppo_hard', 'runs/short_clip_ppo_severe')
    """List of disturbed run paths"""


def plot_ri_across_disturbances(config: RobustnessConfig):
    """
    Plot bar chart of robustness index across disturbance severities.
    
    Args:
        config: Configuration with algorithm name and disturbance runs
    """
    severities = []
    robustness_indices = []
    
    for disturbed_run_path in config.disturbance_runs:
        # Get severity from disturbed run
        severity = get_disturbance_severity(disturbed_run_path)
        severity_name = severity.value.upper()
        
        ri = compute_robustness_index(config.clean_run_path, disturbed_run_path)
        severities.append(severity_name)
        robustness_indices.append(ri)
        
        print(f"{severity_name}: RI = {ri:.3f}")
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    colors = ['lightgreen', 'yellow', 'orange', 'red']  # Mild to severe
    bars = plt.bar(severities, robustness_indices, color=colors[:len(severities)], 
                   alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, ri in zip(bars, robustness_indices):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(robustness_indices)*0.01,
                f'{ri:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add perfect robustness reference line
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Robustness (1.0)')
    
    plt.xlabel('Disturbance Severity')
    plt.ylabel('Robustness Index')
    plt.title(f'{config.algorithm_name}: Robustness Index Across Disturbance Levels')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show(block=True)
    
    return severities, robustness_indices


def plot_robustness_curves_over_time(config: RobustnessConfig):
    """
    Plot robustness index over time with separate lines for each disturbance level.
    
    Args:
        config: Configuration with algorithm name and disturbance runs
    """
    from calculate_metrics import compute_robustness_index_over_time, load_tensorboard_run
    
    plt.figure(figsize=(12, 8))
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    
    # Add perfect robustness reference line
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Robustness (1.0)')
    
    # Plot each disturbed run
    for i, disturbed_run_path in enumerate(config.disturbance_runs):
        # Get severity name
        severity = get_disturbance_severity(disturbed_run_path)
        severity_name = severity.value.capitalize()
        
        # Compute RI over time
        df = compute_robustness_index_over_time(config.clean_run_path, disturbed_run_path)
        
        # Plot line
        color = colors[i % len(colors)]
        final_ri = df['robustness_index'].iloc[-1]
        plt.plot(df['timestep'], df['robustness_index'], 
                color=color, linewidth=2, alpha=0.8,
                label=f'{severity_name} (Final: {final_ri:.3f})')
        
        print(f"{severity_name}: Final RI = {final_ri:.3f}")
    
    plt.xlabel('Training Timesteps')
    plt.ylabel('Robustness Index')
    plt.title(f'{config.algorithm_name}: Robustness Index Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    config = tyro.cli(RobustnessConfig)
    names, ris = plot_ri_across_disturbances(config)
    plot_robustness_curves_over_time(config)