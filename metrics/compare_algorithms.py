"""
Compare multiple algorithms across clean and disturbed environments.
"""
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import numpy as np
import tyro
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'minigrid_experiments'))
from disturbances import DisturbanceSeverity
from calculate_metrics import get_disturbance_severity, compute_robustness_index, compute_robustness_index_over_time, load_tensorboard_run, compute_final_success_rate


@dataclass
class AlgorithmConfig:
    """Configuration for a single algorithm."""
    name: str
    """Algorithm name for display"""
    clean_run_path: str
    """Path to clean (undisturbed) TensorBoard run directory"""
    disturbed_run_paths: List[str]
    """List of disturbed run paths for this algorithm"""


@dataclass 
class ComparisonConfig:
    """Configuration for multi-algorithm comparison."""
    # AlgorithmConfig('PPO', 'runs/old/ppo_clean', ['runs/old/ppo_hard']),
    # AlgorithmConfig('CLIP_PPO', 'runs/old/clip_ppo_clean', ['runs/old/clip_ppo_hard']),
    algorithms: tuple = (
        AlgorithmConfig('PPO', 'runs/ppo_clean', ['runs/ppo_hard']),
        # AlgorithmConfig('CLIP_PPO', 'runs/clip_ppo_clean', ['runs/clip_ppo_hard']),
        # AlgorithmConfig('FROZEN', 'runs/clip_ppo_clean', ['runs/clip_ppo_frozen_clip_hard']),
        # AlgorithmConfig('RANDOM', 'runs/clip_ppo_clean', ['runs/clip_ppo_random_encoder_hard'])
    )
    """List of algorithm configurations"""


def plot_ri_comparison_across_algorithms(algorithms: List[AlgorithmConfig]):
    """
    Plot robustness index comparison across algorithms for each disturbance level.
    
    Args:
        algorithms: List of algorithm configurations
    """
    # Collect all severity levels from all algorithms
    all_severities = set()
    for alg in algorithms:
        for disturbed_path in alg.disturbed_run_paths:
            severity = get_disturbance_severity(disturbed_path)
            all_severities.add(severity.value.upper())
    
    severities = sorted(list(all_severities))
    
    # Create subplot for each severity level
    fig, axes = plt.subplots(1, len(severities), figsize=(5*len(severities), 6))
    if len(severities) == 1:
        axes = [axes]
    
    # Use matplotlib color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    for sev_idx, severity_name in enumerate(severities):
        ax = axes[sev_idx]
        algorithm_names = []
        robustness_indices = []
        
        for alg_idx, alg in enumerate(algorithms):
            # Find the run with this severity level
            for disturbed_path in alg.disturbed_run_paths:
                run_severity = get_disturbance_severity(disturbed_path)
                if run_severity.value.upper() == severity_name:
                    ri = compute_robustness_index(alg.clean_run_path, disturbed_path)
                    algorithm_names.append(alg.name)
                    robustness_indices.append(ri)
                    break
        
        # Create bar plot for this severity level
        bars = ax.bar(algorithm_names, robustness_indices, 
                     color=[colors[i % len(colors)] for i in range(len(algorithm_names))],
                     alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, ri in zip(bars, robustness_indices):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(robustness_indices)*0.01,
                   f'{ri:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add perfect robustness reference line
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Robustness (1.0)')
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Robustness Index')
        ax.set_title(f'{severity_name} Disturbance')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show(block=True)


def plot_learning_curves_comparison(algorithms: List[AlgorithmConfig], window_size: int = 50):
    """
    Plot learning curves comparison across algorithms (disturbed environments).
    
    Args:
        algorithms: List of algorithm configurations
        window_size: Size of rolling window for smoothing
    """
    plt.figure(figsize=(15, 10))
    
    # Use matplotlib color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    for alg_idx, alg in enumerate(algorithms):
        color = colors[alg_idx % len(colors)]
        
        # Plot clean environment first
        timesteps, returns, _ = load_tensorboard_run(alg.clean_run_path)
        import pandas as pd
        df = pd.DataFrame({'timestep': timesteps, 'return': returns})
        df['rolling_mean'] = df['return'].rolling(window=window_size, min_periods=1).mean()
        
        plt.plot(df['timestep'], df['rolling_mean'], 
                label=f'{alg.name} Clean (Final: {df["rolling_mean"].iloc[-1]:.3f})', 
                color=color, linewidth=2, alpha=0.8, linestyle='-')
        
        # Add raw data as faint background
        plt.plot(df['timestep'], df['return'], color=color, alpha=0.05, linewidth=0.5, linestyle='-')
        
        # Plot disturbed environments
        for run_idx, disturbed_run_path in enumerate(alg.disturbed_run_paths):
            # Get severity from disturbed run
            severity = get_disturbance_severity(disturbed_run_path)
            severity_name = severity.value.capitalize() if severity.value != 'none' else 'Clean'
            
            timesteps, returns, _ = load_tensorboard_run(disturbed_run_path)
            
            # Create DataFrame and compute rolling mean
            df = pd.DataFrame({'timestep': timesteps, 'return': returns})
            df['rolling_mean'] = df['return'].rolling(window=window_size, min_periods=1).mean()
            
            # Use different line styles for different disturbance levels (skip solid line)
            linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1))]  # Skip solid line
            linestyle = linestyles[run_idx % len(linestyles)]
            
            plt.plot(df['timestep'], df['rolling_mean'], 
                    label=f'{alg.name} {severity_name} (Final: {df["rolling_mean"].iloc[-1]:.3f})', 
                    color=color, linewidth=2, alpha=0.8, linestyle=linestyle)
            
            # Add raw data as faint background
            plt.plot(df['timestep'], df['return'], color=color, alpha=0.05, linewidth=0.5, linestyle=linestyle)
    
    plt.xlabel('Training Timesteps')
    plt.ylabel('Episodic Return')
    plt.title('Learning Curves Comparison (Clean & Disturbed Environments)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)


def plot_success_rate_comparison(algorithms: List[AlgorithmConfig]):
    """
    Plot success rate comparison across algorithms for all disturbance levels.
    
    Args:
        algorithms: List of algorithm configurations
    """
    # Collect all unique disturbance severities
    all_severities = set()
    for alg in algorithms:
        for disturbed_path in alg.disturbed_run_paths:
            severity = get_disturbance_severity(disturbed_path)
            all_severities.add(severity.value.upper())
    
    severities = sorted(list(all_severities))
    
    # Create subplots for each severity level + clean
    fig, axes = plt.subplots(1, len(severities) + 1, figsize=(6*(len(severities) + 1), 6))
    if len(severities) + 1 == 1:  # Only one subplot total
        axes = [axes]
    elif not isinstance(axes, np.ndarray):  # Handle single subplot case
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    # Plot clean environment success rates
    ax_clean = axes[0]
    algorithm_names = [alg.name for alg in algorithms]
    clean_success_rates = [compute_final_success_rate(alg.clean_run_path) for alg in algorithms]
    
    bars = ax_clean.bar(algorithm_names, clean_success_rates,
                       color=[colors[i % len(colors)] for i in range(len(algorithms))],
                       alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, clean_success_rates):
        ax_clean.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax_clean.set_xlabel('Algorithm')
    ax_clean.set_ylabel('Success Rate (%)')
    ax_clean.set_title('Clean Environment')
    ax_clean.set_ylim(0, 105)
    ax_clean.grid(True, alpha=0.3, axis='y')
    
    # Plot each disturbance severity level
    for sev_idx, severity_name in enumerate(severities):
        ax = axes[sev_idx + 1]
        success_rates = []
        
        for alg in algorithms:
            # Find the run with this severity level
            found_success = 0.0
            for disturbed_path in alg.disturbed_run_paths:
                run_severity = get_disturbance_severity(disturbed_path)
                if run_severity.value.upper() == severity_name:
                    found_success = compute_final_success_rate(disturbed_path)
                    break
            success_rates.append(found_success)
        
        bars = ax.bar(algorithm_names, success_rates,
                     color=[colors[i % len(colors)] for i in range(len(algorithms))],
                     alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'{severity_name} Disturbance')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Success Rate Comparison Across Algorithms', fontsize=16)
    plt.tight_layout()
    plt.show(block=True)
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUCCESS RATE COMPARISON SUMMARY")
    print(f"{'='*60}")
    header = f"{'Algorithm':<15} | {'Clean':<8}"
    for severity in severities:
        header += f" | {severity:<8}"
    print(header)
    print("-" * len(header))
    
    for alg_idx, alg in enumerate(algorithms):
        row = f"{alg.name:<15} | {clean_success_rates[alg_idx]:6.1f}%"
        
        for severity_name in severities:
            success_rate = 0.0
            for disturbed_path in alg.disturbed_run_paths:
                run_severity = get_disturbance_severity(disturbed_path)
                if run_severity.value.upper() == severity_name:
                    success_rate = compute_final_success_rate(disturbed_path)
                    break
            row += f" | {success_rate:6.1f}%"
        
        print(row)


def plot_robustness_curves_comparison(algorithms: List[AlgorithmConfig], disturbance_level: str = None, all_levels: bool = False):
    """
    Plot robustness index over time comparison across algorithms.
    
    Args:
        algorithms: List of algorithm configurations
        disturbance_level: Specific disturbance level to compare (e.g., 'HARD')
        all_levels: If True, create subplots for all disturbance levels
    """
    if disturbance_level and not all_levels:
        # Single disturbance level comparison
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        
        for i, alg in enumerate(algorithms):
            # Find the run with specified disturbance level
            for disturbed_path in alg.disturbed_run_paths:
                run_severity = get_disturbance_severity(disturbed_path)
                if run_severity.value.upper() == disturbance_level.upper():
                    df = compute_robustness_index_over_time(alg.clean_run_path, disturbed_path)
                    
                    color = colors[i % len(colors)]
                    final_ri = df['robustness_index'].iloc[-1]
                    plt.plot(df['timestep'], df['robustness_index'],
                            color=color, linewidth=2, alpha=0.8,
                            label=f'{alg.name} (Final: {final_ri:.3f})')
                    break
        
        # Add perfect robustness reference line
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Robustness (1.0)')
        
        plt.xlabel('Training Timesteps')
        plt.ylabel('Robustness Index')
        plt.title(f'Robustness Index Over Time ({disturbance_level} Disturbance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)
    
    elif all_levels:
        # All disturbance levels on single plot
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        linestyles = ['-', '--', '-.', ':', '-', '--']
        
        # Get all severity levels
        all_severities = set()
        for alg in algorithms:
            for disturbed_path in alg.disturbed_run_paths:
                severity = get_disturbance_severity(disturbed_path)
                all_severities.add(severity.value.upper())
        
        severities = sorted(list(all_severities))
        
        for alg_idx, alg in enumerate(algorithms):
            color = colors[alg_idx % len(colors)]
            
            for sev_idx, severity_name in enumerate(severities):
                # Find the run with this severity level for this algorithm
                for disturbed_path in alg.disturbed_run_paths:
                    run_severity = get_disturbance_severity(disturbed_path)
                    if run_severity.value.upper() == severity_name:
                        df = compute_robustness_index_over_time(alg.clean_run_path, disturbed_path)
                        
                        linestyle = linestyles[sev_idx % len(linestyles)]
                        final_ri = df['robustness_index'].iloc[-1]
                        plt.plot(df['timestep'], df['robustness_index'],
                                color=color, linestyle=linestyle, linewidth=2, alpha=0.8,
                                label=f'{alg.name} {severity_name} (Final: {final_ri:.3f})')
                        break
        
        # Add perfect robustness reference line
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Robustness (1.0)')
        
        plt.xlabel('Training Timesteps')
        plt.ylabel('Robustness Index')
        plt.title('Robustness Index Over Time (All Algorithms & Disturbance Levels)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)
    
    else:
        print("Please specify either disturbance_level or set all_levels=True")


if __name__ == "__main__":
    # Example usage - you can modify this or use tyro.cli for more complex configs
    config = tyro.cli(ComparisonConfig)

    # Generate all comparison plots
    # plot_ri_comparison_across_algorithms(config.algorithms)
    # plot_learning_curves_comparison(config.algorithms)
    plot_success_rate_comparison(config.algorithms)
    # plot_robustness_curves_comparison(config.algorithms, all_levels=True)  # All disturbance levels