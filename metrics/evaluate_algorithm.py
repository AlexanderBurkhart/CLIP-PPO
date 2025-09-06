"""
TensorBoard log analysis for an algorithm run.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from dataclasses import dataclass
import tyro
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from calculate_metrics import compute_robustness_index_over_time


@dataclass
class EvaluationConfig:
    """Configuration for robustness evaluation."""
    clean_run_path: str = 'runs/short_clip_ppo_clean'
    """Path to clean (undisturbed) TensorBoard run directory"""
    disturbed_run_path: str = 'runs/short_clip_ppo_hard'
    """Path to disturbed TensorBoard run directory"""


def load_tensorboard_run(run_path: str) -> Tuple[List[int], List[float], List[float]]:
    """
    Load episodic returns and lengths from TensorBoard logs.
    
    Args:
        run_path: Path to TensorBoard run directory
        
    Returns:
        Tuple of (timesteps, returns, episode_lengths)
    """
    ea = EventAccumulator(run_path)
    ea.Reload()
    
    # Debug: Print available scalar tags
    print(f"Available scalar tags in {run_path}:")
    for tag in ea.Tags()['scalars']:
        print(f"  - {tag}")
    
    # Try to find episodic return data
    possible_return_tags = ['charts/episodic_return', 'episodic_return', 'charts/returns']
    returns_data = None
    
    for tag in possible_return_tags:
        if tag in ea.Tags()['scalars']:
            returns_data = ea.Scalars(tag)
            print(f"Found returns data in tag: {tag}")
            break
    
    if returns_data is None:
        raise ValueError(f"Could not find episodic return data in any expected tags: {possible_return_tags}")
    
    timesteps = [x.step for x in returns_data]
    returns = [x.value for x in returns_data]
    
    # Try to find episode length data
    possible_length_tags = ['charts/episodic_length', 'episodic_length', 'charts/lengths']
    episode_lengths = []
    
    for tag in possible_length_tags:
        if tag in ea.Tags()['scalars']:
            lengths_data = ea.Scalars(tag)
            episode_lengths = [x.value for x in lengths_data]
            print(f"Found length data in tag: {tag}")
            break
    
    return timesteps, returns, episode_lengths


def plot_mean_return_vs_timesteps(run_paths: List[str], labels: List[str], window_size: int = 50, title: str = "Mean Returns vs Timesteps"):
    """
    Plot learning curves (mean return vs timesteps) for multiple runs.
    
    Args:
        run_paths: List of TensorBoard run directory paths
        labels: List of labels for each run
        window_size: Size of rolling window for smoothing
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (run_path, label) in enumerate(zip(run_paths, labels)):
        timesteps, returns, _ = load_tensorboard_run(run_path)
        
        # Create DataFrame and compute rolling mean
        df = pd.DataFrame({'timestep': timesteps, 'return': returns})
        df['rolling_mean'] = df['return'].rolling(window=window_size, min_periods=1).mean()
        
        # Plot with different colors
        color = colors[i % len(colors)]
        plt.plot(df['timestep'], df['rolling_mean'], label=f'{label} (Final: {df["rolling_mean"].iloc[-1]:.3f})', 
                color=color, linewidth=2, alpha=0.8)
        
        # Add raw data as faint background
        plt.plot(df['timestep'], df['return'], color=color, alpha=0.1, linewidth=0.5)
    
    plt.xlabel('Training Timesteps')
    plt.ylabel('Episodic Return')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Learning curves plotted for {len(run_paths)} runs")


def compute_auc_metric(run_path: str) -> float:
    """
    Compute Area Under Curve (AUC) for a learning curve.
    
    Args:
        run_path: TensorBoard run directory path
        
    Returns:
        AUC score (normalized by timestep range)
    """
    timesteps, returns, _ = load_tensorboard_run(run_path)
    
    if len(timesteps) < 2:
        return 0.0
    
    # Sort by timesteps (should already be sorted)
    timesteps_arr = np.array(timesteps)
    returns_arr = np.array(returns)
    
    # Compute AUC using trapezoidal rule
    auc = np.trapezoid(returns_arr, timesteps_arr)
    
    # Normalize by timestep range to get average return
    timestep_range = timesteps_arr[-1] - timesteps_arr[0]
    normalized_auc = auc / timestep_range if timestep_range > 0 else 0.0
    
    return float(normalized_auc)


def plot_auc_comparison(run_paths: List[str], labels: List[str], title: str = "AUC Comparison"):
    """
    Plot bar chart comparing AUC scores across runs.
    
    Args:
        run_paths: List of TensorBoard run directory paths
        labels: List of labels for each run
        title: Plot title
    """
    # Compute AUC for each run
    aucs = []
    for run_path in run_paths:
        auc = compute_auc_metric(run_path)
        aucs.append(auc)
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    bars = plt.bar(labels, aucs, color=colors[:len(labels)], alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(aucs)*0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Run')
    plt.ylabel('Area Under Curve (AUC)')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"AUC comparison plotted for {len(run_paths)} runs")


def plot_returns_over_time_subplot(ax, df: pd.DataFrame):
    """Plot returns over time on given axis."""
    ax.plot(df['timestep'], df['clean_return'], label='Clean', color='blue', alpha=0.8)
    ax.plot(df['timestep'], df['disturbed_return'], label='Disturbed', color='red', alpha=0.8)
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Episodic Return (Rolling Mean)')
    ax.set_title('Learning Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_robustness_index_over_time_subplot(ax, df: pd.DataFrame):
    """Plot robustness index over time on given axis."""
    final_ri = df['robustness_index'].iloc[-1]
    ax.plot(df['timestep'], df['robustness_index'], color='green', linewidth=2, 
           label=f'Robustness Index (Final: {final_ri:.3f})')
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Robustness (1.0)')
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Robustness Index')
    ax.set_title('Robustness Index Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_performance_gap_subplot(ax, df: pd.DataFrame):
    """Plot performance gap over time on given axis."""
    performance_gap = df['clean_return'] - df['disturbed_return']
    ax.fill_between(df['timestep'], 0, performance_gap, alpha=0.6, color='orange')
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Performance Gap (Clean - Disturbed)')
    ax.set_title('Performance Degradation Over Time')
    ax.grid(True, alpha=0.3)


def plot_robustness_distribution_subplot(ax, df: pd.DataFrame):
    """Plot robustness index distribution on given axis."""
    ax.hist(df['robustness_index'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(df['robustness_index'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["robustness_index"].mean():.3f}')
    ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Robustness')
    ax.set_xlabel('Robustness Index')
    ax.set_ylabel('Frequency')
    ax.set_title('Robustness Index Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_robustness_analysis(df: pd.DataFrame, title: str = "Robustness Analysis"):
    """
    Generate comprehensive robustness plots.
    
    Args:
        df: DataFrame from compute_robustness_index_over_time
        title: Plot title
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Call individual subplot functions
    plot_returns_over_time_subplot(axes[0, 0], df)
    plot_robustness_index_over_time_subplot(axes[0, 1], df)
    plot_performance_gap_subplot(axes[1, 0], df)
    plot_robustness_distribution_subplot(axes[1, 1], df)
    
    plt.tight_layout()
    plt.show(block=True)
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"ROBUSTNESS ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"Final Clean Return:      {df['clean_return'].iloc[-1]:.3f}")
    print(f"Final Disturbed Return:  {df['disturbed_return'].iloc[-1]:.3f}")
    print(f"Final Robustness Index:  {df['robustness_index'].iloc[-1]:.3f}")
    print(f"Mean Robustness Index:   {df['robustness_index'].mean():.3f}")
    print(f"Std Robustness Index:    {df['robustness_index'].std():.3f}")
    print(f"Min Robustness Index:    {df['robustness_index'].min():.3f}")
    print(f"Max Robustness Index:    {df['robustness_index'].max():.3f}")


if __name__ == "__main__":
    config = tyro.cli(EvaluationConfig)
    
    # Plot AUC comparison
    plot_auc_comparison(
        run_paths=[config.clean_run_path, config.disturbed_run_path],
        labels=['Clean', 'Disturbed'],
        title="AUC Comparison"
    )
    
    # Plot learning curves comparison
    plot_mean_return_vs_timesteps(
        run_paths=[config.clean_run_path, config.disturbed_run_path],
        labels=['Clean', 'Disturbed'],
        title="Learning Curves Comparison"
    )
    
    # Plot robustness analysis
    df = compute_robustness_index_over_time(config.clean_run_path, config.disturbed_run_path)
    plot_robustness_analysis(df, "Robustness Analysis")
