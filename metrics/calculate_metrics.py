"""
Shared metric calculation functions for evaluation scripts.
"""
import numpy as np
import pandas as pd
import sys
import os
from typing import List, Tuple
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Add path to access disturbances module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'minigrid_experiments'))
from disturbances import DisturbanceSeverity


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
    
    # Try to find episodic return data
    possible_return_tags = ['charts/episodic_return', 'episodic_return', 'charts/returns']
    returns_data = None
    
    for tag in possible_return_tags:
        if tag in ea.Tags()['scalars']:
            returns_data = ea.Scalars(tag)
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
            break
    
    return timesteps, returns, episode_lengths


def get_disturbance_severity(run_path: str) -> DisturbanceSeverity:
    """
    Get disturbance severity from TensorBoard logs.
    
    Args:
        run_path: Path to TensorBoard run directory
        
    Returns:
        DisturbanceSeverity enum value
    """
    ea = EventAccumulator(run_path)
    ea.Reload()
    
    # Try to find disturbance severity in tensor data (TensorBoard text is stored as tensors)
    if 'config/disturbance_severity/text_summary' in ea.Tags().get('tensors', []):
        text_data = ea.Tensors('config/disturbance_severity/text_summary')
        if text_data:
            # Extract text from tensor data
            severity_text = text_data[0].tensor_proto.string_val[0].decode('utf-8')
            
            # Map text to enum
            severity_mapping = {
                'CLEAN': DisturbanceSeverity.NONE,
                'MILD': DisturbanceSeverity.MILD,
                'MODERATE': DisturbanceSeverity.MODERATE,
                'HARD': DisturbanceSeverity.HARD,
                'SEVERE': DisturbanceSeverity.SEVERE,
            }
            
            return severity_mapping.get(severity_text, DisturbanceSeverity.NONE)
    
    # Fallback: try to infer from run path
    if 'clean' in run_path.lower():
        return DisturbanceSeverity.NONE
    elif 'mild' in run_path.lower():
        return DisturbanceSeverity.MILD
    elif 'moderate' in run_path.lower():
        return DisturbanceSeverity.MODERATE
    elif 'hard' in run_path.lower():
        return DisturbanceSeverity.HARD
    elif 'severe' in run_path.lower():
        return DisturbanceSeverity.SEVERE
    
    return DisturbanceSeverity.NONE


def compute_robustness_index_over_time(clean_run_path: str, disturbed_run_path: str, window_size: int = 50) -> pd.DataFrame:
    """
    Compute robustness index over time using rolling windows.
    
    Args:
        clean_run_path: Path to clean TensorBoard run
        disturbed_run_path: Path to disturbed TensorBoard run
        window_size: Size of rolling window for computing means
        
    Returns:
        DataFrame with columns: timestep, clean_return, disturbed_return, robustness_index
    """
    # Load both runs
    clean_timesteps, clean_returns, _ = load_tensorboard_run(clean_run_path)
    disturbed_timesteps, disturbed_returns, _ = load_tensorboard_run(disturbed_run_path)
    
    # Create DataFrames
    clean_df = pd.DataFrame({'timestep': clean_timesteps, 'return': clean_returns})
    disturbed_df = pd.DataFrame({'timestep': disturbed_timesteps, 'return': disturbed_returns})
    
    # Compute rolling means
    clean_df['rolling_mean'] = clean_df['return'].rolling(window=window_size, min_periods=1).mean()
    disturbed_df['rolling_mean'] = disturbed_df['return'].rolling(window=window_size, min_periods=1).mean()
    
    # Merge on episode index (assumes same number of episodes)
    min_episodes = min(len(clean_df), len(disturbed_df))
    clean_df = clean_df.iloc[:min_episodes]
    disturbed_df = disturbed_df.iloc[:min_episodes]
    
    # Compute robustness index
    result_df = pd.DataFrame({
        'timestep': clean_df['timestep'],
        'clean_return': clean_df['rolling_mean'], 
        'disturbed_return': disturbed_df['rolling_mean']
    })
    
    result_df['robustness_index'] = result_df['disturbed_return'] / result_df['clean_return']
    
    print(f"Final robustness index: {result_df['robustness_index'].iloc[-1]:.3f}")
    print(f"Mean robustness index: {result_df['robustness_index'].mean():.3f}")
    
    return result_df


def compute_robustness_index(clean_run_path: str, disturbed_run_path: str, window_size: int = 50) -> float:
    """
    Compute final robustness index between clean and disturbed runs.
    
    Args:
        clean_run_path: Path to clean TensorBoard run
        disturbed_run_path: Path to disturbed TensorBoard run
        window_size: Size of rolling window for computing means (1 for raw values)
        
    Returns:
        Final robustness index (disturbed_final / clean_final)
    """
    df = compute_robustness_index_over_time(clean_run_path, disturbed_run_path, window_size)
    return df['robustness_index'].iloc[-1]


def compute_auc_metric(timesteps: List[int], returns: List[float]) -> float:
    """
    Compute Area Under Curve (AUC) for a learning curve.
    
    Args:
        timesteps: List of timestep values
        returns: List of corresponding return values
        
    Returns:
        AUC score (normalized by timestep range)
    """
    if len(timesteps) < 2:
        return 0.0
    
    timesteps_arr = np.array(timesteps)
    returns_arr = np.array(returns)
    
    # Compute AUC using trapezoidal rule
    auc = np.trapezoid(returns_arr, timesteps_arr)
    
    # Normalize by timestep range to get average return
    timestep_range = timesteps_arr[-1] - timesteps_arr[0]
    normalized_auc = auc / timestep_range if timestep_range > 0 else 0.0
    
    return float(normalized_auc)


def compute_success_rate(run_path: str, success_threshold: float = 0.0) -> float:
    """
    Compute success rate for MiniGrid environment.
    
    Args:
        run_path: TensorBoard run directory path
        success_threshold: Minimum return to consider success (default 0.0 = any positive reward)
        
    Returns:
        Success rate as percentage (0.0 to 100.0)
    """
    timesteps, returns, _ = load_tensorboard_run(run_path)
    
    if not returns:
        return 0.0
    
    # In MiniGrid, success = return > threshold (agent reached goal), failure = return <= threshold (timeout/death)
    successes = sum(1 for r in returns if r > success_threshold)
    total_episodes = len(returns)
    
    success_rate = (successes / total_episodes) * 100.0
    return success_rate


def compute_final_success_rate(run_path: str, window_size: int = 100, success_threshold: float = 0.0) -> float:
    """
    Compute success rate over final episodes (rolling window).
    
    Args:
        run_path: TensorBoard run directory path
        window_size: Number of final episodes to consider
        success_threshold: Minimum return to consider success (default 0.0 = any positive reward)
        
    Returns:
        Final success rate as percentage (0.0 to 100.0)
    """
    timesteps, returns, _ = load_tensorboard_run(run_path)
    
    if not returns:
        return 0.0
    
    # Take final episodes
    final_returns = returns[-window_size:] if len(returns) >= window_size else returns
    
    # Count successes in final episodes (agent reached goal vs timed out)
    successes = sum(1 for r in final_returns if r > success_threshold)
    total_episodes = len(final_returns)
    
    success_rate = (successes / total_episodes) * 100.0
    return success_rate