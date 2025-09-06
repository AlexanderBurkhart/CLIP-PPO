#!/usr/bin/env python3
"""
Run ablation study experiments for CLIP-PPO paper.
This script runs all required experiments sequentially.
"""

import subprocess
import time
import os
from dataclasses import dataclass
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    run_name: str
    ablation_mode: str
    clip_lambda: float
    apply_disturbances: bool
    disturbance_severity: str
    description: str


def run_experiment(config: ExperimentConfig, base_args: List[str]):
    """Run a single experiment with given configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        "python", "minigrid_experiments/clip_ppo/clip_ppo_minigrid.py",
        "--run-name", config.run_name,
        "--ablation-mode", config.ablation_mode.upper(),
        "--clip-lambda", str(config.clip_lambda),
        "--disturbance-severity", config.disturbance_severity,
    ] + base_args + (['--apply-disturbances'] if config.apply_disturbances else [])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run experiment
        result = subprocess.run(cmd, check=True, capture_output=False)
        duration = time.time() - start_time
        print(f"\n✅ {config.name} completed successfully in {duration/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ {config.name} failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {config.name} interrupted by user")
        return False


def main():
    """Run all ablation experiments."""
    
    # Base arguments shared across all experiments
    base_args = [
        "--total-timesteps", "750000",  # 1M timesteps for meaningful results
        "--env-id", "MiniGrid-Fetch-8x8-N3-v0",
        "--seed", "42",
        "--save-model",
        "--save-freq", "250000",  # Save every 250k steps
    ]
    
    # Define experiments
    experiments = [
        ExperimentConfig(
            name="Clean PPO",
            run_name="ppo_clean",
            ablation_mode="none",
            clip_lambda=0.0,
            apply_disturbances=False,
            disturbance_severity="NONE",  # Unused when disturbances disabled
            description="Baseline PPO without disturbances"
        ),
        
        ExperimentConfig(
            name="Clean CLIP-PPO", 
            run_name="clip_ppo_clean",
            ablation_mode="none",
            clip_lambda=0.00001,
            apply_disturbances=False,
            disturbance_severity="NONE",  # Unused when disturbances disabled
            description="CLIP-PPO without disturbances"
        ),
        
        ExperimentConfig(
            name="PPO Hard Disturbance",
            run_name="ppo_hard",
            ablation_mode="none", 
            clip_lambda=0.0,
            apply_disturbances=True,
            disturbance_severity="HARD",
            description="Baseline PPO with hard visual disturbances"
        ),
        
        ExperimentConfig(
            name="CLIP-PPO Hard Disturbance",
            run_name="clip_ppo_hard",
            ablation_mode="none",
            clip_lambda=0.00001,
            apply_disturbances=True,
            disturbance_severity="HARD",
            description="CLIP-PPO with hard visual disturbances"
        ),
        
        ExperimentConfig(
            name="Random Encoder Hard Disturbance",
            run_name="clip_ppo_random_encoder_hard",
            ablation_mode="random_encoder",
            clip_lambda=0.00001,  # Same lambda but random embeddings
            apply_disturbances=True,
            disturbance_severity="HARD",
            description="Random encoder ablation with hard disturbances"
        ),
        
        ExperimentConfig(
            name="Frozen CLIP Hard Disturbance",
            run_name="clip_ppo_frozen_clip_hard", 
            ablation_mode="frozen_clip",
            clip_lambda=0.00001,  # Lambda doesn't matter (disabled for frozen)
            apply_disturbances=True,
            disturbance_severity="HARD",
            description="Frozen CLIP encoder ablation with hard disturbances"
        ),
    ]
    
    print("Starting CLIP-PPO Ablation Study")
    print(f"Total experiments: {len(experiments)}")
    print(f"Estimated time per experiment: ~10-15 minutes (1M timesteps)")
    print(f"Total estimated time: ~{len(experiments) * 12:.0f} minutes")
    
    # Confirm start
    response = input("\nProceed with experiments? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run experiments
    results = {}
    total_start_time = time.time()
    
    for i, config in enumerate(experiments, 1):
        print(f"\n🚀 Starting experiment {i}/{len(experiments)}")
        success = run_experiment(config, base_args)
        results[config.name] = success
        
        if not success:
            response = input(f"\nExperiment {config.name} failed. Continue with remaining experiments? (y/N): ")
            if response.lower() != 'y':
                print("Stopping experiments.")
                break
    
    # Summary
    total_duration = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_duration/60:.1f} minutes")
    print("\nResults:")
    
    successful = 0
    for exp_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {exp_name}: {status}")
        if success:
            successful += 1
    
    print(f"\nCompleted: {successful}/{len(results)} experiments")
    
    if successful == len(results):
        print("\n🎉 All experiments completed successfully!")
        print("\nNext steps:")
        print("1. Run metrics analysis: python metrics/compare_algorithms.py")
        print("2. Generated runs are in: runs/")
        print("3. TensorBoard logs available for visualization")
    else:
        print(f"\n⚠️  {len(results) - successful} experiments failed. Check logs for details.")


if __name__ == "__main__":
    # Change to project directory if needed
    if os.path.basename(os.getcwd()) != "clip-ppo":
        print("Please run this script from the clip-ppo project root directory")
        exit(1)
    
    main()