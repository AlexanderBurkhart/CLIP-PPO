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
import itertools
from shared import disturbances
from shared import clip_ppo_utils


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    run_name: str
    seed: int
    ablation_mode: clip_ppo_utils.AblationMode
    clip_lambda: float
    apply_disturbances: bool
    disturbance_severity: disturbances.DisturbanceSeverity
    timesteps: int
    environment: str = "minigrid"  # "minigrid" or "atari"
    env_id: str = "BreakoutNoFrameskip-v4"  # Environment ID


def run_experiment(config: ExperimentConfig):
    """Run a single experiment with given configuration."""
    # Base arguments shared across all experiments
    base_args = [
        "--save-model",
        "--save-freq", "250000",  # Save every 250k steps
        "--clip-config.clip_modality", "image"
    ]

    print(f"\n{'='*60}")
    print(f"Running: {config.run_name}")
    print(f"{'='*60}")
    
    # Build command with new config structure
    if config.environment == "atari":
        script_path = "atari_experiments/clip_ppo/clip_ppo_atari.py"
    else:  # minigrid
        script_path = "minigrid_experiments/clip_ppo/clip_ppo_minigrid.py"
        
    cmd = [
        "python", script_path,
        "--env-id", config.env_id,
        "--run-name", config.run_name,
        "--seed", str(config.seed),
        "--clip-config.ablation-mode", config.ablation_mode.value.upper(),
        "--clip-config.clip-lambda", str(config.clip_lambda),
        "--clip-config.disturbance-severity", config.disturbance_severity.value.upper(),
        "--total-timesteps", str(config.timesteps),  # 1M timesteps for meaningful results
    ] + base_args + (['--clip-config.apply-disturbances'] if config.apply_disturbances else [])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run experiment
        result = subprocess.run(cmd, check=True, capture_output=False)
        duration = time.time() - start_time
        print(f"\n✅ {config.run_name} completed successfully in {duration/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ {config.run_name} failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {config.run_name} interrupted by user")
        return False


def _setup_main_experiments():
    experiments = []
    
    seeds = (0, 42)
    timesteps = {
        'minigrid': 1_000_000,
        'atari': 100_000,
    }

    """
    CLIP_PPO and PPO GENERATIONS WITH CLEAN FROZENCLIP
    """
    # Configurations
    lambdas = (
        0.000001,  # 1e-6
        0.00001,   # 1e-5 (best result)
        0.0001,    # 1e-4 
    )
    environment = 'atari'
    environment_ids = [
        'ALE/Breakout-v5',
        'ALE/Pong-v5',
        'ALE/Seaquest-v5'
    ]
    for seed in seeds:
        for env_id in environment_ids:
            # PPO
            experiments.append(
                ExperimentConfig(
                        run_name=f"s{seed}_{environment}_{env_id.replace('/', '')}_PPO_CLEAN",
                        seed=seed,
                        ablation_mode=clip_ppo_utils.AblationMode.NONE,
                        clip_lambda=0.0,
                        apply_disturbances=False,
                        disturbance_severity=disturbances.DisturbanceSeverity.NONE,
                        environment=environment,
                        env_id=env_id,
                        timesteps=timesteps[environment],
                    )
            )
            experiments.append(
                ExperimentConfig(
                        run_name=f"s{seed}_{environment}_{env_id.replace('/', '')}_PPO_MODERATE",
                        seed=seed,
                        ablation_mode=clip_ppo_utils.AblationMode.NONE,
                        clip_lambda=0.0,
                        apply_disturbances=True,
                        disturbance_severity=disturbances.DisturbanceSeverity.MODERATE,
                        environment=environment,
                        env_id=env_id,
                        timesteps=timesteps[environment],
                    )
            )
            experiments.append(
                ExperimentConfig(
                        run_name=f"s{seed}_{environment}_{env_id.replace('/', '')}_PPO_SEVERE",
                        seed=seed,
                        ablation_mode=clip_ppo_utils.AblationMode.NONE,
                        clip_lambda=0.0,
                        apply_disturbances=True,
                        disturbance_severity=disturbances.DisturbanceSeverity.SEVERE,
                        environment=environment,
                        env_id=env_id,
                        timesteps=timesteps[environment],
                    )
            )

            # CLIP-PPO
            for l in lambdas:
                experiments.append(
                    ExperimentConfig(
                        run_name=f"s{seed}_{environment}_{env_id.replace('/', '')}_CLIPPPO_CLEAN_image_l{l}",
                        seed=seed,
                        ablation_mode=clip_ppo_utils.AblationMode.NONE,
                        clip_lambda=l,
                        apply_disturbances=False,
                        disturbance_severity=disturbances.DisturbanceSeverity.NONE,
                        environment=environment,
                        env_id=env_id,
                        timesteps=timesteps[environment],
                    )
                )
                experiments.append(
                    ExperimentConfig(
                            run_name=f"s{seed}_{environment}_{env_id.replace('/', '')}_CLIPPPO_MODERATE_image_l{l}",
                            seed=seed,
                            ablation_mode=clip_ppo_utils.AblationMode.NONE,
                            clip_lambda=l,
                            apply_disturbances=True,
                            disturbance_severity=disturbances.DisturbanceSeverity.MODERATE,
                            environment=environment,
                            env_id=env_id,
                            timesteps=timesteps[environment],
                        )
                )
                experiments.append(
                    ExperimentConfig(
                            run_name=f"s{seed}_{environment}_{env_id.replace('/', '')}_CLIPPPO_SEVERE_image_l{l}",
                            seed=seed,
                            ablation_mode=clip_ppo_utils.AblationMode.NONE,
                            clip_lambda=l,
                            apply_disturbances=True,
                            disturbance_severity=disturbances.DisturbanceSeverity.SEVERE,
                            environment=environment,
                            env_id=env_id,
                            timesteps=timesteps[environment],
                        )
                )

            # Frozen encoder
            experiments.append(
                ExperimentConfig(
                    run_name=f"s{seed}_{environment}_{env_id.replace('/', '')}_PPOFROZENCLIP_CLEAN",
                    seed=seed,
                    ablation_mode=clip_ppo_utils.AblationMode.FROZEN_CLIP,
                    clip_lambda=l,
                    apply_disturbances=False,
                    disturbance_severity=disturbances.DisturbanceSeverity.NONE,
                    environment=environment,
                    env_id=env_id,
                    timesteps=timesteps[environment],
                )
            )


    print(f"Generated {len(experiments)} total experiment combinations")
    return experiments


def main():
    """Run all ablation experiments."""
    experiments = _setup_main_experiments()
    
    print("Starting CLIP-PPO Experiments")
    print(f"Total experiments: {len(experiments)}")
    print(f"Estimated time per experiment: ~20 minutes")
    print(f"Total estimated time: ~{len(experiments) * 20:.0f} minutes")
    
    # Confirm start
    response = input("\nProceed with experiments? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run experiments
    results = {}
    total_start_time = time.time()
    experiment_durations = []
    

    start_at = 0
    for i, config in enumerate(experiments, 1):
        if i < start_at: continue
        print(f"\n🚀 Starting experiment {i}/{len(experiments)}")
        
        # Dynamic time estimation
        if experiment_durations:
            avg_duration = sum(experiment_durations) / len(experiment_durations)
            remaining_experiments = len(experiments) - i
            estimated_remaining_time = avg_duration * remaining_experiments
            
            print(f"📊 Time estimates:")
            print(f"   Average per experiment: {avg_duration/60:.1f} minutes")
            print(f"   Estimated remaining: {estimated_remaining_time/3600:.1f} hours")
            print(f"   Estimated completion: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + estimated_remaining_time))}")
        
        exp_start_time = time.time()
        success = run_experiment(config)
        exp_duration = time.time() - exp_start_time
        experiment_durations.append(exp_duration)
        
        results[config.run_name] = success
        
        if not success:
            response = input(f"\nExperiment {config.run_name} failed. Continue with remaining experiments? (y/N): ")
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