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
    name: str
    run_name: str
    seed: int
    ablation_mode: clip_ppo_utils.AblationMode
    clip_lambda: float
    apply_disturbances: bool
    disturbance_severity: disturbances.DisturbanceSeverity
    description: str
    timesteps: int
    environment: str = "minigrid"  # "minigrid" or "atari"
    env_id: str = "BreakoutNoFrameskip-v4"  # Environment ID


def run_experiment(config: ExperimentConfig):
    """Run a single experiment with given configuration."""
    # Base arguments shared across all experiments
    base_args = [
        "--save-model",
        "--save-freq", "250000",  # Save every 250k steps
        "--clip-config.clip_modality", "text"
    ]

    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"Description: {config.description}")
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
    experiments = []
    """
    CLIP_PPO and PPO GENERATIONS
    """
    # Configurations
    seeds = (0, 42, 2025)
    lambdas = (
        0.0,       # PPO Baseline
        0.000001,  # 1e-6
        0.00001,   # 1e-5 (best result)
        0.0001,    # 1e-4 
    )
    severities = (
        disturbances.DisturbanceSeverity.NONE,
        disturbances.DisturbanceSeverity.MILD,
        disturbances.DisturbanceSeverity.MODERATE,
        disturbances.DisturbanceSeverity.SEVERE,
    )
    environments = {
        'minigrid': [
            'MiniGrid-Empty-16x16-v0',  # Easy
            'MiniGrid-FourRooms-v0',  # Medium
            'MiniGrid-DoorKey-16x16-v0',  # Hard
        ],
        'atari': [
            'ALE/Breakout-v5',  # Easy
            'ALE/Pong-v5',  # Easy
        ]
    }
    timesteps = {
        'minigrid': 5_000_000,
        'atari': 10_000_000,
    }
    for environment_type, env_list in environments.items():
        for env_id in env_list:
            for combo in itertools.product(seeds, lambdas, severities):
                seed, lambda_val, severity = combo
                
                # Generate run name
                algo_name = "ppo" if lambda_val == 0.0 else "clip_ppo"
                run_name = f"{algo_name}_l{lambda_val}_{severity.value}_{env_id}_s{seed}"
                
                experiments.append(ExperimentConfig(
                    name=f"{environment_type.title()} {algo_name} λ={lambda_val} {severity.value} {env_id} seed={seed}",
                    run_name=run_name,
                    seed=seed,
                    ablation_mode=clip_ppo_utils.AblationMode.NONE,
                    clip_lambda=lambda_val,
                    apply_disturbances=(severity != disturbances.DisturbanceSeverity.NONE),
                    disturbance_severity=severity,
                    description=f"{algo_name} λ={lambda_val} with {severity.value} disturbances on {env_id}",
                    environment=environment_type,
                    env_id=env_id,
                    timesteps=timesteps[environment_type],
                ))

    """
    ABLATION GENERATIONS
    """
    ablation_seeds = (0, 42)
    ablations = (
        clip_ppo_utils.AblationMode.FROZEN_CLIP,
        clip_ppo_utils.AblationMode.RANDOM_ENCODER,
    )
    best_lambda = 0.00001  # Use best lambda found from sweep
    
    # Reduced environments for ablations - just need representative examples
    ablation_environments = {
        'minigrid': [
            'MiniGrid-FourRooms-v0',  # Medium complexity, good for ablations
        ],
        'atari': [
            'ALE/Breakout-v5',  # Most common benchmark
        ]
    }
    
    for environment_type, env_list in ablation_environments.items():
        for env_id in env_list:
            for ablation_mode in ablations:
                for combo in itertools.product(ablation_seeds, severities):
                    seed, severity = combo
                    
                    # Generate run name
                    ablation_name = ablation_mode.value
                    run_name = f"clip_ppo_{ablation_name}_l{best_lambda}_{severity.value}_{env_id}_s{seed}"
                    
                    experiments.append(ExperimentConfig(
                        name=f"{environment_type.title()} {ablation_name.replace('_', ' ').title()} λ={best_lambda} {severity.value} {env_id} seed={seed}",
                        run_name=run_name,
                        seed=seed,
                        ablation_mode=ablation_mode,
                        clip_lambda=best_lambda,
                        apply_disturbances=(severity != disturbances.DisturbanceSeverity.NONE),
                        disturbance_severity=severity,
                        description=f"{ablation_name.replace('_', ' ')} ablation λ={best_lambda} with {severity.value} disturbances on {env_id}",
                        environment=environment_type,
                        env_id=env_id,
                        timesteps=timesteps[environment_type],
                    ))


    print(f"Generated {len(experiments)} total experiment combinations")
    
    # Original static experiments (commented out)
    # experiments_old = [
    #     ExperimentConfig(
    #         name="'Minigrid Clean PPO",
    #         run_name="ppo_clean",
    #         ablation_mode="none",
    #         clip_lambda=0.0,
    #         apply_disturbances=False,
    #         disturbance_severity="NONE",  # Unused when disturbances disabled
    #         description="Baseline PPO without disturbances",
    #         environment="minigrid",
    #         env_id="MiniGrid-Fetch-8x8-N3-v0"
    #     ),
        
    #     ExperimentConfig(
    #         name="Minigrid Clean CLIP-PPO", 
    #         run_name="clip_ppo_clean",
    #         ablation_mode="none",
    #         clip_lambda=0.00001,
    #         apply_disturbances=False,
    #         disturbance_severity="NONE",  # Unused when disturbances disabled
    #         description="CLIP-PPO without disturbances",
    #         environment="minigrid",
    #         env_id="MiniGrid-Fetch-8x8-N3-v0"
    #     ),
        
    #     ExperimentConfig(
    #         name="Minigrid PPO Hard Disturbance",
    #         run_name="ppo_hard",
    #         ablation_mode="none", 
    #         clip_lambda=0.0,
    #         apply_disturbances=True,
    #         disturbance_severity="SEVERE",
    #         description="Baseline PPO with hard visual disturbances",
    #         environment="minigrid",
    #         env_id="MiniGrid-Fetch-8x8-N3-v0"
    #     ),
        
    #     ExperimentConfig(
    #         name="Minigrid CLIP-PPO Hard Disturbance",
    #         run_name="clip_ppo_hard",
    #         ablation_mode="none",
    #         clip_lambda=0.00001,
    #         apply_disturbances=True,
    #         disturbance_severity="SEVERE",
    #         description="CLIP-PPO with hard visual disturbances",
    #         environment="minigrid",
    #         env_id="MiniGrid-Fetch-8x8-N3-v0"
    #     ),
        
    #     ExperimentConfig(
    #         name="Minigrid Random Encoder Hard Disturbance",
    #         run_name="clip_ppo_random_encoder_hard",
    #         ablation_mode="random_encoder",
    #         clip_lambda=0.00001,  # Same lambda but random embeddings
    #         apply_disturbances=True,
    #         disturbance_severity="SEVERE",
    #         description="Random encoder ablation with hard disturbances",
    #         environment="minigrid",
    #         env_id="MiniGrid-Fetch-8x8-N3-v0"
    #     ),
        
        # ExperimentConfig(
        #     name="Minigrid Frozen CLIP Hard Disturbance",
        #     run_name="clip_ppo_frozen_clip_hard", 
        #     ablation_mode="frozen_clip",
        #     clip_lambda=0.00001,  # Lambda doesn't matter (disabled for frozen)
        #     apply_disturbances=True,
        #     disturbance_severity="HARD",
        #     description="Frozen CLIP encoder ablation with hard disturbances",
        #     environment="minigrid",
        #     env_id="MiniGrid-Fetch-8x8-N3-v0"
        # ),
    # ]

    # Atari experimetns
    # experiments = [
    #     # CLEAN EXPERIMENTS
    #     ExperimentConfig(
    #         name="Atari Clean PPO",
    #         run_name="ppo_clean",
    #         ablation_mode="none",
    #         clip_lambda=0.0,
    #         apply_disturbances=False,
    #         disturbance_severity="NONE",  # Unused when disturbances disabled
    #         description="Baseline PPO without disturbances",
    #         environment="atari",
    #         env_id="BreakoutNoFrameskip-v4"
    #     ),
    #     ExperimentConfig(
    #         name="Atari Clean CLIP-PPO", 
    #         run_name="clip_ppo_clean",
    #         ablation_mode="none",
    #         clip_lambda=0.00001,
    #         apply_disturbances=False,
    #         disturbance_severity="NONE",  # Unused when disturbances disabled
    #         description="CLIP-PPO without disturbances",
    #         environment="atari",
    #         env_id="BreakoutNoFrameskip-v4"
    #     ),
    #     ExperimentConfig(
    #         name="Atari Clean Random Encoder",
    #         run_name="clip_ppo_random_encoder_clean",
    #         ablation_mode="random_encoder",
    #         clip_lambda=0.00001,  # Same lambda but random embeddings
    #         apply_disturbances=False,
    #         disturbance_severity="NONE",
    #         description="Random encoder ablation without disturbances",
    #         environment="atari",
    #         env_id="BreakoutNoFrameskip-v4"
    #     ),
    #     ExperimentConfig(
    #         name="Atari Clean Frozen CLIP",
    #         run_name="clip_ppo_frozen_clip_clean", 
    #         ablation_mode="frozen_clip",
    #         clip_lambda=0.00001,  # Lambda doesn't matter (disabled for frozen)
    #         apply_disturbances=False,
    #         disturbance_severity="NONE",
    #         description="Frozen CLIP encoder ablation without disturbances",
    #         environment="atari",
    #         env_id="BreakoutNoFrameskip-v4"
    #     ),
    #     # DISTURBANCE EXPERIMENTS
    #     ExperimentConfig(
    #         name="Atari PPO Hard Disturbance",
    #         run_name="ppo_hard",
    #         ablation_mode="none", 
    #         clip_lambda=0.0,
    #         apply_disturbances=True,
    #         disturbance_severity="HARD",
    #         description="Baseline PPO with hard visual disturbances",
    #         environment="atari",
    #         env_id="BreakoutNoFrameskip-v4"
    #     ),
        
    #     ExperimentConfig(
    #         name="Atari CLIP-PPO Hard Disturbance",
    #         run_name="clip_ppo_hard",
    #         ablation_mode="none",
    #         clip_lambda=0.00001,
    #         apply_disturbances=True,
    #         disturbance_severity="HARD",
    #         description="CLIP-PPO with hard visual disturbances",
    #         environment="atari",
    #         env_id="BreakoutNoFrameskip-v4"
    #     ),
        
    #     ExperimentConfig(
    #         name="Atari Random Encoder Hard Disturbance",
    #         run_name="clip_ppo_random_encoder_hard",
    #         ablation_mode="random_encoder",
    #         clip_lambda=0.00001,  # Same lambda but random embeddings
    #         apply_disturbances=True,
    #         disturbance_severity="HARD",
    #         description="Random encoder ablation with hard disturbances",
    #         environment="atari",
    #         env_id="BreakoutNoFrameskip-v4"
    #     ),
        
    #     ExperimentConfig(
    #         name="Atari Frozen CLIP Hard Disturbance",
    #         run_name="clip_ppo_frozen_clip_hard", 
    #         ablation_mode="frozen_clip",
    #         clip_lambda=0.00001,  # Lambda doesn't matter (disabled for frozen)
    #         apply_disturbances=True,
    #         disturbance_severity="HARD",
    #         description="Frozen CLIP encoder ablation with hard disturbances",
    #         environment="atari",
    #         env_id="BreakoutNoFrameskip-v4"
    #     ),
    # ]
    
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
    
    for i, config in enumerate(experiments, 1):
        print(f"\n🚀 Starting experiment {i}/{len(experiments)}")
        success = run_experiment(config)
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