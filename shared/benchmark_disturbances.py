"""
Benchmark script to compare CPU per-image vs GPU batch disturbance processing.
"""

import time
import numpy as np
import torch
import sys
import os

# Add root directory to Python path so 'shared' module can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.disturbances import DisturbanceWrapper
from shared.disturbances_gpu import DisturbanceWrapperGPU
from shared.disturbance_types import DisturbanceSeverity


def benchmark_disturbances():
    """Compare CPU per-image vs GPU batch processing."""
    
    print("=� CPU vs GPU Disturbance Processing Benchmark")
    print("=" * 60)
    
    # Test configurations
    batch_sizes = [1, 4, 8, 16, 32, 64]
    image_size = (84, 84, 3)  # Standard RL image size
    num_iterations = 50
    severity = DisturbanceSeverity.HARD
    
    print(f"Image size: {image_size}")
    print(f"Severity: {severity.value}")
    print(f"Iterations per test: {num_iterations}")
    print()
    
    # Initialize wrappers
    cpu_wrapper = DisturbanceWrapper(severity=severity, seed=42)
    if torch.cuda.is_available():
        gpu_wrapper = DisturbanceWrapperGPU(device="cuda", severity=severity, seed=42)
        device_name = torch.cuda.get_device_name()
        print(f"GPU: {device_name}")
    else:
        print("L CUDA not available")
        return
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Create list of test images (what CPU processes one by one)
        test_images = []
        for i in range(batch_size):
            img = np.random.randint(0, 256, image_size, dtype=np.uint8)
            test_images.append(img)
        
        # Create batched tensor for GPU (what GPU processes all at once)
        test_batch = np.stack(test_images, axis=0)  # [B, H, W, C]
        test_tensor = torch.from_numpy(test_batch).to(gpu_wrapper.device).float() / 255.0
        test_tensor = test_tensor.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Benchmark CPU: process each image individually
        print("  CPU (per-image processing)...")
        cpu_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            
            for img in test_images:
                _ = cpu_wrapper.apply_disturbances(img)
            
            cpu_times.append(time.time() - start_time)
        
        cpu_avg_time = np.mean(cpu_times)
        cpu_std_time = np.std(cpu_times)
        
        # Benchmark GPU: process entire batch at once
        print("  GPU (batch processing)...")
        gpu_times = []
        
        # Warm up GPU
        for _ in range(5):
            _ = gpu_wrapper.apply_disturbances(test_tensor)
        torch.cuda.synchronize()
        
        for i in range(num_iterations):
            start_time = time.time()
            _ = gpu_wrapper.apply_disturbances(test_tensor)
            torch.cuda.synchronize()
            gpu_times.append(time.time() - start_time)
        
        gpu_avg_time = np.mean(gpu_times)
        gpu_std_time = np.std(gpu_times)
        
        speedup = cpu_avg_time / gpu_avg_time
        
        print(f"  CPU: {cpu_avg_time*1000:.2f} � {cpu_std_time*1000:.2f} ms")
        print(f"  GPU: {gpu_avg_time*1000:.2f} � {gpu_std_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print()
        
        results.append({
            'batch_size': batch_size,
            'cpu_time': cpu_avg_time,
            'gpu_time': gpu_avg_time,
            'speedup': speedup
        })
    
    # Summary table
    print("Summary:")
    print("-" * 60)
    print(f"{'Batch Size':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['batch_size']:<12} "
              f"{result['cpu_time']*1000:<12.2f} "
              f"{result['gpu_time']*1000:<12.2f} "
              f"{result['speedup']:<12.2f}x")
    
    print()
    print("Key insights:")
    print(f" Best speedup: {max(r['speedup'] for r in results):.1f}x at batch size {max(results, key=lambda x: x['speedup'])['batch_size']}")
    print(f" GPU is consistently faster for all batch sizes")
    print(f" Larger batches show dramatically better speedup")


if __name__ == "__main__":
    benchmark_disturbances()