import numpy as np
import time
import matplotlib.pyplot as plt
from dynamic_cell_tree.connected_components import connected_components, connected_components_3D

def generate_random_data_2D(N):
    """
    Generates a random binary mask and a random vector field for a given 2D size N.
    """
    mask = np.random.choice([0, 1], size=(N, N), p=[0.5, 0.5]).astype(np.int32)
    vector_field = np.random.uniform(-1, 1, size=(2, N, N)).astype(np.float32)
    return mask, vector_field

def generate_random_data_3D(N):
    """
    Generates a random binary mask and a random vector field for a given 3D size N.
    """
    mask = np.random.choice([0, 1], size=(N, N, N), p=[0.5, 0.5]).astype(np.int32)
    vector_field = np.random.uniform(-1, 1, size=(3, N, N, N)).astype(np.float32)
    return mask, vector_field

def benchmark_connected_components_2D(connectivity):
    """
    Benchmarks the `connected_components` function for 2D data across various problem sizes.
    """
    problem_sizes = [10, 20, 50, 100, 200, 300, 1000, 5000, 7000]
    times = []

    # Warm-up call for JIT compilation
    mask, vector_field = generate_random_data_2D(10)
    connected_components(mask, vector_field, connectivity)

    for N in problem_sizes:
        mask, vector_field = generate_random_data_2D(N)
        
        # Measure time to run connected_components
        start_time = time.time()
        labels = connected_components(mask, vector_field, connectivity)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        
        print(f"2D Size: {N}x{N} ({N*N} pixels), Time: {elapsed_time:.4f} seconds")

    # Plot with log-log scale
    total_pixels = [N**2 for N in problem_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(total_pixels, times, marker='o', linestyle='-', color='blue', label='Measured Time (2D)')
    
    slope_1 = times[0] * (np.array(total_pixels) / total_pixels[0])
    slope_2 = times[0] * (np.array(total_pixels) / total_pixels[0])**2
    
    plt.loglog(total_pixels, slope_1, linestyle='--', color='green', label="Slope 1 (Linear)")
    plt.loglog(total_pixels, slope_2, linestyle='--', color='red', label="Slope 2 (Quadratic)")
    
    plt.xlabel("Problem Size (Total Pixels: N * N)")
    plt.ylabel("Time (seconds)")
    plt.title(f"2D Connected Components Benchmark with {connectivity}-Connectivity")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

def benchmark_connected_components_3D(connectivity):
    """
    Benchmarks the `connected_components_3D` function for 3D data across various problem sizes.
    """
    problem_sizes = [10, 20, 30, 40, 50, 100, 200]
    times = []

    # Warm-up call for JIT compilation
    mask, vector_field = generate_random_data_3D(10)
    connected_components_3D(mask, vector_field, connectivity)

    for N in problem_sizes:
        mask, vector_field = generate_random_data_3D(N)
        
        # Measure time to run connected_components_3D
        start_time = time.time()
        labels = connected_components_3D(mask, vector_field, connectivity)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        
        print(f"3D Size: {N}x{N}x{N} ({N**3} voxels), Time: {elapsed_time:.4f} seconds")

    # Plot with log-log scale
    total_voxels = [N**3 for N in problem_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(total_voxels, times, marker='o', linestyle='-', color='blue', label='Measured Time (3D)')
    
    slope_1 = times[0] * (np.array(total_voxels) / total_voxels[0])
    slope_2 = times[0] * (np.array(total_voxels) / total_voxels[0])**2
    
    plt.loglog(total_voxels, slope_1, linestyle='--', color='green', label="Slope 1 (Linear)")
    plt.loglog(total_voxels, slope_2, linestyle='--', color='red', label="Slope 2 (Quadratic)")
    
    plt.xlabel("Problem Size (Total Voxels: N * N * N)")
    plt.ylabel("Time (seconds)")
    plt.title(f"3D Connected Components Benchmark with {connectivity}-Connectivity")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

# Run 2D benchmarks for 4-connectivity and 8-connectivity
benchmark_connected_components_2D(4)
benchmark_connected_components_2D(8)

# Run 3D benchmarks for 6-connectivity and 18-connectivity
benchmark_connected_components_3D(6)
benchmark_connected_components_3D(18)
