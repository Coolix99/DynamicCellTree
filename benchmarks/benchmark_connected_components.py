import numpy as np
import time
import matplotlib.pyplot as plt
from dynamic_cell_tree.connected_components import connected_components

def generate_random_data(N):
    """
    Generates a random binary mask and a random vector field for a given size N.
    
    Parameters:
        N (int): Size of the NxN grid.
        
    Returns:
        mask (np.ndarray): NxN binary mask with random values.
        vector_field (np.ndarray): 3D array (2, N, N) with random vector directions.
    """
    mask = np.random.choice([0, 1], size=(N, N), p=[0.5, 0.5]).astype(np.int32)
    vector_field = np.random.uniform(-1, 1, size=(2, N, N)).astype(np.float32)
    return mask, vector_field

def benchmark_connected_components():
    """
    Benchmarks the `connected_components` function across various problem sizes,
    and plots the scaling results.
    """
    problem_sizes = [10, 20, 50, 100, 200, 300]  # Different sizes to benchmark
    times = []

    # Warm-up call for JIT compilation
    mask, vector_field = generate_random_data(10)
    connected_components(mask, vector_field)

    for N in problem_sizes:
        mask, vector_field = generate_random_data(N)
        
        # Measure time to run connected_components
        start_time = time.time()
        labels = connected_components(mask, vector_field)
        end_time = time.time()
        
        # Calculate elapsed time and store
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        
        print(f"Size: {N}x{N} ({N*N} pixels), Time: {elapsed_time:.4f} seconds")

    # Plot with log-log scale
    total_pixels = [N**2 for N in problem_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(total_pixels, times, marker='o', linestyle='-', color='blue', label='Measured Time')
    
    # Add reference lines for slope 1 (linear) and slope 2 (quadratic)
    slope_1 = times[0] * (np.array(total_pixels) / total_pixels[0])  # Linear scaling reference
    slope_2 = times[0] * (np.array(total_pixels) / total_pixels[0])**2  # Quadratic scaling reference
    
    plt.loglog(total_pixels, slope_1, linestyle='--', color='green', label="Slope 1 (Linear)")
    plt.loglog(total_pixels, slope_2, linestyle='--', color='red', label="Slope 2 (Quadratic)")
    
    # Labels and legend
    plt.xlabel("Problem Size (Total Pixels: N * N)")
    plt.ylabel("Time (seconds)")
    plt.title("Benchmark of Connected Components vs Problem Size (Log-Log Scale)")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

# Run the benchmark
benchmark_connected_components()
