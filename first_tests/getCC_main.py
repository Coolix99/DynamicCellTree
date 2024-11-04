import numpy as np
import matplotlib.pyplot as plt
import os
import json
import h5py

from dynamic_cell_tree.connected_components import connected_components
from dynamic_cell_tree.center_finding import find_label_centers
from dynamic_cell_tree.separation import find_label_separation
from dynamic_cell_tree.build_tree import build_trees_from_splits
from dynamic_cell_tree.label_operations import relabel_sequentially
from dynamic_cell_tree.merge_near_centers import merge_close_labels


def load_compressed_array(filename):
    # Load the mask, non-zero elements, and original shape from the file
    with h5py.File(filename, 'r') as f:
        mask = f['mask'][:]
        non_zero_elements = f['non_zero_elements'][:]
        shape = f.attrs['shape']

    # Expand the mask to the original shape
    expanded_mask = np.broadcast_to(mask, shape)
    
    # Create an empty array of the original shape
    array = np.zeros(shape, dtype=non_zero_elements.dtype)

    # Reconstruct the original array using the expanded mask and the non-zero elements
    array[expanded_mask] = non_zero_elements
    
    return array

def get_JSON(dir,name=None):
    if name is None:
        name='MetaData.json'
    else:
        name = 'MetaData_'+name+'.json'
    json_file_path=os.path.join(dir, name)
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print("MetaData doesn't exist", dir, name)
        data = {}  # Create an empty dictionary if the file doesn't exist
    return data


def generate_example_data():

    apply_folder='20220610_mAG-zGem_H2a-mcherry_78hpf_LM_A3_analyzed_nuclei'
    print(apply_folder)
    applyresult_folder_path=(r'/media/max_kotz/random_data/applyresult/')
    #propresult_folder_path=(r'/media/max_kotz/random_data/propresult/{}').format("")
    #prop_dir_path=os.path.join(propresult_folder_path,apply_folder)
    apply_dir_path=os.path.join(applyresult_folder_path,apply_folder)

    
    
    MetaData_apply=get_JSON(apply_dir_path)["apply_MetaData"]
    flow_file=os.path.join(apply_dir_path,MetaData_apply['pred_flows file'])
    mask_file=os.path.join(apply_dir_path,MetaData_apply['segmentation file'])

    flow=load_compressed_array(flow_file)[1:,300,200:800,200:800]
    mask=load_compressed_array(mask_file)[300,200:800,200:800]

    return mask,flow

def generate_example_data_small(N=20):
    """
    Generates a small NxN binary mask with four squares of size (N//5)x(N//5).
    Also generates a flow field with vectors pointing to the centers of each square.
    
    Parameters:
        N (int): Size of the NxN grid.
        
    Returns:
        mask (np.ndarray): NxN binary mask with four squares.
        vector_field (np.ndarray): 3D array (2, N, N) where each vector points to the center 
                                   of the respective square in the binary mask.
    """
    mask = np.zeros((N, N), dtype=int)
    vector_field = np.zeros((2, N, N), dtype=float)
    square_size = N // 5
    print(square_size)
    # Define square centers for the four squares
    centers = [
        (2*square_size // 2, 2*square_size // 2),  # Top-left
        (int(3.5*square_size // 2), int(3.5*square_size // 2)),  # extra 
        (int(3.5*square_size // 2), int(5*square_size // 2)),  # extra 
        (int(3.5*square_size // 2), int(7*square_size // 2)),  # extra 
        (2*square_size // 2, 8 * square_size // 2),  # Top-right
        (8 * square_size // 2, 2*square_size // 2),  # Bottom-left
        (int(7*square_size // 2), int(3*square_size // 2)),  # extra 
        (8 * square_size // 2, 8 * square_size // 2)  # Bottom-right
    ]
    
    # Create four squares in the binary mask and assign flow field directions
    for idx, (center_x, center_y) in enumerate(centers):
        print(center_x,center_y)
        # Calculate top-left corner of the square
        start_x = center_x - square_size//2
        start_y = center_y - square_size//2

        mask[start_x:start_x + square_size, start_y:start_y + square_size] = 1

        # Generate flow field pointing to the center of each square
        for i in range(start_x, start_x + square_size):
            for j in range(start_y, start_y + square_size):
                dx, dy = center_x - i, center_y - j
                norm = np.sqrt(dx**2 + dy**2) if dx != 0 or dy != 0 else 1
                vector_field[0, i, j] = dx / norm
                vector_field[1, i, j] = dy / norm

    return mask, vector_field



def show_results(mask, vector_field, labels, plot_labels=True):
    """
    Displays the binary mask, vector field, and connected components.
    """
    N = mask.shape[0]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the binary mask
    axs[0].imshow(mask, cmap="gray")
    axs[0].set_title("Binary Mask of Cells")

    # Plot the vector field
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    U = vector_field[1, :, :]  # Swapped due to coordinate alignment in quiver
    V = -vector_field[0, :, :] # Negative to align with image Y-axis direction
    axs[1].imshow(mask, cmap="gray", alpha=0.5)
    axs[1].quiver(X, Y, U, V, color="blue", scale=1, scale_units="xy")
    axs[1].set_title("Vector Field (Pointing to Center)")

    # Plot the connected components if requested
    if plot_labels:
        axs[2].imshow(labels, cmap="Set1")
        axs[2].set_title("Connected Components")
    else:
        axs[2].axis("off")
    
    
    plt.show()

import napari
def show_results_napari(mask, vector_field, labels):
    """
    Displays the binary mask, vector field, and connected components using napari.
    """
    N = mask.shape[0]
    
    # Prepare vector data for napari Vectors layer
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    start_points = np.vstack([Y.ravel(), X.ravel()]).T  # (N^2, 2) array of starting points
    U = vector_field[1, :, :].ravel()  # X-component
    V = vector_field[0, :, :].ravel()  # Y-component (negative for display alignment)
    
    # Compute end points by adding the direction to the start point
    direction = np.vstack([V, U]).T
    vector_data = np.stack([start_points, direction], axis=1)  # Shape (N^2, 2, 2)

    # Open napari viewer
    viewer = napari.Viewer()

    # Add the binary mask as an image layer
    viewer.add_image(mask, name="Binary Mask", colormap="gray", opacity=0.5)

    # Add the vector field as a vectors layer
    viewer.add_vectors(vector_data, edge_color="blue", name="Vector Field")

    # Add connected components as a labels layer
    viewer.add_labels(labels, name="Connected Components")

    # Start napari event loop
    napari.run()



def main():
    #mask, vector_field = generate_example_data_small(40)
    mask, vector_field = generate_example_data()

    # Identify connected components using the vector field and CCL
    labels = connected_components(mask, vector_field,connectivity=8)
   
    labels=relabel_sequentially(labels)
    print(np.unique(labels))

    centers=find_label_centers(labels, vector_field,connectivity=8)
    print(centers)

    labels=merge_close_labels(labels, centers, merge_distance=2)

    splits=find_label_separation(labels, vector_field, cutoff=20,connectivity=8)
    print(splits)
    print(build_trees_from_splits(splits))
    
    show_results_napari(mask,vector_field,labels)
    # # Show results with components optionally displayed
    # show_results(mask, vector_field, labels, plot_labels=True)

import time
def timing():
    #warm up
    mask, vector_field=generate_example_data_small(20)
    labels = connected_components(mask, vector_field)

    sizes = [20, 50, 100, 200, 1000]
    times = []

    for N in sizes:
        print(f"\nTesting with N = {N}")
        
        # Generate example data
        mask, vector_field = generate_example_data_small(N)
        
        # Measure the time for connected_components
        start_time = time.time()
        labels = connected_components(mask, vector_field)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        print(f"Time taken for connected_components with N={N}: {elapsed_time:.4f} seconds")

    # Display timing results summary
    print("\nSummary of timings for different sizes:")
    for N, t in zip(sizes, times):
        print(f"N = {N}: {t:.4f} seconds")
    
    # Convert sizes and times to numpy arrays for plotting
    total_pixels = np.array(sizes)**2  # N^2 represents the total number of pixels
    times_np = np.array(times)

    # Plot times vs total_pixels (N^2) in log-log scale
    plt.figure(figsize=(10, 6))
    plt.loglog(total_pixels, times_np, marker='o', label="Measured Time", color='blue')
    
    # Plot reference lines for linear and quadratic scaling based on total pixels
    slope_1 = times_np[0] * (total_pixels / total_pixels[0])  # Linear scaling reference
    plt.loglog(total_pixels, slope_1, linestyle='--', label="Slope 1 (Linear)", color='green')
    
    slope_2 = times_np[0] * (total_pixels / total_pixels[0])**2  # Quadratic scaling reference
    plt.loglog(total_pixels, slope_2, linestyle='--', label="Slope 2 (Quadratic)", color='red')
    
    # Labels and legend
    plt.xlabel("Total Pixels (N^2)")
    plt.ylabel("Time (seconds)")
    plt.title("Timing Analysis of Connected Components Labeling (vs Total Pixels)")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()


        
        

if __name__ == "__main__":
    main()
    #timing()
