import numpy as np
import matplotlib.pyplot as plt
import os
import json
import h5py
from numba import njit

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



@njit
def find(parent, label):
    """Finds the root of the label with path compression."""
    if parent[label] != label:
        parent[label] = find(parent, parent[label])
    return parent[label]

@njit
def union(parent, rank, label1, label2):
    """Unites two labels using union by rank."""
    root1, root2 = find(parent, label1), find(parent, label2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        elif rank[root1] < rank[root2]:
            parent[root1] = root2
        else:
            parent[root2] = root1
            rank[root1] += 1

@njit
def convert_to_direction(vx, vy):
    """Converts vector field values to a directional step."""
    if abs(vx) > abs(vy):
        return (1, 0) if vx > 0 else (-1, 0)
    else:
        return (0, 1) if vy > 0 else (0, -1)

@njit
def connected_components(mask, vector_field):
    """
    Identifies connected components in a binary mask using a two-pass
    Union-Find algorithm with vector field connectivity.
    
    Parameters:
        mask (np.ndarray): Binary image (2D array).
        vector_field (np.ndarray): 3D array (2, N, N) for vector directions.
        
    Returns:
        labels (np.ndarray): Labeled image where each component has a unique label.
    """
    N = mask.shape[0]
    labels = np.zeros((N, N), dtype=np.int32)
    parent = np.arange(N * N, dtype=np.int32)  # Flattened parent array
    rank = np.zeros(N * N, dtype=np.int32)
    current_label = 1
    
    # First pass: Label and union-find setup
    for i in range(N):
        for j in range(N):
            if mask[i, j] == 1:
                if labels[i, j] == 0:
                    labels[i, j] = current_label
                    current_label += 1

                vx, vy = vector_field[0, i, j], vector_field[1, i, j]
                dx, dy = convert_to_direction(vx, vy)
                nx, ny = i + dx, j + dy

                # Check if neighbor is within bounds and also in the mask
                if 0 <= nx < N and 0 <= ny < N and mask[nx, ny] == 1:
                    if labels[nx, ny] == 0:
                        labels[nx, ny] = labels[i, j]
                    label1 = i * N + j
                    label2 = nx * N + ny
                    union(parent, rank, label1, label2)

    # Second pass: Flatten labels to root labels
    for i in range(N):
        for j in range(N):
            if labels[i, j] != 0:
                labels[i, j] = find(parent, i * N + j)

    return labels


@njit
def relabel_sequentially(labels):
    """
    Remaps labels to sequential values starting from 1.

    Parameters:
        labels (np.ndarray): The labeled image array with potentially non-sequential labels.

    Returns:
        np.ndarray: The relabeled image with sequential labels starting from 1.
    """
    unique_labels = np.unique(labels)
    label_mapping = {}

    # Create a sequential mapping starting from 1
    new_label = 0
    for old_label in unique_labels:
        label_mapping[old_label] = new_label
        new_label += 1
    
    # Apply the new sequential mapping
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] in label_mapping:
                labels[i, j] = label_mapping[labels[i, j]]
            else:
                raise

    return labels

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

def compute_progenitors(label, target_x, target_y, vector_field, labels, cutoff=10):
    """
    Performs a DFS to find all progenitor pixels pointing towards the target pixel.
    
    Parameters:
        label (int): Label of the target pixel.
        target_x (int): X-coordinate of the target pixel.
        target_y (int): Y-coordinate of the target pixel.
        vector_field (np.ndarray): The vector field indicating directions.
        labels (np.ndarray): The label matrix of the image.
        cutoff (int): Maximum depth to trace back progenitors.

    Returns:
        int: The depth of progenitor chain up to the cutoff, representing the "separation time".
    """
    stack = [(target_x, target_y)]
    visited = set()
    depth = 0

    while stack and depth < cutoff:
        current_size = len(stack)  # Track current depth level
        for _ in range(current_size):
            current_x, current_y = stack.pop()
            
            # Skip if this pixel has already been visited
            if (current_x, current_y) in visited:
                continue
            
            # Mark as visited
            visited.add((current_x, current_y))
            
            # Check all 4 neighbors to see if they point to the current pixel
            neighbors = [(current_x + 1, current_y), (current_x - 1, current_y), 
                         (current_x, current_y + 1), (current_x, current_y - 1)]
            for nx, ny in neighbors:
                if (nx < 0 or nx >= labels.shape[0] or ny < 0 or ny >= labels.shape[1]):
                    continue  # Skip out-of-bounds
                
                # Check if the neighbor belongs to the same label
                if labels[nx, ny] != label:
                    continue
                
                # Get the direction vector of the neighbor
                vx, vy = vector_field[0, nx, ny], vector_field[1, nx, ny]
                dx, dy = convert_to_direction(vx, vy)
                
                # Check if the neighbor points to the current pixel
                if nx + dx == current_x and ny + dy == current_y:
                    # Add the neighbor to the stack to continue the DFS
                    stack.append((nx, ny))

        # Increase depth level after exploring current depth level
        depth += 1
    
    return depth


def find_label_separation(labels, vector_field, cutoff=10):
    """
    Computes a hierarchy of separation times between neighboring labels.

    Parameters:
        labels (np.ndarray): Labeled image with different connected components.
        vector_field (np.ndarray): The vector field indicating directions.
        cutoff (int): Maximum depth to trace progenitors.

    Returns:
        dict: Dictionary with keys as (label1, label2) and values as the maximum separation time.
    """
    separation_times = {}

    # Iterate over each pixel to find neighbors with different labels
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            current_label = labels[x, y]
            if current_label == 0:
                continue  # Skip background pixels
            
            # Check 4-neighboring pixels
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            for nx, ny in neighbors:
                if nx < 0 or nx >= labels.shape[0] or ny < 0 or ny >= labels.shape[1]:
                    continue  # Skip out-of-bounds
                neighbor_label = labels[nx, ny]
                
                # Only consider pixels with different labels (non-background)
                if neighbor_label != 0 and neighbor_label != current_label:
                    # Check if vectors point away from each other
                    dx1, dy1 = convert_to_direction(vector_field[0, x, y], vector_field[1, x, y])
                    dx2, dy2 = convert_to_direction(vector_field[0, nx, ny], vector_field[1, nx, ny])
                    if dx1 == dx2 and  dy1 == dy2:  # Vectors point towards each other, not separated
                        continue
                    
                    # Compute progenitor chain lengths for both labels
                    s1 = compute_progenitors(current_label, x, y, vector_field, labels, cutoff)
                    s2 = compute_progenitors(neighbor_label, nx, ny, vector_field, labels, cutoff)
                    
                    # Separation time for this label pair
                    separation_time = min(s1, s2)
                    label_pair = (min(current_label, neighbor_label), max(current_label, neighbor_label))
                    
                    # Update the maximum separation time for the label pair
                    if label_pair in separation_times:
                        separation_times[label_pair] = max(separation_times[label_pair], separation_time)
                    else:
                        separation_times[label_pair] = separation_time

    return separation_times


@njit
def find_label_centers(labels, vector_field, distance_threshold=2, steps_check=5):
    """
    Finds the 'singular' center for each label by following vector directions until the position gets stuck.
    
    Parameters:
        labels (np.ndarray): Labeled image with different connected components.
        vector_field (np.ndarray): The vector field indicating directions.
        distance_threshold (int): Distance threshold to determine if the walk has "stuck".
        steps_check (int): Number of steps to check if the movement has reduced.
        
    Returns:
        dict: Dictionary with keys as labels and values as (x, y) center coordinates.
    """
    visited = set()
    centers = {}

    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            current_label = labels[x, y]
            if current_label == 0 or current_label in visited:
                continue  # Skip background or already processed labels
            
            # Initialize the starting point for this label
            current_x, current_y = x, y
            last_positions = [(current_x, current_y)]  # Track last positions for distance check
            
            while True:
                # Get the direction from the vector field
                vx, vy = vector_field[0, current_x, current_y], vector_field[1, current_x, current_y]
                dx, dy = convert_to_direction(vx, vy)
                next_x, next_y = current_x + dx, current_y + dy
                
                # Check if the next position is within bounds and belongs to the same label
                if (next_x < 0 or next_x >= labels.shape[0] or 
                    next_y < 0 or next_y >= labels.shape[1] or
                    labels[next_x, next_y]==0):
                    break  # Stop if out of bounds 
                
                # Update positions
                current_x, current_y = next_x, next_y
                last_positions.append((current_x, current_y))
                
                # Maintain only the last `steps_check` positions
                if len(last_positions) > steps_check:
                    last_positions.pop(0)
                
                    # Check if the position is "stuck"
                    total_distance = 0
                    for i in range(len(last_positions) - 1):
                        x1, y1 = last_positions[0]
                        x2, y2 = last_positions[-1]
                        total_distance += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    if total_distance < distance_threshold:
                        break
            
            # Record the center of the label
            centers[current_label] = (current_x, current_y)
            visited.add(current_label)

    return centers

@njit
def merge_close_labels(labels, centers, merge_distance=2):
    """
    Merges labels with centers closer than a specified distance by updating the `labels` array.
    
    Parameters:
        labels (np.ndarray): Labeled image with different connected components.
        centers (dict): Dictionary with labels as keys and (x, y) centers as values.
        merge_distance (int): Distance threshold for merging labels.
        
    Returns:
        np.ndarray: Updated label array after merging close labels.
    """
    label_keys = list(centers.keys())
    num_labels = len(label_keys)

    # Iterate over each pair of label centers
    for i in range(num_labels):
        label1 = label_keys[i]
        x1, y1 = centers[label1]

        for j in range(i + 1, num_labels):
            label2 = label_keys[j]
            x2, y2 = centers[label2]
            
            # Calculate distance between centers
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if label1==46 and label2==47:
                print(label1,label2)
                print(distance)
                print(distance < merge_distance)
            # Merge if centers are closer than merge_distance
            if distance < merge_distance:
                # Update labels array: replace `label2` with `label1`
                for x in range(labels.shape[0]):
                    for y in range(labels.shape[1]):
                        if labels[x, y] == label2:
                            labels[x, y] = label1

    
    return labels

from collections import defaultdict

class QTUnionFind:
    def __init__(self, size):
        self.parent = [-1] * size
        self.rank = [0] * size
        self.size = 0

    def make_set(self, q):
        self.parent[self.size] = -1
        self.rank[self.size] = 0
        self.size += 1

    def find_canonical(self, q):
        r = q
        while self.parent[r] >= 0:
            r = self.parent[r]
        while self.parent[q] >= 0:
            tmp = q
            q = self.parent[q]
            self.parent[tmp] = r
        return r

    def union(self, cx, cy):
        if self.rank[cx] > self.rank[cy]:
            cx, cy = cy, cx
        if self.rank[cx] == self.rank[cy]:
            self.rank[cy] += 1
        self.parent[cx] = cy
        return cy

class QEBTUnionFind:
    def __init__(self, size):
        self.QBT_parent = [-1] * (2 * size - 1)
        self.QBT_children = defaultdict(list)
        self.QBT_size = size
        self.QT = QTUnionFind(2 * size - 1)
        self.Root = list(range(size))
        self.split_times = {}  # To store separation times for each new parent

    def make_set(self, q):
        self.Root[q] = q
        self.QBT_parent[q] = -1
        self.QT.make_set(q)

    def find_canonical(self, q):
        return self.QT.find_canonical(q)

    def union(self, cx, cy, sep_time):
        tu = self.Root[cx]
        tv = self.Root[cy]
        new_parent = self.QBT_size
        self.QBT_size += 1
        self.QBT_parent[tu] = new_parent
        self.QBT_parent[tv] = new_parent
        self.QBT_children[new_parent].extend([tu, tv])

        combined_root = self.QT.union(cx, cy)
        self.Root[combined_root] = new_parent
        self.QBT_parent[new_parent] = -1

        # Store separation time for the new parent node
        self.split_times[new_parent] = sep_time

        return new_parent

def build_trees_from_splits(splits):
    unique_labels = set()
    for (label1, label2), _ in splits.items():
        unique_labels.add(label1)
        unique_labels.add(label2)

    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    biggest_label = max(unique_labels)
    size = len(unique_labels)
    qebt = QEBTUnionFind(size=size)

    for (label1, label2), sep_time in sorted(splits.items(), key=lambda x: x[1], reverse=True):
        idx1 = label_to_index[label1]
        idx2 = label_to_index[label2]
        cx = qebt.find_canonical(idx1)
        cy = qebt.find_canonical(idx2)
        if cx != cy:
            qebt.union(cx, cy, sep_time)

    def construct_forest():
        connections = defaultdict(dict)
        roots = []

        for idx, parent in enumerate(qebt.QBT_parent):
            if parent != -1:
                parent_label = index_to_label.get(parent, parent+biggest_label-size+1)
                child_label = index_to_label.get(idx, idx+biggest_label-size+1)
                connections[parent_label].setdefault("children", []).append(child_label)
                connections[parent_label]["split_time"] = qebt.split_times.get(parent, None)
            else:
                roots.append(index_to_label.get(idx, idx+biggest_label-size+1))

        roots = [root for root in roots if root in connections]

        return dict(connections), roots
    print(qebt.QBT_parent)
    return construct_forest()

def main():
    #mask, vector_field = generate_example_data_small(40)
    mask, vector_field = generate_example_data()

    # Identify connected components using the vector field and CCL
    labels = connected_components(mask, vector_field)
   
    labels=relabel_sequentially(labels)
    print(np.unique(labels))

    centers=find_label_centers(labels, vector_field)
    print(centers)

    labels=merge_close_labels(labels, centers, merge_distance=2)
    # labels=relabel_sequentially(labels)
    # print(np.unique(labels))

    splits=find_label_separation(labels, vector_field, cutoff=20)
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
