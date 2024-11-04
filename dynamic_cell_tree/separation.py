import numpy as np
from numba import njit, prange,int64,types
from numba.typed import Dict
from typing import List, Tuple
from dynamic_cell_tree.connected_components import (
    convert_to_direction_4, convert_to_direction_8, 
    convert_to_direction_6, convert_to_direction_18
)

@njit
def get_neighbors_2D(x, y, connectivity):
    """Get neighbors in 2D based on connectivity (4 or 8)."""
    if connectivity == 4:
        return [(x + 1, y),(x, y + 1), (x - 1, y) , (x, y - 1)]
    elif connectivity == 8:
        return [(x + 1, y),(x, y + 1),(x + 1, y + 1), (x + 1, y - 1) , (x - 1, y),(x, y - 1),(x - 1, y - 1), (x - 1, y + 1)]

@njit
def compute_progenitors(label, target_x, target_y, vector_field, labels, cutoff, connectivity):
    """
    Performs a DFS to find all progenitor pixels pointing towards the target pixel.
    """
    # Select the direction conversion function based on connectivity
    if connectivity == 4:
        convert_to_direction = convert_to_direction_4
    elif connectivity == 8:
        convert_to_direction = convert_to_direction_8
    else:
        raise ValueError("Unsupported connectivity: Only 4 or 8 supported in 2D.")

    stack = [(target_x, target_y)]
    visited = np.zeros_like(labels, dtype=np.bool_)
    depth = 0

    while stack and depth < cutoff:
        current_size = len(stack)  
        for _ in range(current_size):
            current_x, current_y = stack.pop()
            if visited[current_x, current_y]:
                continue
            visited[current_x, current_y] = True

            # Retrieve neighbors based on connectivity
            neighbors = get_neighbors_2D(current_x, current_y, connectivity)
            for nx, ny in neighbors:
                if (0 <= nx < labels.shape[0] and 0 <= ny < labels.shape[1] and labels[nx, ny] == label):
                    vx, vy = vector_field[0, nx, ny], vector_field[1, nx, ny]
                    dx, dy = convert_to_direction(vx, vy)
                    if nx + dx == current_x and ny + dy == current_y:
                        stack.append((nx, ny))
        depth += 1

    return depth

#@njit(parallel=True)
def find_label_separation(labels, vector_field, cutoff=30, connectivity=4):
    """
    Computes a hierarchy of separation times between neighboring labels.
    """
    # Select direction conversion function based on connectivity
    if connectivity == 4:
        convert_to_direction = convert_to_direction_4
    elif connectivity == 8:
        convert_to_direction = convert_to_direction_8
    else:
        raise ValueError("Unsupported connectivity: Only 4 or 8 supported in 2D.")

    # separation_times = Dict.empty(
    #     key_type=(int64, int64), 
    #     value_type=int64
    # )
    separation_times={}

    for x in prange(labels.shape[0]):
        for y in range(labels.shape[1]):
            current_label = labels[x, y]
            if current_label == 0:
                continue
            # Iterate over neighbors based on connectivity
            neighbors = get_neighbors_2D(x, y, connectivity)[0:connectivity//2]
            for nx, ny in neighbors:
                if 0 <= nx < labels.shape[0] and 0 <= ny < labels.shape[1]:
                    neighbor_label = labels[nx, ny]
                    if neighbor_label != 0 and neighbor_label != current_label:
                        dx1, dy1 = convert_to_direction(vector_field[0, x, y], vector_field[1, x, y])
                        dx2, dy2 = convert_to_direction(vector_field[0, nx, ny], vector_field[1, nx, ny])
                        if dx1 == dx2 and dy1 == dy2:
                            continue
            
                        # Compute progenitor chain lengths for both labels
                        s1 = compute_progenitors(current_label, x, y, vector_field, labels, cutoff, connectivity)
                        s2 = compute_progenitors(neighbor_label, nx, ny, vector_field, labels, cutoff, connectivity)
                        
                        separation_time = min(s1, s2)
                        label_pair = (min(current_label, neighbor_label), max(current_label, neighbor_label))
                        
                        # Update the maximum separation time for the label pair
                        if label_pair in separation_times:
                            separation_times[label_pair] = max(separation_times[label_pair], separation_time)
                        else:
                            separation_times[label_pair] = separation_time

    return separation_times
