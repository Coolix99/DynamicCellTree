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

# @njit
# def get_neighbors_3D(x, y, z, connectivity):
#     """Get neighbors in 3D based on connectivity (6, 18, or 26)."""
#     if connectivity == 6:
#         return [
#             (x + 1, y, z), (x - 1, y, z), (x, y + 1, z), 
#             (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)
#         ]
#     elif connectivity == 18:
#         neighbors = [
#             (x + 1, y, z), (x - 1, y, z), (x, y + 1, z), 
#             (x, y - 1, z), (x, y, z + 1), (x, y, z - 1),
#             (x + 1, y + 1, z), (x + 1, y - 1, z), (x - 1, y + 1, z),
#             (x - 1, y - 1, z), (x + 1, y, z + 1), (x - 1, y, z + 1),
#             (x, y + 1, z + 1), (x, y - 1, z + 1), (x + 1, y, z - 1),
#             (x - 1, y, z - 1), (x, y + 1, z - 1), (x, y - 1, z - 1)
#         ]
#         return neighbors
#     elif connectivity == 26:
#         neighbors = []
#         for dx in [-1, 0, 1]:
#             for dy in [-1, 0, 1]:
#                 for dz in [-1, 0, 1]:
#                     if dx != 0 or dy != 0 or dz != 0:
#                         neighbors.append((x + dx, y + dy, z + dz))
#         return neighbors
    
# @njit
# def compute_progenitors_3D(label, target_x, target_y, target_z, vector_field, labels, cutoff, connectivity):
#     """
#     Perform a DFS to find all progenitor voxels pointing towards the target voxel in 3D.
#     """
#     # Select the direction conversion function based on connectivity
#     if connectivity == 6:
#         convert_to_direction = convert_to_direction_6
#     elif connectivity == 18:
#         convert_to_direction = convert_to_direction_18
#     else:
#         raise ValueError("Unsupported connectivity: Only 6, 18, or 26 are supported in 3D.")

#     stack = [(target_x, target_y, target_z)]
#     visited = np.zeros_like(labels, dtype=np.bool_)
#     depth = 0

#     while stack and depth < cutoff:
#         current_size = len(stack)
#         for _ in range(current_size):
#             current_x, current_y, current_z = stack.pop()
#             if visited[current_x, current_y, current_z]:
#                 continue
#             visited[current_x, current_y, current_z] = True

#             # Retrieve neighbors based on connectivity
#             neighbors = get_neighbors_3D(current_x, current_y, current_z, connectivity)
#             for nx, ny, nz in neighbors:
#                 if (0 <= nx < labels.shape[0] and 0 <= ny < labels.shape[1] and 0 <= nz < labels.shape[2] and labels[nx, ny, nz] == label):
#                     vx, vy, vz = vector_field[0, nx, ny, nz], vector_field[1, nx, ny, nz], vector_field[2, nx, ny, nz]
#                     dx, dy, dz = convert_to_direction(vx, vy, vz)
#                     if nx + dx == current_x and ny + dy == current_y and nz + dz == current_z:
#                         stack.append((nx, ny, nz))
#         depth += 1

#     return depth

# #@njit(parallel=True)
# def find_label_separation_3D(labels, vector_field, cutoff=30, connectivity=6):
#     """
#     Computes a hierarchy of separation times between neighboring labels in 3D.
#     """
#     # Select the direction conversion function based on connectivity
#     if connectivity == 6:
#         convert_to_direction = convert_to_direction_6
#     elif connectivity == 18:
#         convert_to_direction = convert_to_direction_18
#     else:
#         raise ValueError("Unsupported connectivity for 3D.")

#     separation_times = {}

#     for x in prange(labels.shape[0]):
#         for y in range(labels.shape[1]):
#             print(x,y)
#             for z in range(labels.shape[2]):
#                 current_label = labels[x, y, z]
#                 if current_label == 0:
#                     continue
#                 # Iterate over neighbors based on connectivity
#                 neighbors = get_neighbors_3D(x, y, z, connectivity)
#                 for nx, ny, nz in neighbors:
#                     if 0 <= nx < labels.shape[0] and 0 <= ny < labels.shape[1] and 0 <= nz < labels.shape[2]:
#                         neighbor_label = labels[nx, ny, nz]
#                         if neighbor_label != 0 and neighbor_label != current_label:
#                             dx1, dy1, dz1 = convert_to_direction(vector_field[0, x, y, z], vector_field[1, x, y, z], vector_field[2, x, y, z])
#                             dx2, dy2, dz2 = convert_to_direction(vector_field[0, nx, ny, nz], vector_field[1, nx, ny, nz], vector_field[2, nx, ny, nz])
#                             if dx1 == dx2 and dy1 == dy2 and dz1 == dz2:
#                                 continue
            
#                             # Compute progenitor chain lengths for both labels
#                             s1 = compute_progenitors_3D(current_label, x, y, z, vector_field, labels, cutoff, connectivity)
#                             s2 = compute_progenitors_3D(neighbor_label, nx, ny, nz, vector_field, labels, cutoff, connectivity)
                            
#                             separation_time = min(s1, s2)
#                             label_pair = (min(current_label, neighbor_label), max(current_label, neighbor_label))
                            
#                             # Update the maximum separation time for the label pair
#                             if label_pair in separation_times:
#                                 separation_times[label_pair] = max(separation_times[label_pair], separation_time)
#                             else:
#                                 separation_times[label_pair] = separation_time

#     return separation_times


import ctypes
import numpy as np
import os

class SeparationEntry(ctypes.Structure):
    _fields_ = [
        ("label1", ctypes.c_int),
        ("label2", ctypes.c_int),
        ("separation_time", ctypes.c_int)
    ]

class SparseMap(ctypes.Structure):
    _fields_ = [
        ("entries", ctypes.POINTER(SeparationEntry)),  # Pointer to array of entries
        ("count", ctypes.c_size_t),                   # Number of entries
        ("capacity", ctypes.c_size_t)                 # Allocated capacity
    ]

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, 'libseparation.so')
lib = ctypes.CDLL(lib_path)

# Define argument and return types
lib.find_label_separation_3D.argtypes = [
    ctypes.POINTER(ctypes.c_int),  # labels (flat array)
    ctypes.POINTER(ctypes.c_float),  # vector_field (flat array)
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # dim1, dim2, dim3
    ctypes.c_int, ctypes.c_int,  # cutoff, connectivity
]
lib.find_label_separation_3D.restype = SparseMap

lib.free_sparse_map_memory.argtypes = [ctypes.POINTER(SparseMap)]
lib.free_sparse_map_memory.restype = None

def find_label_separation_3D(labels, vector_field, cutoff=30, connectivity=6):
    dim3, dim2, dim1 = labels.shape

    # Flatten arrays and create ctypes pointers
    flat_labels = labels.ravel().astype(np.int32)
    flat_vector_field = vector_field.ravel().astype(np.float32)
    
    sparse_map = lib.find_label_separation_3D(
        flat_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        flat_vector_field.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dim1, dim2, dim3,
        cutoff, connectivity
    )

    # Process the entries in the SparseMap
    separation_dict = {}
    for i in range(sparse_map.count):
        entry = sparse_map.entries[i]
        separation_dict[(entry.label1, entry.label2)] = entry.separation_time

    lib.free_sparse_map_memory(ctypes.byref(sparse_map))

    return separation_dict
