import numpy as np
from numba import njit


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
                labels[i, j] = find(parent, i * N + j) + 1
    return labels

