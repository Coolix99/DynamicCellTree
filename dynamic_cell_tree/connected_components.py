import numpy as np
from numba import njit, prange

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
def convert_to_direction_4(vx, vy):
    """Converts vector field values to a directional step for 4-connectivity."""
    if abs(vx) > abs(vy):
        return (1, 0) if vx > 0 else (-1, 0)
    else:
        return (0, 1) if vy > 0 else (0, -1)

@njit
def convert_to_direction_8(vx, vy):
    """Converts vector field values to a directional step for 8-connectivity."""
    v1=vx
    v2=(vx+vy)/1.4142135
    v3=vy
    v4=(vx-vy)/1.4142135

    max_val = abs(v1)
    direction = (1, 0) if v1 > 0 else (-1, 0)  # Horizontal direction

    # Check if v2 has a larger absolute value
    if abs(v2) > max_val:
        max_val = abs(v2)
        direction = (1, 1) if v2 > 0 else (-1, -1)  # Diagonal direction (bottom-right or top-left)

    # Check if v3 has a larger absolute value
    if abs(v3) > max_val:
        max_val = abs(v3)
        direction = (0, 1) if v3 > 0 else (0, -1)  # Vertical direction (down or up)

    # Check if v4 has a larger absolute value
    if abs(v4) > max_val:
        direction = (1, -1) if v4 > 0 else (-1, 1)  # Diagonal direction (bottom-left or top-right)

    return direction


@njit(parallel=False)
def initial_labeling_pass_4(mask, vector_field, labels, parent, rank, N):
    """
    Initial parallel pass for labeling. Each row (or segment) is processed independently
    with a unique label range to avoid clashes.
    """
    for i in prange(N):
        for j in range(N):
            if mask[i, j] == 1:
                if labels[i, j] == 0:
                    labels[i, j] = N*N 

                vx, vy = vector_field[0, i, j], vector_field[1, i, j]
                dx, dy = convert_to_direction_4(vx, vy)
                nx, ny = i + dx, j + dy

                # Check if neighbor is within bounds and also in the mask
                if 0 <= nx < N and 0 <= ny < N and mask[nx, ny] == 1:
                    if labels[nx, ny] == 0:
                        labels[nx, ny] = N*N
                    label1 = i * N + j
                    label2 = nx * N + ny
                    union(parent, rank, label1, label2)

@njit(parallel=False)
def initial_labeling_pass_8(mask, vector_field, labels, parent, rank, N):
    """
    Initial parallel pass for labeling. Each row (or segment) is processed independently
    with a unique label range to avoid clashes.
    """
    for i in prange(N):
        for j in range(N):
            if mask[i, j] == 1:
                if labels[i, j] == 0:
                    labels[i, j] = N*N 

                vx, vy = vector_field[0, i, j], vector_field[1, i, j]
                dx, dy = convert_to_direction_8(vx, vy)
                nx, ny = i + dx, j + dy

                # Check if neighbor is within bounds and also in the mask
                if 0 <= nx < N and 0 <= ny < N and mask[nx, ny] == 1:
                    if labels[nx, ny] == 0:
                        labels[nx, ny] = N*N
                    label1 = i * N + j
                    label2 = nx * N + ny
                    union(parent, rank, label1, label2)

@njit(parallel=True)
def resolve_labels(labels, parent, N):
    """
    Resolves labels to ensure consistent labeling after the initial pass.
    """
    for i in prange(N):
        for j in range(N):
            if labels[i, j] != 0:
                labels[i, j] = find(parent, i * N + j) + 1


def connected_components(mask, vector_field, connectivity=4):
    """
    Parallelized connected components using Union-Find with isolated label spaces per row.
    
    Parameters:
        mask (np.ndarray): Binary image (2D array).
        vector_field (np.ndarray): 3D array (2, N, N) for vector directions.
        connectivity (int): Connectivity type, either 4 or 8.
        
    Returns:
        labels (np.ndarray): Labeled image where each component has a unique label.
    """
    N = mask.shape[0]
    labels = np.zeros((N, N), dtype=np.int32)
    parent = np.arange(N * N, dtype=np.int32)  # Flattened parent array
    rank = np.zeros(N * N, dtype=np.int32)
  
    if connectivity == 4:
        initial_labeling_pass = initial_labeling_pass_4
    elif connectivity == 8:
        initial_labeling_pass = initial_labeling_pass_8
    else:
        raise ValueError("Connectivity must be either 4 or 8")
    initial_labeling_pass(mask, vector_field, labels, parent, rank, N)

    # Resolve labels after initial parallel pass
    resolve_labels(labels, parent, N)

    return labels


@njit
def convert_to_direction_6(vx, vy, vz):
    """Converts vector field values to a directional step for 6-connectivity (3D)."""
    # Determine the axis with the largest movement
    abs_vx, abs_vy, abs_vz = abs(vx), abs(vy), abs(vz)
    
    if abs_vx >= abs_vy and abs_vx >= abs_vz:
        return (1, 0, 0) if vx > 0 else (-1, 0, 0)
    elif abs_vy >= abs_vx and abs_vy >= abs_vz:
        return (0, 1, 0) if vy > 0 else (0, -1, 0)
    else:
        return (0, 0, 1) if vz > 0 else (0, 0, -1)

@njit
def convert_to_direction_18(vx, vy, vz):
    """Converts vector field values to a directional step for 18-connectivity (3D)."""
    v1 = vx
    v2 = (vx + vy) / 1.4142135
    v3 = vy
    v4 = (vx - vy) / 1.4142135
    v5 = (vx + vz) / 1.4142135
    v6 = (vy + vz) / 1.4142135
    v7 = vz
    v8 = (vx - vz) / 1.4142135
    v9 = (vy - vz) / 1.4142135
    max_val = abs(v1)
    direction = (1, 0, 0) if v1 > 0 else (-1, 0, 0)

    # Sequentially update direction based on the maximum absolute value
    if abs(v2) > max_val:
        max_val = abs(v2)
        direction = (1, 1, 0) if v2 > 0 else (-1, -1, 0)
    if abs(v3) > max_val:
        max_val = abs(v3)
        direction = (0, 1, 0) if v3 > 0 else (0, -1, 0)
    if abs(v4) > max_val:
        max_val = abs(v4)
        direction = (1, -1, 0) if v4 > 0 else (-1, 1, 0)
    if abs(v5) > max_val:
        max_val = abs(v5)
        direction = (1, 0, 1) if v5 > 0 else (-1, 0, -1)
    if abs(v6) > max_val:
        max_val = abs(v6)
        direction = (0, 1, 1) if v6 > 0 else (0, -1, -1)
    if abs(v7) > max_val:
        max_val = abs(v7)
        direction = (0, 0, 1) if v7 > 0 else (0, 0, -1)
    if abs(v8) > max_val:
        max_val = abs(v8)
        direction = (1, 0, -1) if v8 > 0 else (-1, 0, 1)
    if abs(v9) > max_val:
        direction = (0, 1, -1) if v9 > 0 else (0, -1, 1)

    return direction

@njit(parallel=False)
def initial_labeling_pass_6(mask, vector_field, labels, parent, rank, N):
    """Initial pass for 3D labeling using 6-connectivity."""
    for i in prange(N):
        for j in range(N):
            for k in range(N):
                if mask[i, j, k] == 1:
                    if labels[i, j, k] == 0:
                        labels[i, j, k] = N * N * N  # Initial label assignment

                    vx, vy, vz = vector_field[0, i, j, k], vector_field[1, i, j, k], vector_field[2, i, j, k]
                    dx, dy, dz = convert_to_direction_6(vx, vy, vz)
                    nx, ny, nz = i + dx, j + dy, k + dz

                    # Check if neighbor is within bounds and also in the mask
                    if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N and mask[nx, ny, nz] == 1:
                        if labels[nx, ny, nz] == 0:
                            labels[nx, ny, nz] = N * N * N
                        label1 = i * N * N + j * N + k
                        label2 = nx * N * N + ny * N + nz
                        union(parent, rank, label1, label2)

@njit(parallel=False)
def initial_labeling_pass_18(mask, vector_field, labels, parent, rank, N):
    """Initial pass for 3D labeling using 18-connectivity."""
    for i in prange(N):
        for j in range(N):
            for k in range(N):
                if mask[i, j, k] == 1:
                    if labels[i, j, k] == 0:
                        labels[i, j, k] = N * N * N  # Initial label assignment

                    vx, vy, vz = vector_field[0, i, j, k], vector_field[1, i, j, k], vector_field[2, i, j, k]
                    dx, dy, dz = convert_to_direction_18(vx, vy, vz)
                    nx, ny, nz = i + dx, j + dy, k + dz

                    # Check if neighbor is within bounds and also in the mask
                    if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N and mask[nx, ny, nz] == 1:
                        if labels[nx, ny, nz] == 0:
                            labels[nx, ny, nz] = N * N * N
                        label1 = i * N * N + j * N + k
                        label2 = nx * N * N + ny * N + nz
                        union(parent, rank, label1, label2)

@njit(parallel=True)
def resolve_labels_3D(labels, parent, N):
    """Resolve labels in 3D after the initial parallel pass."""
    for i in prange(N):
        for j in range(N):
            for k in range(N):
                if labels[i, j, k] != 0:
                    labels[i, j, k] = find(parent, i * N * N + j * N + k) + 1

def connected_components_3D(mask, vector_field, connectivity=6):
    """
    3D version of connected components with Union-Find.

    Parameters:
        mask (np.ndarray): Binary image (3D array).
        vector_field (np.ndarray): 3D array (3, N, N, N) for vector directions.
        connectivity (int): Connectivity type, either 6 or 18.
        
    Returns:
        labels (np.ndarray): Labeled 3D image where each component has a unique label.
    """
    N = mask.shape[0]
    labels = np.zeros((N, N, N), dtype=np.int32)
    parent = np.arange(N * N * N, dtype=np.int32)
    rank = np.zeros(N * N * N, dtype=np.int32)

    if connectivity == 6:
        initial_labeling_pass = initial_labeling_pass_6
    elif connectivity == 18:
        initial_labeling_pass = initial_labeling_pass_18
    else:
        raise ValueError("Connectivity must be either 6 or 18")

    initial_labeling_pass(mask, vector_field, labels, parent, rank, N)
    resolve_labels_3D(labels, parent, N)
    return labels
