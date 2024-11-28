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
def initialize_union_find(labels):
    """
    Initializes Union-Find structures.

    Parameters:
        labels (np.ndarray): Labeled image.

    Returns:
        parent (np.ndarray): Parent array for Union-Find.
        rank (np.ndarray): Rank array for Union-Find.
    """
    num_labels = np.max(labels)
    parent = np.arange(num_labels, dtype=np.int32)
    rank = np.zeros(num_labels, dtype=np.int32)
    return parent, rank

from scipy.spatial import cKDTree
def determine_merge_relationships_optimized(centers, parent, rank, merge_distance):
    """
    Determines merging relationships between labels using Union-Find and a k-d Tree for efficiency.

    Parameters:
        centers (numba.typed.Dict): Dictionary with labels as keys and (x, y, z) centers as values.
        parent (np.ndarray): Parent array for Union-Find.
        rank (np.ndarray): Rank array for Union-Find.
        merge_distance (int): Distance threshold for merging labels.
    """
    # Convert centers from numba.typed.Dict to numpy array for k-d Tree
    labels = list(centers.keys())
    coordinates = np.array([centers[label] for label in labels], dtype=np.float64)
    
    # Build a 3D k-d tree
    tree = cKDTree(coordinates)
    
    # Query the k-d tree for neighbors within merge_distance
    for i, label1 in enumerate(labels):
        neighbors = tree.query_ball_point(coordinates[i], merge_distance)
        for j in neighbors:
            if i != j:  # Avoid self-merging
                label2 = labels[j]
                union(parent, rank, label1 - 1, label2 - 1)  # Convert to 0-based index

@njit
def apply_merge_map(labels, parent):
    """
    Applies the merge map to the labels array.

    Parameters:
        labels (np.ndarray): Labeled image.
        parent (np.ndarray): Parent array for Union-Find.

    Returns:
        np.ndarray: Updated labels array.
    """
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            for z in range(labels.shape[2]):
                if labels[x, y, z] > 0:
                    labels[x, y, z] = find(parent, labels[x, y, z] - 1) + 1  # Back to 1-based index
    return labels

def merge_close_labels_3D(labels, centers, merge_distance=4):

    parent, rank = initialize_union_find(labels)
    determine_merge_relationships_optimized(centers, parent, rank, merge_distance)
    updated_labels = apply_merge_map(labels, parent)
    return updated_labels

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


    # """
    # Merges labels with centers closer than a specified distance in 3D by updating the `labels` array.
    
    # Parameters:
    #     labels (np.ndarray): 3D labeled image with different connected components.
    #     centers (dict): Dictionary with labels as keys and (x, y, z) centers as values.
    #     merge_distance (int): Distance threshold for merging labels.
        
    # Returns:
    #     np.ndarray: Updated label array after merging close labels.
    # """
    # label_keys = list(centers.keys())
    # num_labels = len(label_keys)
    
    # # Create a dictionary to track which labels should be merged into which target label
    # merge_map = {}
    # # Step 1: Determine merging relationships between labels
    # for i in range(num_labels):
    #     label1 = label_keys[i]
    #     x1, y1, z1 = centers[label1]

    #     for j in range(i + 1, num_labels):
    #         label2 = label_keys[j]
    #         x2, y2, z2 = centers[label2]
            
    #         # Calculate distance between centers in 3D
    #         distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            
    #         # If the centers are within the merge distance, map label2 to label1
    #         if distance < merge_distance:
    #             # Merge smaller label to larger one for consistency
    #             root_label = min(label1, label2)
    #             merged_label = max(label1, label2)
                
    #             # Update merge_map to point merged_label to root_label
    #             merge_map[merged_label] = root_label

    # # Step 2: Apply the merge map to the labels array
    # # We iterate once through the whole labels array and update according to merge_map
    # for x in range(labels.shape[0]):
    #     for y in range(labels.shape[1]):
    #         for z in range(labels.shape[2]):
    #             current_label = labels[x, y, z]
    #             # Follow the chain to get the final root label
    #             while current_label in merge_map:
    #                 current_label = merge_map[current_label]
    #             labels[x, y, z] = current_label

    # return labels


