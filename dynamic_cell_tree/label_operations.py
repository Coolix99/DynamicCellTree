import numpy as np
from numba import njit


def relabel_sequentially(labels):
    """
    Remaps labels in the input array to sequential values starting from 1.
    
    Parameters:
        labels (np.ndarray): A 2D array of integer labels, potentially non-sequential.
        
    Returns:
        np.ndarray: A 2D array with the same shape as `labels`, where each unique label
                    is replaced by a sequential integer starting from 1.
    """
    # Identify unique labels in the array, ignoring background (0)
    unique_labels = np.unique(labels[labels > 0])
    label_mapping = {}

    # Map each unique label to a sequential value starting from 1
    for new_label, old_label in enumerate(unique_labels, start=1):
        label_mapping[old_label] = new_label

    # Create a new array to hold the relabeled output
    relabeled = np.zeros_like(labels, dtype=np.int32)

    # Apply the mapping to create the relabeled output
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if labels[x, y] > 0:
                relabeled[x, y] = label_mapping[labels[x, y]]

    return relabeled

def relabel_sequentially_3D(labels):
    """
    Remaps labels in a 3D array to sequential values starting from 1.
    
    Parameters:
        labels (np.ndarray): A 3D array of integer labels, potentially non-sequential.
        
    Returns:
        np.ndarray: A 3D array with sequential labels starting from 1.
    """
    # Identify unique labels in the array, ignoring background (0)
    unique_labels = np.unique(labels[labels > 0])
    label_mapping = {}

    # Map each unique label to a sequential value starting from 1
    for new_label, old_label in enumerate(unique_labels, start=1):
        label_mapping[old_label] = new_label

    # Create a new array to hold the relabeled output
    relabeled = np.zeros_like(labels, dtype=np.int32)

    # Apply the mapping to create the relabeled output
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            for z in range(labels.shape[2]):
                if labels[x, y, z] > 0:
                    relabeled[x, y, z] = label_mapping[labels[x, y, z]]

    return relabeled
