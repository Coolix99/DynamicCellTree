import numpy as np
from numba import njit

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

@njit
def merge_close_labels_3D(labels, centers, merge_distance=2):
    """
    Merges labels with centers closer than a specified distance in 3D by updating the `labels` array.
    
    Parameters:
        labels (np.ndarray): 3D labeled image with different connected components.
        centers (dict): Dictionary with labels as keys and (x, y, z) centers as values.
        merge_distance (int): Distance threshold for merging labels.
        
    Returns:
        np.ndarray: Updated label array after merging close labels.
    """
    label_keys = list(centers.keys())
    num_labels = len(label_keys)

    # Iterate over each pair of label centers
    for i in range(num_labels):
        label1 = label_keys[i]
        x1, y1, z1 = centers[label1]

        for j in range(i + 1, num_labels):
            label2 = label_keys[j]
            x2, y2, z2 = centers[label2]
            
            # Calculate distance between centers in 3D
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            
            # Merge if centers are closer than merge_distance
            if distance < merge_distance:
                # Update labels array: replace `label2` with `label1`
                for x in range(labels.shape[0]):
                    for y in range(labels.shape[1]):
                        for z in range(labels.shape[2]):
                            if labels[x, y, z] == label2:
                                labels[x, y, z] = label1

    return labels