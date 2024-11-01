import numpy as np
from numpy.testing import assert_array_equal
import pytest
from dynamic_cell_tree.connected_components import connected_components  # Adjust import based on your module structure

def relabel_sequential(labels):
    """
    Relabels connected component labels sequentially starting from 1,
    in the order they first appear in the flattened array.
    
    Parameters:
        labels (np.ndarray): A labeled image array with non-sequential labels.

    Returns:
        np.ndarray: A relabeled array with labels starting from 1, in the order of appearance.
    """
    # Flatten the labels and get unique labels in the order they appear, excluding 0
    flat_labels = labels.flatten()
    unique_labels = [label for label in flat_labels if label != 0]
    
    # Maintain the order of first appearance in the flattened array
    seen = set()
    ordered_labels = []
    for label in unique_labels:
        if label not in seen:
            ordered_labels.append(label)
            seen.add(label)

    # Create a mapping from old labels to new sequential labels based on order of appearance
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(ordered_labels, start=1)}
    
    # Apply the mapping to create a new array with sequential labels
    relabeled = np.zeros_like(labels, dtype=np.int32)
    for old_label, new_label in label_mapping.items():
        relabeled[labels == old_label] = new_label
    
    return relabeled

def test_connected_components_basic():
    """
    Test `connected_components` with a simple mask and vector field.
    It checks if the function correctly identifies components with 
    different connectivity directions.
    """

    # Define a simple binary mask
    mask = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ], dtype=np.int32)

    # Define a vector field directing each component toward its neighbor
    vector_field = np.zeros((2, 4, 4), dtype=np.float32)
    
    # Manually set vectors to point toward neighbors within each component
    # Component 1
    vector_field[:, 0, 0] = [0, 1]
    vector_field[:, 1, 0] = [0, 1]
    vector_field[:, 1, 1] = [-1, 0]
    
    # Component 2
    vector_field[:, 0, 2] = [1, 0]
    
    # Component 3
    vector_field[:, 1, 3] = [1, 0]
    vector_field[:, 2, 3] = [1, 0]
    vector_field[:, 2, 2] = [0, 1]
    
    # Component 4
    vector_field[:, 3, 0] = [0, 1]
    vector_field[:, 3, 1] = [0, 1]
    
    # Expected output labels
    expected_labels = np.array([
        [5, 0, 2, 0],
        [1, 1, 0, 3],
        [0, 0, 3, 3],
        [4, 4, 0, 0]
    ], dtype=np.int32)

    # Run the function
    labels = connected_components(mask, vector_field)
    labels = relabel_sequential(labels)
    
    # Relabel the expected output to ensure consistency in label comparison
    expected_labels = relabel_sequential(expected_labels)

    # Check if the relabeled output matches the expected labels
    assert_array_equal(labels, expected_labels)

def test_connected_components_single_component():
    """
    Test `connected_components` with a single connected component.
    The entire mask is filled, so the entire grid should be one component.
    """

    # Single component binary mask
    mask = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.int32)

    # Vector field pointing all towards the top-left corner
    vector_field = np.zeros((2, 3, 3), dtype=np.float32)
    vector_field[0, :, :] = -1
    vector_field[1, :, :] = 0

    vector_field[0, 0, 0] = 0
    vector_field[1, 0, 0] = 1
    vector_field[0, 0, 1] = 0
    vector_field[1, 0, 1] = 1

    # Expected labels: a single component labeled as 1
    expected_labels = np.ones((3, 3), dtype=np.int32)

    # Run the function and relabel the output
    labels = connected_components(mask, vector_field)
    labels = relabel_sequential(labels)
    
    # Relabel the expected output to ensure consistency in label comparison
    expected_labels = relabel_sequential(expected_labels)

    # Check if the relabeled output matches the expected labels
    assert_array_equal(labels, expected_labels)

def test_connected_components_disconnected():
    """
    Test `connected_components` with multiple disconnected components.
    Each component should be uniquely labeled.
    """

    # Define a disconnected binary mask
    mask = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.int32)

    # Vector field is irrelevant as there are no connections
    vector_field = np.zeros((2, 3, 3), dtype=np.float32)

    # Expected labels: each '1' in mask is a separate component
    expected_labels = np.array([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]
    ], dtype=np.int32)

    # Run the function and relabel the output
    labels = connected_components(mask, vector_field)
    labels = relabel_sequential(labels)

    # Relabel the expected output to ensure consistency in label comparison
    expected_labels = relabel_sequential(expected_labels)

    # Check if the relabeled output matches the expected labels
    assert_array_equal(labels, expected_labels)

# Optional main function to run tests manually
if __name__ == "__main__":
    pytest.main()
