import numpy as np
from numpy.testing import assert_array_equal
import pytest
from dynamic_cell_tree.connected_components import connected_components, connected_components_3D

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
    """Test `connected_components` with a simple mask and vector field for both connectivities."""
    # Define a simple binary mask
    mask = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ], dtype=np.int32)

    # Define a vector field directing each component toward its neighbor
    vector_field = np.zeros((2, 4, 4), dtype=np.float32)
    vector_field[:, 0, 0] = [0, 1]
    vector_field[:, 1, 0] = [0, 1]
    vector_field[:, 1, 1] = [-1, 0]
    vector_field[:, 0, 2] = [1, 0]
    vector_field[:, 1, 3] = [1, 0]
    vector_field[:, 2, 3] = [1, 0]
    vector_field[:, 2, 2] = [0, 1]
    vector_field[:, 3, 0] = [0, 1]
    vector_field[:, 3, 1] = [0, 1]
    
    # Expected output labels
    expected_labels = np.array([
        [5, 0, 2, 0],
        [1, 1, 0, 3],
        [0, 0, 3, 3],
        [4, 4, 0, 0]
    ], dtype=np.int32)

    for connectivity in [4, 8]:
        labels = connected_components(mask, vector_field, connectivity=connectivity)
        labels = relabel_sequential(labels)
        assert_array_equal(labels, relabel_sequential(expected_labels))

def test_connected_components_single_component():
    """Test `connected_components` with a single connected component for both connectivities."""
    N = 3
    mask = np.ones((N, N), dtype=np.int32)
    vector_field = np.zeros((2, N, N), dtype=np.float32)
    vector_field[0, :, :] = -1
    vector_field[1, :, :] = 0
    vector_field[0, 0, 0] = 0
    vector_field[1, 0, 0] = 1
    vector_field[0, 0, 1] = 0
    vector_field[1, 0, 1] = 1

    expected_labels = np.ones((N, N), dtype=np.int32)

    for connectivity in [4, 8]:
        labels = connected_components(mask, vector_field, connectivity=connectivity)
        labels = relabel_sequential(labels)
        assert_array_equal(labels, relabel_sequential(expected_labels))

def test_connected_components_large_single_component():
    """Test `connected_components` with a large single connected component for both connectivities."""
    N = 100
    mask = np.ones((N, N), dtype=np.int32)
    vector_field = np.zeros((2, N, N), dtype=np.float32)
    vector_field[0, :, :] = -1
    vector_field[1, :, :] = 0
    vector_field[0, 0, :] = 0
    vector_field[1, 0, :] = 1

    expected_labels = np.ones((N, N), dtype=np.int32)

    for connectivity in [4, 8]:
        labels = connected_components(mask, vector_field, connectivity=connectivity)
        labels = relabel_sequential(labels)
        assert_array_equal(labels, relabel_sequential(expected_labels))

def test_connected_components_disconnected():
    """Test `connected_components` with multiple disconnected components for both connectivities."""
    mask = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.int32)
    vector_field = np.zeros((2, 3, 3), dtype=np.float32)

    expected_labels = np.array([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]
    ], dtype=np.int32)

    for connectivity in [4, 8]:
        labels = connected_components(mask, vector_field, connectivity=connectivity)
        labels = relabel_sequential(labels)
        assert_array_equal(labels, relabel_sequential(expected_labels))

def test_connected_components_different_connectivity():
    """
    Test `connected_components` where the outcome is different for 4-connectivity and 8-connectivity.
    """
    mask = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ], dtype=np.int32)

    vector_field = np.ones((2, 3, 3), dtype=np.float32)

    # Expected result with 4-connectivity
    expected_labels_4 = np.array([
        [1, 1, 0],
        [0, 2, 2],
        [3, 0, 4]
    ], dtype=np.int32)

    # Expected result with 8-connectivity
    expected_labels_8 = np.array([
        [1, 2, 0],
        [0, 1, 2],
        [3, 0, 1]
    ], dtype=np.int32)

    # Test 4-connectivity
    labels_4 = connected_components(mask, vector_field, connectivity=4)
    labels_4 = relabel_sequential(labels_4)
    assert_array_equal(labels_4, relabel_sequential(expected_labels_4))

    # Test 8-connectivity
    labels_8 = connected_components(mask, vector_field, connectivity=8)
    labels_8 = relabel_sequential(labels_8)
    assert_array_equal(labels_8, relabel_sequential(expected_labels_8))

def test_connected_components_3D_x_direction_component():
    """
    Test `connected_components_3D` with components connected only along the X direction.
    Each YZ slice should be independent.
    """
    N = 5
    mask = np.ones((N, N, N), dtype=np.int32)
    vector_field = np.zeros((3, N, N, N), dtype=np.float32)
    
    # Only connect in the X-direction
    vector_field[0, :, :, :] = -1  # X direction
    vector_field[1, :, :, :] = 0   # Y direction
    vector_field[2, :, :, :] = 0   # Z direction

    # Expected labels: Each YZ slice should be independently labeled
    expected_labels = np.zeros((N, N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            expected_labels[:, i, j] = i*N+j+1
    for connectivity in [6, 18]:
        labels = connected_components_3D(mask, vector_field, connectivity=connectivity)
        labels = relabel_sequential(labels)
        print('--')
        print(labels)
        assert_array_equal(labels, relabel_sequential(expected_labels))

def test_connected_components_3D_disconnected():
    """Test `connected_components_3D` with multiple disconnected components for both 6- and 18-connectivity."""
    mask = np.array([
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]]
    ], dtype=np.int32)

    vector_field = np.zeros((3, 3, 3, 3), dtype=np.float32)

    expected_labels = np.array([
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 2, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 3]]
    ], dtype=np.int32)

    for connectivity in [6, 18]:
        labels = connected_components_3D(mask, vector_field, connectivity=connectivity)
        labels = relabel_sequential(labels)
        assert_array_equal(labels, relabel_sequential(expected_labels))


def test_connected_components_3D_different_connectivity():
    """
    Test `connected_components_3D` where the outcome differs for 6- and 18-connectivity.
    """
    mask = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],

        [[1, 0, 0],
         [0, 1, 1],
         [0, 1, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]]
    ], dtype=np.int32)

    vector_field = np.ones((3, 3, 3, 3), dtype=np.float32)

    # Expected result with 6-connectivity
    expected_labels_6 = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],

        [[2, 0, 0],
         [0, 1, 3],
         [0, 4, 5]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 5]]
    ], dtype=np.int32)


    expected_labels_18 = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],

        [[2, 0, 0],
         [0, 3, 4],
         [0, 1, 5]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 4]]
    ], dtype=np.int32)



    # Test 6-connectivity
    labels_6 = connected_components_3D(mask, vector_field, connectivity=6)
    labels_6 = relabel_sequential(labels_6)
    assert_array_equal(labels_6, relabel_sequential(expected_labels_6))

    # Test 18-connectivity
    labels_18 = connected_components_3D(mask, vector_field, connectivity=18)
    labels_18 = relabel_sequential(labels_18)
    assert_array_equal(labels_18, relabel_sequential(expected_labels_18))


if __name__ == "__main__":
    pytest.main()
