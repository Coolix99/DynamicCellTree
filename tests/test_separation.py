import numpy as np
from numpy.testing import assert_equal
import pytest
from dynamic_cell_tree.separation import find_label_separation,find_label_separation_3D

def test_find_label_separation_4_connectivity():
    """
    Test `find_label_separation` for 4-connectivity in a simple configuration.
    """
    labels = np.array([
        [1, 1, 1, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4]
    ], dtype=np.int32)

    vector_field = np.zeros((2, 5, 5), dtype=np.float32)

    # Assign vector directions that do not cause neighbor overlap
    # Label 1 and Label 2
    vector_field[:, 0, 2] = [0, -1]
    vector_field[:, 0, 3] = [0, 1]


    # Expected separation times between labels
    expected_separation = {
        (1, 2): 1,
    }

    # Run the function
    separation_times = find_label_separation(labels, vector_field, cutoff=10, connectivity=4)

    # Check if the separation times match expected values
    assert separation_times == expected_separation, f"Expected {expected_separation}, but got {separation_times}"

def test_find_label_separation_8_connectivity():
    """
    Test `find_label_separation` for 8-connectivity in a more complex setup.
    """
    labels = np.array([
        [1, 1, 1, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4]
    ], dtype=np.int32)

    vector_field = np.zeros((2, 5, 5), dtype=np.float32)

    # Assign vector directions that do not cause neighbor overlap
    # Label 1 and Label 2
    vector_field[:, 0, 2] = [1, -1]
    vector_field[:, 0, 3] = [1, 1]


    # Expected separation times between labels
    expected_separation = {
        (1, 2): 1,
    }


    separation_times = find_label_separation(labels, vector_field, cutoff=10, connectivity=8)
    assert separation_times == expected_separation, f"Expected {expected_separation}, but got {separation_times}"

def test_find_label_separation_no_overlap():
    """
    Test `find_label_separation` where vectors are configured to prevent any separation.
    """
    labels = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4]
    ], dtype=np.int32)

    vector_field = np.zeros((2, 5, 5), dtype=np.float32)
    
    # Set vectors so they do not separate; they point away
    vector_field[:, 0, 0] = [1, 0]
    vector_field[:, 1, 1] = [-1, 0]
    vector_field[:, 0, 3] = [0, 1]
    vector_field[:, 1, 4] = [0, -1]
    vector_field[:, 3, 0] = [-1, 0]
    vector_field[:, 4, 1] = [1, 0]
    vector_field[:, 3, 4] = [0, -1]
    vector_field[:, 4, 3] = [0, 1]

    expected_separation = {}  # No pairs should separate

    separation_times = find_label_separation(labels, vector_field, cutoff=10, connectivity=4)
    assert separation_times == expected_separation, f"Expected {expected_separation}, but got {separation_times}"

def test_find_label_separation_3D_6_connectivity():
    """
    Test `find_label_separation_3D` for 6-connectivity in a simple configuration.
    """
    labels = np.array([
        [[1, 1, 1, 2, 2],
         [1, 1, 0, 2, 2],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    ], dtype=np.int32)

    vector_field = np.zeros((3, 2, 3, 5), dtype=np.float32)

    # Set vector directions to simulate simple 6-connectivity overlap
    vector_field[:, 0, 0, 2] = [0, 0, -1]  # Label 1 pointing to Label 2
    vector_field[:, 0, 0, 3] = [0, 0, 1]   # Label 2 pointing to Label 1

    # Expected separation times between labels
    expected_separation = {
        (1, 2): 1,
    }

    # Run the function
    separation_times = find_label_separation_3D(labels, vector_field, cutoff=10, connectivity=6)

    # Check if the separation times match expected values
    assert separation_times == expected_separation, f"Expected {expected_separation}, but got {separation_times}"

def test_find_label_separation_3D_18_connectivity():
    """
    Test `find_label_separation_3D` for 18-connectivity in a more complex setup.
    """
    labels = np.array([
        [[1, 1, 1, 2, 2],
         [1, 1, 0, 2, 2],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 4, 0, 0]]
    ], dtype=np.int32)

    vector_field = np.zeros((3, 2, 3, 5), dtype=np.float32)

    # Set vector directions to simulate 18-connectivity overlap
    vector_field[:, 0, 0, 2] = [1, 0, -1]  # Label 1 pointing towards Label 2
    vector_field[:, 0, 0, 3] = [1, 0, 1]   # Label 2 pointing towards Label 1

    # Expected separation times between labels
    expected_separation = {
        (1, 2): 1,
    }

    # Run the function
    separation_times = find_label_separation_3D(labels, vector_field, cutoff=10, connectivity=18)
    assert separation_times == expected_separation, f"Expected {expected_separation}, but got {separation_times}"

def test_find_label_separation_3D_no_overlap():
    """
    Test `find_label_separation_3D` where vectors are configured to prevent any separation.
    """
    labels = np.array([
        [[1, 1, 0, 2, 2],
         [1, 1, 0, 2, 2],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 3, 0, 0]]
    ], dtype=np.int32)

    vector_field = np.zeros((3, 2, 3, 5), dtype=np.float32)
    
    # Set vectors so they do not cause separation; they point away from each other
    vector_field[:, 0, 0, 0] = [1, 0, 0]
    vector_field[:, 1, 0, 1] = [-1, 0, 0]
    vector_field[:, 0, 0, 3] = [0, 1, 0]
    vector_field[:, 1, 1, 4] = [0, -1, 0]
    vector_field[:, 1, 2, 0] = [-1, 0, 0]
    vector_field[:, 0, 2, 1] = [1, 0, 0]
    vector_field[:, 1, 2, 4] = [0, -1, 0]
    vector_field[:, 0, 2, 3] = [0, 1, 0]

    expected_separation = {}  # No pairs should separate

    # Run the function
    separation_times = find_label_separation_3D(labels, vector_field, cutoff=10, connectivity=6)
    assert separation_times == expected_separation, f"Expected {expected_separation}, but got {separation_times}"

# Optional main function to run tests manually
if __name__ == "__main__":
    pytest.main()
