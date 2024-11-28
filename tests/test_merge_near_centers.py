import numpy as np
from numpy.testing import assert_array_equal
import pytest
from numba.typed import Dict
from numba import types
from  dynamic_cell_tree.merge_near_centers import merge_close_labels, merge_close_labels_3D

def type_dict(dict,n):
    # Centers for the components
    key_type = types.int64
    value_type = types.UniTuple(types.int64, n)  # For (x, y) tuples
    centers_typed = Dict.empty(key_type, value_type)
    for key, value in dict.items():
        centers_typed[key] = value
    return centers_typed

def test_merge_close_labels_2D():
    """Test `merge_close_labels` for merging labels based on 2D distance."""
    # Labels array with two components
    labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 2],
        [0, 0, 2, 2]
    ], dtype=np.int32)
    
    
    centers = {
        1: (0, 0),  # Center of label 1
        2: (1, 3)   # Center of label 2
    }
    centers=type_dict(centers,2)
    
    # Merge distance threshold
    merge_distance = 10
    
    # Expected result after merging
    expected_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 1]
    ], dtype=np.int32)
    
    # Run the function
    merged_labels = merge_close_labels(labels, centers,merge_distance)
    
    # Check if the result matches the expected output
    assert_array_equal(merged_labels, expected_labels)

def test_merge_close_labels_3D():
    """Test `merge_close_labels_3D` for merging labels based on 3D distance."""
    # 3D Labels array with two components
    labels = np.array([
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 2],
            [0, 0, 2],
            [0, 0, 2]
        ]
    ], dtype=np.int32)
    
    # Centers for the components
    centers = {
        1: (0, 0, 0),  # Center of label 1
        2: (1, 2, 2)   # Center of label 2
    }
    centers=type_dict(centers,3)
    # Merge distance threshold
    merge_distance = 4
    
    # Expected result after merging
    expected_labels = np.array([
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]
    ], dtype=np.int32)
    
    # Run the function
    merged_labels = merge_close_labels_3D(labels, centers, merge_distance=merge_distance)
    
    # Check if the result matches the expected output
    assert_array_equal(merged_labels, expected_labels)

def test_merge_close_labels_no_merge():
    """Test `merge_close_labels` with no merging."""
    labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 2],
        [0, 0, 2, 2]
    ], dtype=np.int32)
    
    centers = {
        1: (0, 0),
        2: (1, 3)
    }
    centers=type_dict(centers,2)

    merge_distance = 1  # Too small to merge
    
    expected_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 2],
        [0, 0, 2, 2]
    ], dtype=np.int32)
    
    merged_labels = merge_close_labels(labels, centers, merge_distance=merge_distance)
    assert_array_equal(merged_labels, expected_labels)

def test_merge_close_labels_3D_no_merge():
    """Test `merge_close_labels_3D` with no merging."""
    labels = np.array([
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 2],
            [0, 0, 2],
            [0, 0, 2]
        ]
    ], dtype=np.int32)
    
    centers = {
        1: (0, 0, 0),
        2: (1, 2, 2)
    }
    centers=type_dict(centers,3)

    merge_distance = 1  # Too small to merge
    
    expected_labels = np.array([
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 2],
            [0, 0, 2],
            [0, 0, 2]
        ]
    ], dtype=np.int32)
    
    merged_labels = merge_close_labels_3D(labels, centers, merge_distance=merge_distance)
    assert_array_equal(merged_labels, expected_labels)

if __name__ == "__main__":
    pytest.main()
