import numpy as np
from numpy.testing import assert_equal
import pytest
from dynamic_cell_tree.center_finding import find_label_centers,find_label_centers_3D

def test_find_label_centers():
    """
    Test `find_label_centers` by setting up labeled regions with clear directional vectors
    that point to known center locations.
    """
    # Define a simple label matrix
    labels = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4]
    ], dtype=np.int32)

    # Define a vector field where each component points towards its center
    vector_field = np.zeros((2, 5, 5), dtype=np.float32)
    
    # Define the vector directions
    # Component 1 points to (0, 0)
    vector_field[:, 0, 0] = [0, 0]
    vector_field[:, 0, 1] = [0, -1]
    vector_field[:, 1, 0] = [-1, 0]
    vector_field[:, 1, 1] = [-1, -1]

    # Component 2 points to (0, 4)
    vector_field[:, 0, 3] = [0, 1]
    vector_field[:, 0, 4] = [0, 0]
    vector_field[:, 1, 3] = [-1, 1]
    vector_field[:, 1, 4] = [-1, 0]

    # Component 3 points to (4, 0)
    vector_field[:, 3, 0] = [1, 0]
    vector_field[:, 3, 1] = [1, -1]
    vector_field[:, 4, 0] = [0, 0]
    vector_field[:, 4, 1] = [0, -1]

    # Component 4 points to (4, 4)
    vector_field[:, 3, 3] = [1, 1]
    vector_field[:, 3, 4] = [1, 0]
    vector_field[:, 4, 3] = [0, 1]
    vector_field[:, 4, 4] = [0, 0]

    # Expected centers for each label
    expected_centers = {
        1: (0, 0),
        2: (0, 4),
        3: (4, 0),
        4: (4, 3) #becuase of oszillation
    }

    # Run the center finding function
    centers = find_label_centers(labels, vector_field)

    # Check if the detected centers match expected values
    assert centers == expected_centers, f"Expected centers {expected_centers}, but got {centers}"

def test_find_label_centers_3D():
    """
    Test `find_label_centers_3D` by setting up labeled regions with clear directional vectors
    that point to known center locations in 3D space.
    """
    # Define a simple 3D label matrix
    labels = np.array([
        [[1, 1, 0, 2, 2],
         [1, 1, 0, 2, 2],
         [0, 0, 0, 0, 0],
         [3, 3, 0, 4, 4],
         [3, 3, 0, 4, 4]],

        [[1, 1, 0, 2, 2],
         [1, 1, 0, 2, 2],
         [0, 0, 0, 0, 0],
         [3, 3, 0, 4, 4],
         [3, 3, 0, 4, 4]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    ], dtype=np.int32).swapaxes(0,2)

    # Define a vector field where each component points towards its center
    vector_field = np.zeros((3, 5, 5, 3), dtype=np.float32)
    # Define the vector directions for each component in 3D
    # Component 1 points to (0, 0, 0)
    vector_field[:, 0, 0, 0] = [0, 0, 0]
    vector_field[:, 0, 1, 0] = [0, -1, 0]
    vector_field[:, 1, 0, 0] = [-1, 0, 0]
    vector_field[:, 1, 1, 0] = [-1, -1, 0]
    vector_field[:, 0, 0, 1] = [0, 0, -1]
    vector_field[:, 0, 1, 1] = [0, 0, -1]
    vector_field[:, 1, 0, 1] = [0, 0, -1]
    vector_field[:, 1, 1, 1] = [0, 0, -1]

    # Component 3 points to (0, 4, 0)
    vector_field[:, 0, 3, 0] = [0, 1, 0]
    vector_field[:, 0, 4, 0] = [0, 0, 0]
    vector_field[:, 1, 3, 0] = [-1, 1, 0]
    vector_field[:, 1, 4, 0] = [-1, 0, 0]
    vector_field[:, 0, 3, 1] = [0, 0, -1]
    vector_field[:, 0, 4, 1] = [0, 0, -1]
    vector_field[:, 1, 3, 1] = [0, 0, -1]
    vector_field[:, 1, 4, 1] = [0, 0, -1]

    # Component 2 points to (4, 0, 0)
    vector_field[:, 3, 0, 0] = [1, 0, 0]
    vector_field[:, 3, 1, 0] = [1, -1, 0]
    vector_field[:, 4, 0, 0] = [0, 0, 0]
    vector_field[:, 4, 1, 0] = [0, -1, 0]
    vector_field[:, 3, 0, 1] = [0, 0, -1]
    vector_field[:, 3, 1, 1] = [0, 0, -1]
    vector_field[:, 4, 0, 1] = [0, 0, -1]
    vector_field[:, 4, 1, 1] = [0, 0, -1]

    # Component 4 points to (4, 4, 0)
    vector_field[:, 3, 3, 0] = [1, 1, 0]
    vector_field[:, 3, 4, 0] = [1, 0, 0]
    vector_field[:, 4, 3, 0] = [0, 1, 0]
    vector_field[:, 4, 4, 0] = [0, 0, 0]
    vector_field[:, 3, 3, 1] = [0, 0, -1]
    vector_field[:, 3, 4, 1] = [0, 0, -1]
    vector_field[:, 4, 3, 1] = [0, 0, -1]
    vector_field[:, 4, 4, 1] = [0, 0, -1]

    # Expected centers for each label
    expected_centers = {
        1: (0, 0, 0),
        3: (0, 4, 0),
        2: (3, 0, 0), #becasue of oscillation
        4: (4, 4, 0)
    }

    # Run the center finding function
    centers = find_label_centers_3D(labels, vector_field)

    # Check if the detected centers match expected values
    assert centers == expected_centers, f"Expected centers {expected_centers}, but got {centers}"

# Optional main function to run tests manually
if __name__ == "__main__":
    pytest.main()
