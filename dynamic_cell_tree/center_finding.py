import numpy as np
from numba import njit
from dynamic_cell_tree.connected_components import convert_to_direction_4,convert_to_direction_8,convert_to_direction_6,convert_to_direction_18

@njit
def find_label_centers(labels, vector_field, distance_threshold=2, steps_check=5,connectivity=4):
    """
    Finds the 'singular' center for each label by following vector directions until the position gets stuck.
    
    Parameters:
        labels (np.ndarray): Labeled image with different connected components.
        vector_field (np.ndarray): The vector field indicating directions.
        distance_threshold (int): Distance threshold to determine if the walk has "stuck".
        steps_check (int): Number of steps to check if the movement has reduced.
        
    Returns:
        dict: Dictionary with keys as labels and values as (x, y) center coordinates.
    """
    visited = set()
    centers = {}

    if connectivity == 4:
        convert_to_direction = convert_to_direction_4
    elif connectivity == 8:
        convert_to_direction = convert_to_direction_8
    else:
        raise ValueError("Connectivity must be either 4 or 8")

    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            current_label = np.int64(labels[x, y])
            if current_label == 0 or current_label in visited:
                continue  # Skip background or already processed labels
            
            # Initialize the starting point for this label
            current_x, current_y = x, y
            last_positions = [(current_x, current_y)]  # Track last positions for distance check
            
            while True:
                # Get the direction from the vector field
                vx, vy = vector_field[0, current_x, current_y], vector_field[1, current_x, current_y]
                dx, dy = convert_to_direction(vx, vy)
                next_x, next_y = current_x + dx, current_y + dy
                
                # Check if the next position is within bounds and belongs to the same label
                if (next_x < 0 or next_x >= labels.shape[0] or 
                    next_y < 0 or next_y >= labels.shape[1] or
                    labels[next_x, next_y] != current_label):
                    break  # Stop if out of bounds or label mismatch
                
                # Update positions
                current_x, current_y = next_x, next_y
                last_positions.append((current_x, current_y))
                
                # Maintain only the last `steps_check` positions
                if len(last_positions) > steps_check:
                    last_positions.pop(0)
                
                    # Check if the position is "stuck"
                    x1, y1 = last_positions[0]
                    x2, y2 = last_positions[-1]
                    total_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    if total_distance < distance_threshold:
                        break
            
            # Record the center of the label
            centers[current_label] = (current_x, current_y)
            visited.add(current_label)

    return centers

@njit
def find_label_centers_3D(labels, vector_field, distance_threshold=3, steps_check=7, connectivity=6, dt=0.3):
    """
    Finds the 'singular' center for each label in a 3D labeled array by following vector directions until the position gets stuck.
    
    Parameters:
        labels (np.ndarray): Labeled 3D image with different connected components.
        vector_field (np.ndarray): 4D array representing the vector field with directions.
        distance_threshold (int): Distance threshold to determine if the walk has "stuck".
        steps_check (int): Number of steps to check if the movement has reduced.
        connectivity (int): Connectivity type, either 6 or 18.
        
    Returns:
        dict: Dictionary with keys as labels and values as (x, y, z) center coordinates.
    """
    visited = set()
    centers = {}

    # Set the direction conversion function based on connectivity
    if connectivity == 6:
        convert_to_direction = convert_to_direction_6
    elif connectivity == 18:
        convert_to_direction = convert_to_direction_18
    else:
        raise ValueError("Connectivity must be either 6 or 18")

    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            for z in range(labels.shape[2]):
                current_label = np.int64(labels[x, y, z])
                if current_label == 0 or current_label in visited:
                    continue  # Skip background or already processed labels
                
                # Initialize the starting point for this label
                current_x, current_y, current_z = x, y, z
                last_positions = [(current_x, current_y, current_z)]  # Track last positions for distance check
                while True:
                    # Get the direction from the vector field
                    vx, vy, vz = vector_field[0, current_x, current_y, current_z], vector_field[1, current_x, current_y, current_z], vector_field[2, current_x, current_y, current_z]
                    dx, dy, dz = convert_to_direction(vx, vy, vz)
                    next_x, next_y, next_z = current_x + dx, current_y + dy, current_z + dz
                    # Check if the next position is within bounds and belongs to the same label
                    if (next_x < 0 or next_x >= labels.shape[0] or 
                        next_y < 0 or next_y >= labels.shape[1] or
                        next_z < 0 or next_z >= labels.shape[2] or
                        labels[next_x, next_y, next_z] != current_label):
                        break  # Stop if out of bounds or label mismatch
                    
                    # Update positions
                    current_x, current_y, current_z = next_x, next_y, next_z
                    last_positions.append((current_x, current_y, current_z))
                    
                    # Maintain only the last `steps_check` positions
                    if len(last_positions) > steps_check:
                        last_positions.pop(0)
                    
                        # Check if the position is "stuck"
                        x1, y1, z1 = last_positions[0]
                        x2, y2, z2 = last_positions[-1]
                        total_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
                        
                        if total_distance < distance_threshold:
                            break
                
                refined_x, refined_y, refined_z = float(current_x), float(current_y), float(current_z)
                for i in range(20):
                    int_x, int_y, int_z = int(refined_x + 0.5), int(refined_y + 0.5), int(refined_z + 0.5)
                    vx, vy, vz = vector_field[0, int_x, int_y, int_z], vector_field[1, int_x, int_y, int_z], vector_field[2, int_x, int_y, int_z]
                    dx, dy, dz = dt * vx, dt * vy, dt * vz
                    refined_x, refined_y, refined_z = refined_x + dx, refined_y + dy, refined_z + dz

                # Record the center of the label
                centers[current_label] = (refined_x, refined_y, refined_z)
                visited.add(current_label)

    return centers

