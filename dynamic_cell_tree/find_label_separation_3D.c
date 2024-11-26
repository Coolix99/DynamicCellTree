#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "sparse_map.h"

// Define a struct for a 3D vector
typedef struct {
    int x1, x2, x3;
} Vector3D;

// Function to get neighbors based on connectivity
void get_neighbors_3D_6(Vector3D *neighbors) {
    neighbors[0] = (Vector3D){1, 0, 0};
    neighbors[1] = (Vector3D){-1, 0, 0};
    neighbors[2] = (Vector3D){0, 1, 0};
    neighbors[3] = (Vector3D){0, -1, 0};
    neighbors[4] = (Vector3D){0, 0, 1};
    neighbors[5] = (Vector3D){0, 0, -1};
    
}

void get_neighbors_3D_18(Vector3D *neighbors) {
    // Populate 18-connectivity neighbors
    neighbors[0] = (Vector3D){1, 0, 0};
    neighbors[1] = (Vector3D){-1, 0, 0};
    neighbors[2] = (Vector3D){0, 1, 0};
    neighbors[3] = (Vector3D){0, -1, 0};
    neighbors[4] = (Vector3D){0, 0, 1};
    neighbors[5] = (Vector3D){0, 0, -1};
    neighbors[6] = (Vector3D){1, 1, 0};
    neighbors[7] = (Vector3D){1, -1, 0};
    neighbors[8] = (Vector3D){-1, 1, 0};
    neighbors[9] = (Vector3D){-1, -1, 0};
    neighbors[10] = (Vector3D){1, 0, 1};
    neighbors[11] = (Vector3D){-1, 0, 1};
    neighbors[12] = (Vector3D){0, 1, 1};
    neighbors[13] = (Vector3D){0, -1, 1};
    neighbors[14] = (Vector3D){1, 0, -1};
    neighbors[15] = (Vector3D){-1, 0, -1};
    neighbors[16] = (Vector3D){0, 1, -1};
    neighbors[17] = (Vector3D){0, -1, -1};
}

// Convert vector field values to a directional step for 6-connectivity
Vector3D convert_to_direction_6(float v1, float v2, float v3) {
    int abs_v1 = abs(v1);
    int abs_v2 = abs(v2);
    int abs_v3 = abs(v3);

    if (abs_v1 >= abs_v2 && abs_v1 >= abs_v3) {
        return (Vector3D){(v1 > 0 ? 1 : -1), 0, 0};
    } else if (abs_v2 >= abs_v1 && abs_v2 >= abs_v3) {
        return (Vector3D){0, (v2 > 0 ? 1 : -1), 0};
    } else {
        return (Vector3D){0, 0, (v3 > 0 ? 1 : -1)};
    }
}

// Convert vector field values to a directional step for 18-connectivity
Vector3D convert_to_direction_18(float v_1, float v_2, float v_3) {
    float v1 = v_1;
    float v2 = (v_1 + v_2) / 1.4142135f;
    float v3 = v_2;
    float v4 = (v_1 - v_2) / 1.4142135f;
    float v5 = (v_1 + v_3) / 1.4142135f;
    float v6 = (v_2 + v_3) / 1.4142135f;
    float v7 = v_3;
    float v8 = (v_1 - v_3) / 1.4142135f;
    float v9 = (v_2 - v_3) / 1.4142135f;

    float max_val = fabsf(v1);
    Vector3D direction = (Vector3D){(v1 > 0 ? 1 : -1), 0, 0};

    if (fabsf(v2) > max_val) {
        max_val = fabsf(v2);
        direction = (Vector3D){(v2 > 0 ? 1 : -1), (v2 > 0 ? 1 : -1), 0};
    }
    if (fabsf(v3) > max_val) {
        max_val = fabsf(v3);
        direction = (Vector3D){0, (v3 > 0 ? 1 : -1), 0};
    }
    if (fabsf(v4) > max_val) {
        max_val = fabsf(v4);
        direction = (Vector3D){(v4 > 0 ? 1 : -1), (v4 > 0 ? -1 : 1), 0};
    }
    if (fabsf(v5) > max_val) {
        max_val = fabsf(v5);
        direction = (Vector3D){(v5 > 0 ? 1 : -1), 0, (v5 > 0 ? 1 : -1)};
    }
    if (fabsf(v6) > max_val) {
        max_val = fabsf(v6);
        direction = (Vector3D){0, (v6 > 0 ? 1 : -1), (v6 > 0 ? 1 : -1)};
    }
    if (fabsf(v7) > max_val) {
        max_val = fabsf(v7);
        direction = (Vector3D){0, 0, (v7 > 0 ? 1 : -1)};
    }
    if (fabsf(v8) > max_val) {
        max_val = fabsf(v8);
        direction = (Vector3D){(v8 > 0 ? 1 : -1), 0, (v8 > 0 ? -1 : 1)};
    }
    if (fabsf(v9) > max_val) {
        direction = (Vector3D){0, (v9 > 0 ? 1 : -1), (v9 > 0 ? -1 : 1)};
    }

    return direction;
}

int compute_progenitors_3D(int *labels, int dim1, int dim2, int dim3, 
                           int target_1, int target_2, int target_3, 
                           float *vector_field, int label, int cutoff, int connectivity) {
    Vector3D stack[1000];  // Adjust stack size if necessary
    bool *visited = calloc(dim1 * dim2 * dim3, sizeof(bool));
    if (!visited) {
        fprintf(stderr, "Memory allocation failed for visited array\n");
        exit(EXIT_FAILURE);
    }

    // Predefine the direction function
    Vector3D (*convert_to_direction)(float, float, float);
    if (connectivity == 6) {
        convert_to_direction = &convert_to_direction_6;
    } else if (connectivity == 18) {
        convert_to_direction = &convert_to_direction_18;
    } else {
        fprintf(stderr, "Unsupported connectivity\n");
        free(visited);
        return -1;
    }

    // Predefine the direction function
    void (*get_neighbors_3D)(Vector3D*);
    if (connectivity == 6) {
        get_neighbors_3D = &get_neighbors_3D_6;
    } else if (connectivity == 18) {
        get_neighbors_3D = &get_neighbors_3D_18;
    } else {
        fprintf(stderr, "Unsupported connectivity\n");
        free(visited);
        return -1;
    }

    int depth = 0, stack_size = 1;
    stack[0] = (Vector3D){target_1, target_2, target_3};

    Vector3D neighbors[connectivity];
    get_neighbors_3D(neighbors);

    while (stack_size > 0 && depth < cutoff) {
        int current_size = stack_size;
        stack_size = 0;

        for (int i = 0; i < current_size; i++) {
            Vector3D pos = stack[i];
            int idx = pos.x3 * dim1 * dim2 + pos.x2 * dim1 + pos.x1;

            if (visited[idx]) continue;
            visited[idx] = true;

            for (int j = 0; j < connectivity; j++) {
                Vector3D neighbor = {
                    pos.x1 + neighbors[j].x1,
                    pos.x2 + neighbors[j].x2,
                    pos.x3 + neighbors[j].x3
                };

                int n1 = neighbor.x1, n2 = neighbor.x2, n3 = neighbor.x3;
                if (n1 >= 0 && n1 < dim1 && n2 >= 0 && n2 < dim2 && n3 >= 0 && n3 < dim3) {
                    int nidx = n3 * dim1 * dim2 + n2 * dim1 + n1;

                    if (labels[nidx] == label && !visited[nidx]) {
                        float v1 = vector_field[0 * dim1 * dim2 * dim3 + nidx];
                        float v2 = vector_field[1 * dim1 * dim2 * dim3 + nidx];
                        float v3 = vector_field[2 * dim1 * dim2 * dim3 + nidx];

                        Vector3D direction = convert_to_direction(v1, v2, v3);

                        if (n1 + direction.x1 == pos.x1 && n2 + direction.x2 == pos.x2 && n3 + direction.x3 == pos.x3) {
                            stack[stack_size++] = neighbor;
                        }
                    }
                }
            }
        }
        depth++;
    }

    free(visited);
    return depth;
}

// Main function to find separation times between labels
SparseMap find_label_separation_3D(int *labels, float *vector_field, int dim1, int dim2, int dim3,
                            int cutoff, int connectivity){
    SparseMap sparse_map;
    init_sparse_map(&sparse_map, 10);  
                            
    Vector3D neighbors[connectivity];
    
    // Predefine the direction function
    void (*get_neighbors_3D)(Vector3D*);
    if (connectivity == 6) {
        get_neighbors_3D = &get_neighbors_3D_6;
    } else if (connectivity == 18) {
        get_neighbors_3D = &get_neighbors_3D_18;
    } else {
        fprintf(stderr, "Unsupported connectivity\n");
        return sparse_map;
    }

    get_neighbors_3D(neighbors);

    // Iterate over each voxel in the 3D grid
    for (int x1 = 0; x1 < dim1; x1++) {
        for (int x2 = 0; x2 < dim2; x2++) {
            for (int x3 = 0; x3 < dim3; x3++) {
                int idx = x3 * dim1 * dim2 + x2 * dim1 + x1;
                int current_label = labels[idx];
                if (current_label == 0) continue;  // Skip unlabeled voxels

                // Iterate over neighbors
                for (int n = 0; n < connectivity; n++) {
                    int n1 = x1 + neighbors[n].x1;
                    int n2 = x2 + neighbors[n].x2;
                    int n3 = x3 + neighbors[n].x3;

                    // Check if neighbor is within bounds
                    if (n1 >= 0 && n1 < dim1 && n2 >= 0 && n2 < dim2 && n3 >= 0 && n3 < dim3) {
                        int nidx = n3 * dim1 * dim2 + n2 * dim1 + n1;
                        int neighbor_label = labels[nidx];

                        // Skip if neighbor has the same label or is unlabeled
                        if (neighbor_label != 0 && neighbor_label != current_label) {
                            // Compute progenitor separation times
                            int s1 = compute_progenitors_3D(labels, dim1, dim2, dim3, x1, x2, x3, vector_field, current_label, cutoff, connectivity);
                            int s2 = compute_progenitors_3D(labels, dim1, dim2, dim3, n1, n2, n3, vector_field, neighbor_label, cutoff, connectivity);
                            int separation_time = (s1 < s2) ? s1 : s2;

                            // Update the sparse map
                            add_or_update_sparse_map(&sparse_map, current_label, neighbor_label, separation_time);
                        }
                    }
                }
            }
        }
    }
    return sparse_map;
}