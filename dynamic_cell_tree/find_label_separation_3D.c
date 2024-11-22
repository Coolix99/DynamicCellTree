#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// Define a struct for a 3D vector
typedef struct {
    int x, y, z;
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
Vector3D convert_to_direction_6(float vx, float vy, float vz) {
    int abs_vx = abs(vx);
    int abs_vy = abs(vy);
    int abs_vz = abs(vz);

    if (abs_vx >= abs_vy && abs_vx >= abs_vz) {
        return (Vector3D){(vx > 0 ? 1 : -1), 0, 0};
    } else if (abs_vy >= abs_vx && abs_vy >= abs_vz) {
        return (Vector3D){0, (vy > 0 ? 1 : -1), 0};
    } else {
        return (Vector3D){0, 0, (vz > 0 ? 1 : -1)};
    }
}

// Convert vector field values to a directional step for 18-connectivity
Vector3D convert_to_direction_18(float vx, float vy, float vz) {
    float v1 = vx;
    float v2 = (vx + vy) / 1.4142135f;
    float v3 = vy;
    float v4 = (vx - vy) / 1.4142135f;
    float v5 = (vx + vz) / 1.4142135f;
    float v6 = (vy + vz) / 1.4142135f;
    float v7 = vz;
    float v8 = (vx - vz) / 1.4142135f;
    float v9 = (vy - vz) / 1.4142135f;

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

int compute_progenitors_3D(int *labels, int x_dim, int y_dim, int z_dim, 
                           int target_x, int target_y, int target_z, 
                           float *vector_field, int label, int cutoff, int connectivity) {
    Vector3D stack[1000];  // Adjust stack size if necessary
    bool *visited = calloc(x_dim * y_dim * z_dim, sizeof(bool));
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
    stack[0] = (Vector3D){target_x, target_y, target_z};

    Vector3D neighbors[connectivity];
    get_neighbors_3D(neighbors);

    while (stack_size > 0 && depth < cutoff) {
        int current_size = stack_size;
        stack_size = 0;

        for (int i = 0; i < current_size; i++) {
            Vector3D pos = stack[i];
            int idx = pos.z * x_dim * y_dim + pos.y * x_dim + pos.x;

            if (visited[idx]) continue;
            visited[idx] = true;

            for (int j = 0; j < connectivity; j++) {
                Vector3D neighbor = {
                    pos.x + neighbors[j].x,
                    pos.y + neighbors[j].y,
                    pos.z + neighbors[j].z
                };

                int nx = neighbor.x, ny = neighbor.y, nz = neighbor.z;
                if (nx >= 0 && nx < x_dim && ny >= 0 && ny < y_dim && nz >= 0 && nz < z_dim) {
                    int nidx = nz * x_dim * y_dim + ny * x_dim + nx;

                    if (labels[nidx] == label && !visited[nidx]) {
                        float vx = vector_field[0 * x_dim * y_dim * z_dim + nidx];
                        float vy = vector_field[1 * x_dim * y_dim * z_dim + nidx];
                        float vz = vector_field[2 * x_dim * y_dim * z_dim + nidx];

                        Vector3D direction = convert_to_direction(vx, vy, vz);

                        if (nx + direction.x == pos.x && ny + direction.y == pos.y && nz + direction.z == pos.z) {
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
void find_label_separation_3D(int *labels, float *vector_field, int x_dim, int y_dim, int z_dim,
                              int cutoff, int connectivity, int *separation_times) {
    Vector3D neighbors[connectivity];
    // Predefine the direction function

    void (*get_neighbors_3D)(Vector3D*);
    if (connectivity == 6) {
        get_neighbors_3D = &get_neighbors_3D_6;
    } else if (connectivity == 18) {
        get_neighbors_3D = &get_neighbors_3D_18;
    } else {
        fprintf(stderr, "Unsupported connectivity\n");
    }

    get_neighbors_3D(neighbors);

    for (int x = 0; x < x_dim; x++) {
        for (int y = 0; y < y_dim; y++) {
            for (int z = 0; z < z_dim; z++) {
                int idx = z * x_dim * y_dim + y * x_dim + x;
                int current_label = labels[idx];
                if (current_label == 0) continue;

                for (int n = 0; n < connectivity; n++) {
                    int nx = x + neighbors[n].x;
                    int ny = y + neighbors[n].y;
                    int nz = z + neighbors[n].z;
                    if (nx >= 0 && nx < x_dim && ny >= 0 && ny < y_dim && nz >= 0 && nz < z_dim) {
                        int nidx = nz * x_dim * y_dim + ny * x_dim + nx;
                        int neighbor_label = labels[nidx];

                        if (neighbor_label != 0 && neighbor_label != current_label) {
                            int s1 = compute_progenitors_3D(labels, x_dim, y_dim, z_dim, x, y, z, vector_field, current_label, cutoff, connectivity);
                            int s2 = compute_progenitors_3D(labels, x_dim, y_dim, z_dim, nx, ny, nz, vector_field, neighbor_label, cutoff, connectivity);
                            int separation_time = (s1 < s2) ? s1 : s2;

                            int key = current_label * 1000 + neighbor_label;  // Simple hash
                            if (separation_times[key] < separation_time) {
                                separation_times[key] = separation_time;
                            }
                        }
                    }
                }
            }
        }
    }
}
