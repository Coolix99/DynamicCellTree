#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Define a struct for a 3D vector
typedef struct {
    int x, y, z;
} Vector3D;

// Function to get neighbors based on connectivity
void get_neighbors_3D(Vector3D *neighbors, int *neighbor_count, int connectivity) {
    if (connectivity == 6) {
        *neighbor_count = 6;
        neighbors[0] = (Vector3D){1, 0, 0};
        neighbors[1] = (Vector3D){-1, 0, 0};
        neighbors[2] = (Vector3D){0, 1, 0};
        neighbors[3] = (Vector3D){0, -1, 0};
        neighbors[4] = (Vector3D){0, 0, 1};
        neighbors[5] = (Vector3D){0, 0, -1};
    } else if (connectivity == 18) {
        // Add 18-connectivity logic if needed
    }
    // Add more cases if necessary
}

int compute_progenitors_3D(int *labels, int x_dim, int y_dim, int z_dim, 
                           int target_x, int target_y, int target_z, 
                           int *vector_field, int label, int cutoff, int connectivity) {
    // Initialize stack for DFS
    Vector3D stack[1000];  // Adjust stack size if necessary
    bool *visited = calloc(x_dim * y_dim * z_dim, sizeof(bool));
    if (!visited) {
        fprintf(stderr, "Memory allocation failed for visited array\n");
        exit(EXIT_FAILURE);
    }

    int depth = 0, stack_size = 1;
    stack[0] = (Vector3D){target_x, target_y, target_z};

    Vector3D neighbors[26];
    int neighbor_count;

    // Populate neighbors based on connectivity
    get_neighbors_3D(neighbors, &neighbor_count, connectivity);

    while (stack_size > 0 && depth < cutoff) {
        int current_size = stack_size;
        stack_size = 0;

        for (int i = 0; i < current_size; i++) {
            Vector3D pos = stack[i];
            int idx = pos.z * x_dim * y_dim + pos.y * x_dim + pos.x;

            if (visited[idx]) continue;
            visited[idx] = true;

            // Iterate through neighbors
            for (int j = 0; j < neighbor_count; j++) {
                Vector3D neighbor = {
                    pos.x + neighbors[j].x,
                    pos.y + neighbors[j].y,
                    pos.z + neighbors[j].z
                };

                int nx = neighbor.x, ny = neighbor.y, nz = neighbor.z;
                if (nx >= 0 && nx < x_dim && ny >= 0 && ny < y_dim && nz >= 0 && nz < z_dim) {
                    int nidx = nz * x_dim * y_dim + ny * x_dim + nx;

                    if (labels[nidx] == label && !visited[nidx]) {
                        // Retrieve the vector field components for this neighbor
                        int vx = vector_field[0 * x_dim * y_dim * z_dim + nidx];
                        int vy = vector_field[1 * x_dim * y_dim * z_dim + nidx];
                        int vz = vector_field[2 * x_dim * y_dim * z_dim + nidx];

                        // Check if this neighbor points back to the current voxel
                        int dx = pos.x - nx;
                        int dy = pos.y - ny;
                        int dz = pos.z - nz;

                        if (vx == dx && vy == dy && vz == dz) {
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
void find_label_separation_3D(int *labels, int *vector_field, int x_dim, int y_dim, int z_dim,
                              int cutoff, int connectivity, int *separation_times) {
    Vector3D neighbors[26];
    int neighbor_count;
    get_neighbors_3D(neighbors, &neighbor_count, connectivity);

    for (int x = 0; x < x_dim; x++) {
        for (int y = 0; y < y_dim; y++) {
            for (int z = 0; z < z_dim; z++) {
                int idx = z * x_dim * y_dim + y * x_dim + x;
                int current_label = labels[idx];
                if (current_label == 0) continue;

                for (int n = 0; n < neighbor_count; n++) {
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

// Test function in the main program
int main() {
    const int x_dim = 5, y_dim = 3, z_dim = 2;
    int labels[] = {
        1, 1, 1, 2, 2,
        1, 1, 0, 2, 2,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };

    int *vector_field = calloc(3 * x_dim * y_dim * z_dim, sizeof(int));
    if (!vector_field) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize vector_field if needed
    for (int i = 0; i < 3 * x_dim * y_dim * z_dim; i++) {
        vector_field[i] = 0;
    }

    
    int cutoff = 10;
    int connectivity = 6;
    int *separation_times = calloc(1000 * 1000, sizeof(int));
    if (!separation_times) {
        fprintf(stderr, "Memory allocation for separation_times failed\n");
        return 1;
    }

    find_label_separation_3D(labels, vector_field, x_dim, y_dim, z_dim, cutoff, connectivity, separation_times);

    for (int i = 0; i < 1000; i++) {
        for (int j = i + 1; j < 1000; j++) {
            int key = i * 1000 + j;
            if (separation_times[key] > 0) {
                printf("Labels (%d, %d): Separation Time = %d\n", i, j, separation_times[key]);
            }
        }
    }

    free(separation_times);
    return 0;
}
