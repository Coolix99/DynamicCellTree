#include "find_label_separation_3D.c"

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

    float *vector_field = calloc(3 * x_dim * y_dim * z_dim, sizeof(float));
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
