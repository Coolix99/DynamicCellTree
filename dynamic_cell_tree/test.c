#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "find_label_separation_3D.c"

int main() {
    const int x_dim = 5, y_dim = 3, z_dim = 2;

    // Define the labels array as per the Python test
    int labels[] = {
        1, 1, 0, 2, 2,
        1, 1, 0, 2, 2,
        0, 0, 0, 0, 0,

        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 4, 0, 0
    };

    // Allocate and initialize the vector field
    float *vector_field = calloc(3 * x_dim * y_dim * z_dim, sizeof(float));
    if (!vector_field) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Simulate 18-connectivity overlap in the vector field
    vector_field[0 * x_dim * y_dim * z_dim + 0 * x_dim * y_dim + 0 * x_dim + 2] = 1;  // vx for (0, 0, 2)
    vector_field[1 * x_dim * y_dim * z_dim + 0 * x_dim * y_dim + 0 * x_dim + 2] = 0;  // vy for (0, 0, 2)
    vector_field[2 * x_dim * y_dim * z_dim + 0 * x_dim * y_dim + 0 * x_dim + 2] = -1; // vz for (0, 0, 2)

    vector_field[0 * x_dim * y_dim * z_dim + 0 * x_dim * y_dim + 0 * x_dim + 3] = 1;  // vx for (0, 0, 3)
    vector_field[1 * x_dim * y_dim * z_dim + 0 * x_dim * y_dim + 0 * x_dim + 3] = 0;  // vy for (0, 0, 3)
    vector_field[2 * x_dim * y_dim * z_dim + 0 * x_dim * y_dim + 0 * x_dim + 3] = 1;  // vz for (0, 0, 3)
    
    // Parameters
    int cutoff = 10;
    int connectivity = 18;

    // Allocate the separation times array
    int *separation_times = calloc(1000 * 1000, sizeof(int));
    if (!separation_times) {
        fprintf(stderr, "Memory allocation for separation_times failed\n");
        free(vector_field);
        return 1;
    }

    // Call the function
    find_label_separation_3D(labels, vector_field, x_dim, y_dim, z_dim, cutoff, connectivity, separation_times);

    // Print the results
    for (int i = 0; i < 1000; i++) {
        for (int j = i + 1; j < 1000; j++) {
            int key = i * 1000 + j;
            if (separation_times[key] > 0) {
                printf("Labels (%d, %d): Separation Time = %d\n", i, j, separation_times[key]);
            }
        }
    }

    // Free allocated memory
    free(vector_field);
    free(separation_times);

    return 0;
}
