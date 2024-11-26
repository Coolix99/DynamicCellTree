#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "find_label_separation_3D.c"
#include "sparse_map.h"

int main() {
    const int dim1 = 5, dim2 = 3, dim3 = 2;

    // Define the labels array as per the Python test
    int labels[] = {
        1, 1, 1, 2, 2,
        1, 1, 0, 2, 2,
        0, 0, 0, 0, 0,

        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 3, 0, 0
    };

    // Allocate and initialize the vector field
    float *vector_field = calloc(3 * dim1 * dim2 * dim3, sizeof(float));
    if (!vector_field) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Simulate 18-connectivity overlap in the vector field
    vector_field[0 * dim1 * dim2 * dim3 + 0 * dim1 * dim2 + 0 * dim1 + 2] = 1;  // vx for (0, 0, 2)
    vector_field[1 * dim1 * dim2 * dim3 + 0 * dim1 * dim2 + 0 * dim1 + 2] = 0;  // vy for (0, 0, 2)
    vector_field[2 * dim1 * dim2 * dim3 + 0 * dim1 * dim2 + 0 * dim1 + 2] = -1; // vz for (0, 0, 2)

    vector_field[0 * dim1 * dim2 * dim3 + 0 * dim1 * dim2 + 0 * dim1 + 3] = 1;  // vx for (0, 0, 3)
    vector_field[1 * dim1 * dim2 * dim3 + 0 * dim1 * dim2 + 0 * dim1 + 3] = 0;  // vy for (0, 0, 3)
    vector_field[2 * dim1 * dim2 * dim3 + 0 * dim1 * dim2 + 0 * dim1 + 3] = 1;  // vz for (0, 0, 3)
    
    // Parameters
    int cutoff = 10;
    int connectivity = 6;

    // Initialize the sparse map
    SparseMap sparse_map;
    //init_sparse_map(&sparse_map, 10);  // Initial capacity set to 10

    // Call the updated function
    sparse_map = find_label_separation_3D(labels, vector_field, dim1, dim2, dim3, cutoff, connectivity);

    // Print the results
    for (size_t i = 0; i < sparse_map.count; i++) {
        printf("Labels (%d, %d): Separation Time = %d\n",
               sparse_map.entries[i].pair.label1,
               sparse_map.entries[i].pair.label2,
               sparse_map.entries[i].separation_time);
    }

    // Free allocated memory
    free_sparse_map(&sparse_map);
    free(vector_field);



    return 0;
}
