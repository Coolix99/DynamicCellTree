#include "sparse_map.h"

// Initialize the sparse map
void init_sparse_map(SparseMap *map, size_t initial_capacity) {
    map->entries = malloc(initial_capacity * sizeof(SeparationEntry));
    map->count = 0;
    map->capacity = initial_capacity;
}

// Add or update an entry in the sparse map
void add_or_update_sparse_map(SparseMap *map, int label1, int label2, int separation_time) {
    // Ensure label1 <= label2 for consistent ordering
    if (label1 > label2) {
        int temp = label1;
        label1 = label2;
        label2 = temp;
    }

    // Check if the entry already exists
    for (size_t i = 0; i < map->count; i++) {
        if (map->entries[i].pair.label1 == label1 && map->entries[i].pair.label2 == label2) {
            if (map->entries[i].separation_time < separation_time) {
                map->entries[i].separation_time = separation_time;
            }
            return;
        }
    }

    // Add new entry if not found
    if (map->count == map->capacity) {
        map->capacity *= 2;
        map->entries = realloc(map->entries, map->capacity * sizeof(SeparationEntry));
        if (!map->entries) {
            fprintf(stderr, "Failed to reallocate memory for sparse map\n");
            exit(EXIT_FAILURE);
        }
    }

    map->entries[map->count++] = (SeparationEntry){{label1, label2}, separation_time};
}

// Free the sparse map
void free_sparse_map(SparseMap *map) {
    free(map->entries);
}

void free_sparse_map_memory(SparseMap *map) {
    if (map) {
        free_sparse_map(map);  // Frees internal entries
    }
}