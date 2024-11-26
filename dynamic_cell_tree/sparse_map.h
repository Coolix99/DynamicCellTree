#ifndef SPARSE_MAP_H
#define SPARSE_MAP_H

#include <stdlib.h>
#include <stdio.h>

// Define a pair of labels
typedef struct {
    int label1;
    int label2;
} LabelPair;

// Define a sparse map entry
typedef struct {
    LabelPair pair;
    int separation_time;
} SeparationEntry;

// Define the sparse map
typedef struct {
    SeparationEntry *entries;
    size_t count;
    size_t capacity;
} SparseMap;

// Function prototypes
void init_sparse_map(SparseMap *map, size_t initial_capacity);
void add_or_update_sparse_map(SparseMap *map, int label1, int label2, int separation_time);
void free_sparse_map(SparseMap *map);
void free_sparse_map_memory(SparseMap *map);


#endif // SPARSE_MAP_H
