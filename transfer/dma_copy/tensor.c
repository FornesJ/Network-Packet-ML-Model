#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "dma_copy.h"



struct tensor* create_tensor() {
    struct tensor *t = malloc(sizeof(struct tensor));
    if (!t) {
        return NULL;
    }
    return t;
}


void delete_tensor(struct tensor *t) {
    if (t) {
        free(t->shape);
        free(t->buffer);
        free(t);
    }
}


int set_tensor_data(struct tensor *t, int num_elements, int dim, int *shape, float *buffer) {
    if (t) {
        // set num elements
        t->num_elements = num_elements;

        // set dim
        t->dim = dim;
        
        // set shape
        t->shape = malloc(dim * sizeof(int));
        memcpy(t->shape, shape, dim * sizeof(int));

        // set buffer
        t->buffer = malloc(num_elements * sizeof(float));
        memcpy(t->buffer, buffer, num_elements * sizeof(float));

        return EXIT_SUCCESS;
    }

    return EXIT_FAILURE;
}


int get_tensor_num_elements(struct tensor *t) {
    if (!t) {
        return -1;
    }
    
    return t->num_elements;
}


int get_tensor_dim(struct tensor *t) {
    if (!t) {
        return -1;
    }
    
    return t->dim;
}


int* get_tensor_shape(struct tensor *t) {
    if (!t) {
        return NULL;
    }

    return t->shape;
}

float* get_tensor_buffer(struct tensor *t) {
    if (!t) {
        return NULL;
    }

    return t->buffer;
}


