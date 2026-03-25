#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include "dma_copy.h"


/*
uint8_t* serialize(struct serialized *data) {
    size_t out_size = sizeof(size_t) + sizeof(Type) + data->buf_size
    uint8_t *buffer = malloc(out_size);
    if (!buffer) return NULL;
    
    // copy data to buffer
    uint8_t *ptr = buffer;

    // copy buf_size
    memcpy(ptr, &data->buf_size, sizeof(size_t));
    ptr += sizeof(size_t);

    // copy data type
    memcopy(ptr, &data->type, sizeof(Type));
    ptr += sizeof(Type);

    // copy struct
    switch (data->type) {
        case 
    }
}
*/
uint8_t* serialize(void *data, Type type) {
    uint8_t *buffer;
    switch (type) {
        case Type.BUF_CONF:
            buffer = serialize_buf_conf(data);
            break;
        case Type.EXPORT_CONF:
            buffer = serialize_export_conf(data);
            break;
        case Type.TENSOR:
            buffer = serialize_tensor(data);
            break;
        case Type.DMA_STATUS:
            buffer = serialize_dma_status(data);
            break;
        default:
            buffer = NULL;
            break;
    }
    return buffer;
}

uint8_t* serialize_dma_status(struct dma_status *status) {
    size_t out_size = sizeof(struct dma_status);

    uint8_t buffer[out_size];
    memcpy(buffer, status, out_size);

    return buffer;
}

uint8_t* serialize_buf_conf(struct buf_conf *buf) {
    size_t out_size = sizeof(struct buf_conf);

    uint8_t buffer[out_size];
    memcpy(buffer, buf, out_size);

    return buffer;
}

uint8_t* serialize_export_conf(struct export_conf *export) {
    size_t out_size = sizeof(size_t) + export->export_desc_len;

    uint8_t *buffer = malloc(out_size);
    if (!buffer) return NULL;
    
    // pointer to buffer
    uint8_t *ptr = buffer;

    // copy export_desc_len to buffer
    memcpy(ptr, &export->export_desc_len, sizeof(size_t));
    ptr += sizeof(size_t);

    // copy export to buffer
    memcpy(ptr, export->xport_desc, export->export_desc_len);

    return buffer;
}

uint8_t* serialize_tensor(struct tensor *tensor) {
    size_t out_size = 2 * sizeof(int) + tensor->dim * sizeof(int) + tensor->num_elements * sizeof(float);
    
    uint8_t *buffer = malloc(out_size);
    if (!buffer) return NULL;
    
    // pointer to buffer
    uint8_t *ptr = buffer;

    // copy num elements to buffer
    memcpy(ptr, &tensor->num_elements, sizeof(int));
    ptr += sizeof(int);

    // copy dim to buffer
    memcpy(ptr, &tensor->dim, sizeof(int));
    ptr += sizeof(int);

    // copy shape to buffer
    memcpy(ptr, tensor->shape, tensor->dim * sizeof(int));
    ptr += tensor->dim * sizeof(int);


    // copy tensor buffer to buffer
    memcpy(ptr, tensor->buffer, tensor->num_elements * sizeof(float));

    return buffer;
}





