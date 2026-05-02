#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include "dma_copy.h"





struct buf_conf* alloc_buf_conf() {
    struct buf_conf *buf = malloc(sizeof(struct buf_conf));
    if (!buf) return NULL;
    return buf;
}

void delete_buf_conf(struct buf_conf *buf) {
    if (buf) {
        free(buf->buffer);
        free(buf);
    }
}



struct buf_conf* serialize(void *data, Type type) {
    struct buf_conf* buf;
    switch (type) {
        case Type.EXPORT_CONF:
            buf = serialize_export_conf(data);
            break;
        case Type.TENSOR:
            buf = serialize_tensor(data);
            break;
        case Type.DMA_STATUS:
            buf = serialize_dma_status(data);
            break;
        default:
            buf = NULL;
            break;
    }
    return buf;
}

void* deserialize(struct buf_conf *buf, Type type) {
    void *data;
    switch (type) {
        case Type.EXPORT_CONF:
            data = (void*) deserialize_export_conf(buf);
            break;
        case Type.TENSOR:
            data = (void*) deserialize_tensor(buf);
            break;
        case Type.DMA_STATUS:
            data = (void*) deserialize_dma_status(buf);
            break;
        default:
            data = NULL;
            break;
    }
    return data;
}








struct buf_conf* serialize_dma_status(struct dma_status *status) {
    struct buf_conf *buf = alloc_buf_conf();
    if (!buf) return NULL;

    size_t out_size = sizeof(struct dma_status);

    uint8_t buffer[out_size];
    memcpy(buffer, status, out_size);

    buf->buf_size = out_size;
    buf->buffer = buffer;

    return buf;
}

struct buf_conf* serialize_export_conf(struct export_conf *export) {
    struct buf_conf *buf = alloc_buf_conf();
    if (!buf) return NULL;

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

    buf->buf_size = out_size;
    buf->buffer = buffer;

    return buf;
}

struct buf_conf* serialize_tensor(struct tensor *tensor) {
    struct buf_conf *buf = alloc_buf_conf();
    if (!buf) return NULL;

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

    buf->buf_size = out_size;
    buf->buffer = buffer;

    return buf;
}









struct dma_status* deserialize_dma_status(struct buf_conf *buf) {
    struct dma_status* status = malloc(sizeof(struct dma_status));
    if (!status) return NULL;

    memcpy(status, buf->buffer, buf->buf_size);

    // delete buffer
    delete_buf_conf(buf);

    return status;
}

struct export_conf* deserialize_export_conf(struct buf_conf *buf) {
    struct export_conf *export = malloc(sizeof(struct export_conf));
    if (!export) return NULL;

    uint8_t *ptr = buf->buffer;

    // copy export_desc_len from buffer
    memcpy(&export->export_desc_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);

    export->export_desc = malloc(export->export_desc_len);
    if (!export->export_desc) return NULL;

    // copy export from buffer
    memcpy(export->xport_desc, ptr, export->export_desc_len);

    // delete buffer
    delete_buf_conf(buf);

    return export;
}

struct tensor* deserialize_tensor(struct buf_conf *buf) {
    struct tensor *tensor = malloc(sizeof(struct tensor));
    if (!t) return NULL;

    uint8_t *ptr = buf->buffer;

    // copy num elements from buffer
    memcpy(&tensor->num_elements, ptr, sizeof(int));
    ptr += sizeof(int);

    // copy dim to buffer
    memcpy(&tensor->dim, ptr, sizeof(int));
    ptr += sizeof(int);

    // malloc shape
    tensor->shape = malloc(tensor->dim * sizeof(int));

    // copy shape from buffer
    memcpy(tensor->shape, ptr, tensor->dim * sizeof(int));
    ptr += tensor->dim * sizeof(int);

    // malloc tensor from buffer
    tensor->buffer = malloc(tensor->num_elements * sizeof(float));

    // copy tensor buffer to buffer
    memcpy(tensor->buffer, ptr, tensor->num_elements * sizeof(float));

    // delete buffer
    delete_buf_conf(buf);

    return tensor;
}

