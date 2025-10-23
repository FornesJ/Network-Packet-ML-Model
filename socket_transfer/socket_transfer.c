#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <unistd.h>

#include "socket_transfer.h"

struct tensor* alloc_tensor() {
    struct tensor *t = malloc(sizeof(struct tensor));
    if (!t) {
        return NULL;
    }
    return t;
}

void free_tensor(struct tensor *t) {
    if (t) {
        free(t->shape);
        free(t->buffer);
        free(t);
    }
}

int dpu_send_buffer(float* buffer, int size, int dim, int* shape, char* host_adress) {
    struct dpu_socket *socket_conf = malloc(sizeof(struct dpu_socket));
    if (!socket_conf) return EXIT_FAILURE;

    // create dpu socket
    socket_conf->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_conf->fd < 0) {
        printf("\n Socket creation error \n");
        goto fail;
    }

    socket_conf->host_addr.sin_family = AF_INET;
    socket_conf->host_addr.sin_port = htons(PORT);

    // Convert IPv4 and IPv6 addresses from text to binary form
    socket_conf->wc = inet_pton(AF_INET, host_adress, &socket_conf->host_addr.sin_addr);
    if (socket_conf->wc < 0) {
        printf("\n Function inet_pton failed! \n");
        goto fail;
    }
    if (socket_conf->wc == 0) {
        printf("\n Invalid address/ Address not supported! \n");
        goto fail;
    }

    // connect to host
    socket_conf->wc = connect(socket_conf->fd, (struct sockaddr*)&socket_conf->host_addr, sizeof(socket_conf->host_addr));
    if (socket_conf->wc < 0) {
        printf("\n Connection Failed \n");
        goto fail;
    }

    // send dimention of tensor to host
    int dim_net = htonl(dim);
    socket_conf->wc = send(socket_conf->fd, &dim_net, sizeof(dim_net), 0);
    if (socket_conf->wc < 0) {
        printf("\n Send tensor dim to host failed! \n");
        goto fail;
    }

    // send shape of tensor to host
    socket_conf->wc = send(socket_conf->fd, shape, dim * sizeof(int), 0);
    if (socket_conf->wc < 0) {
        printf("\n Send tensor shape to host failed! \n");
        goto fail;
    }

    // send size of buffer to host
    int size_net = htonl(size);
    socket_conf->wc = send(socket_conf->fd, &size_net, sizeof(size_net), 0);
    if (socket_conf->wc < 0) {
        printf("\n Send buf size to host failed! \n");
        goto fail;
    }

    // send buffer to host
    socket_conf->wc = send(socket_conf->fd, buffer, size * sizeof(float), 0);
    if (socket_conf->wc < 0) {
        printf("\n Send buf size to host failed! \n");
        goto fail;
    }
    printf("Buffer of size %i bytes sent to host! \n", socket_conf->wc);

    close(socket_conf->fd);
    free(socket_conf);
    return EXIT_SUCCESS;

fail:
    close(socket_conf->fd);
    free(socket_conf);
    return EXIT_FAILURE;
}

int host_recv_buffer(struct tensor *new_tensor) {
    struct host_socket *socket_conf = malloc(sizeof(struct host_socket));
    if (!socket_conf) return EXIT_FAILURE;
    socket_conf->opt = 1;
    socket_conf->addrlen = sizeof(socket_conf->address);

    // create host socket
    socket_conf->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_conf->fd < 0) {
        perror("socket failed");
        goto fail;
    }

    // set socket option
    if (setsockopt(socket_conf->fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &socket_conf->opt, sizeof(socket_conf->opt))) {
        perror("setsockopt");
        goto fail;
    }

    socket_conf->address.sin_family = AF_INET;
    socket_conf->address.sin_addr.s_addr = INADDR_ANY;
    socket_conf->address.sin_port = htons(PORT);

    // bind socket to adress
    socket_conf->wc = bind(socket_conf->fd, (struct sockaddr*)&socket_conf->address, sizeof(socket_conf->address));
    if (socket_conf->wc < 0) {
        perror("bind failed");
        goto fail;
    }

    // listen on port 8086
    socket_conf->wc = listen(socket_conf->fd, 3);
    if (socket_conf->wc < 0) {
        perror("listen");
        goto fail;
    }

    // accept connection from dpu socket
    socket_conf->dpu_socket = accept(socket_conf->fd, (struct sockaddr*)&socket_conf->address, &socket_conf->addrlen);
    if (socket_conf->dpu_socket < 0) {
        perror("accept");
        close(socket_conf->dpu_socket);
        goto fail;
    }

    // read dimention of tensor from dpu
    int dim_net;
    socket_conf->rc = read(socket_conf->dpu_socket, &dim_net, sizeof(dim_net));
    if (socket_conf->rc < 0) {
        printf("\n Failed to tensor dim from dpu! \n");
        close(socket_conf->dpu_socket);
        goto fail;
    }
    new_tensor->dim = ntohl(dim_net); // tensor dim

    // allocate memory for shape
    new_tensor->shape = malloc(new_tensor->dim * sizeof(int));
    if (!new_tensor->shape) {
        printf("\n Failed to malloc shape! \n");
        close(socket_conf->dpu_socket);
        goto fail;
    }

    // read tensor shape from dpu
    socket_conf->rc = read(socket_conf->dpu_socket, new_tensor->shape, new_tensor->dim * sizeof(int));
    if (socket_conf->rc < 0) {
        printf("\n Failed to read tensor shape from dpu! \n");
        close(socket_conf->dpu_socket);
        goto fail;
    }

    // read size of buffer from host
    int size_net;
    socket_conf->rc = read(socket_conf->dpu_socket, &size_net, sizeof(size_net));
    if (socket_conf->rc < 0) {
        printf("\n Failed to read size from host! \n");
        close(socket_conf->dpu_socket);
        goto fail;
    }
    new_tensor->size = ntohl(size_net); // buffer size

    new_tensor->buffer = malloc(new_tensor->size * sizeof(float));
    if (!new_tensor->buffer) {
        printf("\n Failed to malloc buffer! \n");
        close(socket_conf->dpu_socket);
        goto fail;
    }

    // read data from dpu into host buffer
    socket_conf->rc = read(socket_conf->dpu_socket, new_tensor->buffer, new_tensor->size * sizeof(float));
    if (socket_conf->rc < 0) {
        printf("\n Failed to read buffer from host! \n");
        close(socket_conf->dpu_socket);
        goto fail;
    }
    
    close(socket_conf->dpu_socket);
    close(socket_conf->fd);
    free(socket_conf);

    return EXIT_SUCCESS;

fail:
    close(socket_conf->fd);
    free(socket_conf);
    return EXIT_FAILURE;
}   