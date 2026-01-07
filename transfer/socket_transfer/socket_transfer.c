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

struct dpu_socket* alloc_dpu_sock() {
    struct dpu_socket *s = malloc(sizeof(struct dpu_socket));
    if (!s) return NULL;
    return s;
}

struct host_socket* alloc_host_sock() {
    struct host_socket *s = malloc(sizeof(struct host_socket));
    if (!s) return NULL;
    return s;
}

void free_tensor(struct tensor *t) {
    if (t) {
        free(t->shape);
        free(t->buffer);
        free(t);
    }
}

void close_dpu_sock(struct dpu_socket *s) {
    if (s) {
        close(s->fd);
        free(s);
    }
}

void close_host_sock(struct host_socket *s) {
    if (s) {
        close(s->dpu_socket);
        close(s->fd);
        free(s);
    }
}

int open_dpu_socket(struct dpu_socket *socket_conf, char* host_adress) {
    struct timeval tv;
    // create dpu socket
    socket_conf->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_conf->fd < 0) {
        printf("\n Socket creation error \n");
        goto fail;
    }

    tv.tv_sec = 100;
    tv.tv_usec = 0;
    if (setsockopt(socket_conf->fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv))) {
        perror("setsockopt");
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
    int countdown = 100;
    while ((socket_conf->wc = 
        connect(socket_conf->fd, (struct sockaddr*)&socket_conf->host_addr, sizeof(socket_conf->host_addr))
    ) < 0) {
        if (countdown <= 0) {
            printf("\n Connecting to host timed out! \n");
            goto fail;
        }
        sleep(1);
        countdown--;
    }

    printf("Success, opend dpu socket and connected to host (%s) at port (%d)!\n", host_adress, PORT);
    return EXIT_SUCCESS;

fail:
    close(socket_conf->fd);
    free(socket_conf);
    return EXIT_FAILURE;
}

int open_host_socket(struct host_socket *socket_conf) {
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
    printf("Success, opend host socket and listening on port (%d)!\n", PORT);

    // accept connection from dpu socket
    socket_conf->dpu_socket = accept(socket_conf->fd, (struct sockaddr*)&socket_conf->address, &socket_conf->addrlen);
    if (socket_conf->dpu_socket < 0) {
        perror("accept");
        close(socket_conf->dpu_socket);
        goto fail;
    }

    printf("Success, accepted connection from dpu!\n");
    return EXIT_SUCCESS;

fail:
    close(socket_conf->fd);
    free(socket_conf);
    return EXIT_FAILURE;
}

int wait_ready_signal(struct dpu_socket *socket_conf) {
    // wiat for signal from host
    int ready_net;
    socket_conf->wc = recv(socket_conf->fd, &ready_net, sizeof(ready_net), 0);
    if (socket_conf->wc < 0) {
        printf("\n Wait for signal timed out! \n");
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }
    if (ntohl(ready_net) != SIG_READY) {
        printf("\n Wait for signal failed! \n");
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int send_ready_signal(struct host_socket *socket_conf) {
    int ready_net = htonl(SIG_READY);
    socket_conf->wc = send(socket_conf->dpu_socket, &ready_net, sizeof(ready_net), 0);
    if (socket_conf->wc < 0) {
        printf("\n Send signal to dpu failed! \n");
        close(socket_conf->dpu_socket);
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int send_dpu_buffer(struct dpu_socket *socket_conf, float* buffer, int size, int dim, int* shape, int log) {
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
    int total_bytes = size * sizeof(float);
    int bytes_left = total_bytes;
    char *buf_ptr = (char *)buffer;

    while (bytes_left > 0) {
        socket_conf->wc = send(socket_conf->fd, buf_ptr, bytes_left, 0);
        if (socket_conf->wc < 0) {
            perror("send failed");
            break;
        }
        if (log == LOG) printf("Sent %d bytes\n", socket_conf->wc);

        bytes_left -= socket_conf->wc;
        buf_ptr += socket_conf->wc;
    }

    if (log == LOG) printf("Success, sent tensor to host!\n");
    return EXIT_SUCCESS;

fail:
    close(socket_conf->fd);
    free(socket_conf);
    return EXIT_FAILURE;
}

int recv_host_buffer(struct host_socket *socket_conf, struct tensor *new_tensor, int log) {
    // read dimention of tensor from dpu
    int dim_net;
    socket_conf->rc = recv(socket_conf->dpu_socket, &dim_net, sizeof(dim_net), 0);
    if (socket_conf->rc < 0) {
        printf("\n Failed to tensor dim from dpu! \n");
        goto fail;
    }
    new_tensor->dim = ntohl(dim_net); // tensor dim

    // allocate memory for shape
    new_tensor->shape = malloc(new_tensor->dim * sizeof(int));
    if (!new_tensor->shape) {
        printf("\n Failed to malloc shape! \n");
        goto fail;
    }

    // read tensor shape from dpu
    socket_conf->rc = recv(socket_conf->dpu_socket, new_tensor->shape, new_tensor->dim * sizeof(int), 0);
    if (socket_conf->rc < 0) {
        printf("\n Failed to read tensor shape from dpu! \n");
        goto fail;
    }

    // read size of buffer from host
    int size_net;
    socket_conf->rc = recv(socket_conf->dpu_socket, &size_net, sizeof(size_net), 0);
    if (socket_conf->rc < 0) {
        printf("\n Failed to read size from host! \n");
        goto fail;
    }
    new_tensor->size = ntohl(size_net); // buffer size

    new_tensor->buffer = malloc(new_tensor->size * sizeof(float));
    if (!new_tensor->buffer) {
        printf("\n Failed to malloc buffer! \n");
        goto fail;
    }

    int total_bytes = new_tensor->size * sizeof(float);
    int bytes_left = total_bytes;
    char *buf_ptr = (char *)new_tensor->buffer;

    while (bytes_left > 0) {
        socket_conf->rc = recv(socket_conf->dpu_socket, buf_ptr, bytes_left, 0);
        if (socket_conf->rc < 0) {
            perror("read failed");
            goto fail;
        } else if (socket_conf->rc == 0) {
            // Connection closed by peer
            break;
        }

        bytes_left -= socket_conf->rc;
        buf_ptr += socket_conf->rc;
    }
    
    if (log == LOG) printf("Success, recieved tensor from dpu!\n");
    return EXIT_SUCCESS;

fail:
    close(socket_conf->dpu_socket);
    close(socket_conf->fd);
    free(socket_conf);
    return EXIT_FAILURE;
}   