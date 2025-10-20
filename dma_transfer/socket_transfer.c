#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <unistd.h>
#define PORT 8065

struct host_socket {
    int fd;
    int wc;
    int dpu_socket;
    ssize_t rc;
    struct sockaddr_in address;
    int opt;
    socklen_t addrlen;
};

struct dpu_socket {
    int fd;
    int wc;
    struct sockaddr_in host_addr;
};

void close_dpu_socket(struct dpu_socket *socket) {
    close(socket->fd);
    free(socket);
}

int dpu_send_buffer(float* tensor, int size, char* host_adress) {
    struct dpu_socket *socket_conf = malloc(sizeof(struct dpu_socket));

    // create dpu socket
    socket_conf->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_conf->fd < 0) {
        printf("\n Socket creation error \n");
        close_dpu_socket(socket_conf);
        return EXIT_FAILURE;
    }

    socket_conf->host_addr.sin_family = AF_INET;
    socket_conf->host_addr.sin_port = htons(PORT);

    // Convert IPv4 and IPv6 addresses from text to binary form
    socket_conf->wc = inet_pton(AF_INET, host_adress, &socket_conf->host_addr.sin_addr);
    if (socket_conf->wc < 0) {
        printf("\n Function inet_pton failed! \n");
        close_dpu_socket(socket_conf);
        return EXIT_FAILURE;
    }
    if (socket_conf->wc == 0) {
        printf("\n Invalid address/ Address not supported! \n");
        close_dpu_socket(socket_conf);
        return EXIT_FAILURE;
    }

    // connect to host
    socket_conf->wc = connect(socket_conf->fd, (struct sockaddr*)&socket_conf->host_addr, sizeof(socket_conf->host_addr));
    if (socket_conf->wc < 0) {
        printf("\n Connection Failed \n");
        close_dpu_socket(socket_conf);
        return EXIT_FAILURE;
    }

    // send size of buffer to host
    int size_net = htonl(size);
    socket_conf->wc = send(socket_conf->fd, &size_net, sizeof(size_net), 0);
    if (socket_conf->wc < 0) {
        printf("\n Send buf size to host failed! \n");
        close_dpu_socket(socket_conf);
        return EXIT_FAILURE;
    }

    // send buffer to host
    socket_conf->wc = send(socket_conf->fd, tensor, size * sizeof(float), 0);
    if (socket_conf->wc < 0) {
        printf("\n Send buf size to host failed! \n");
        close_dpu_socket(socket_conf);
        return EXIT_FAILURE;
    }
    printf("\n Buffer of size %i bytes sent to host! \n", socket_conf->wc);

    close_dpu_socket(socket_conf);
    return EXIT_SUCCESS;
}

int host_recv_buffer() {
    struct host_socket *socket_conf = malloc(sizeof(struct host_socket));
    socket_conf->opt = 1;
    socket_conf->addrlen = sizeof(socket_conf->address);

    // create host socket
    socket_conf->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_conf->fd < 0) {
        perror("socket failed");
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }

    // set socket option
    if (setsockopt(socket_conf->fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &socket_conf->opt, sizeof(socket_conf->opt))) {
        perror("setsockopt");
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }

    socket_conf->address.sin_family = AF_INET;
    socket_conf->address.sin_addr.s_addr = INADDR_ANY;
    socket_conf->address.sin_port = htons(PORT);

    // bind socket to adress
    socket_conf->wc = bind(socket_conf->fd, (struct sockaddr*)&socket_conf->address, sizeof(socket_conf->address));
    if (socket_conf->wc < 0) {
        perror("bind failed");
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }

    // listen on port 8086
    socket_conf->wc = listen(socket_conf->fd, 3);
    if (socket_conf->wc < 0) {
        perror("listen");
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }

    // accept connection from dpu socket
    socket_conf->dpu_socket = accept(socket_conf->fd, (struct sockaddr*)&socket_conf->address, &socket_conf->addrlen);
    if (socket_conf->dpu_socket < 0) {
        perror("accept");
        close(socket_conf->dpu_socket);
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }

    int size_net;
    socket_conf->rc = read(socket_conf->dpu_socket, &size_net, sizeof(size_net));
    if (socket_conf->rc < 0) {
        printf("\n Failed to read size from host! \n");
        close(socket_conf->dpu_socket);
        close(socket_conf->fd);
        free(socket_conf);
    }
    int size = ntohl(size_net);

    // allocate buffer
    float *tensor = malloc(size * sizeof(float));
    if (!tensor) {
        printf("\n Failed to allocate tesnor on host! \n");
        close(socket_conf->dpu_socket);
        close(socket_conf->fd);
        free(socket_conf);
        free(tensor);
    }

    socket_conf->rc = read(socket_conf->dpu_socket, tensor, size * sizeof(float));
    if (socket_conf->rc < 0) {
        printf("\n Failed to read buffer from host! \n");
        close(socket_conf->dpu_socket);
        close(socket_conf->fd);
        free(socket_conf);
        free(tensor);
    }

    for (int i = 0; i < size; i++) {
        printf("buffer[%d] = %f\n", i, tensor[i]);
    }
    close(socket_conf->dpu_socket);
    close(socket_conf->fd);
    free(socket_conf);
    free(tensor);

    return EXIT_SUCCESS;
}

/*
int main(int argc, char* argv[]) {
    int exit_status = EXIT_FAILURE;
    float buffer_msg[] = {1.2f, 3.4f, 5.6f};
    int buf_size = sizeof(buffer_msg) / sizeof(buffer_msg[0]);
    char* adress = "127.0.0.1"; //"10.128.14.17";

    if (argc > 2) {
        printf("\n To many arguments, need only one argument (dpu/host) \n");
        return exit_status;
    }

    char *device = argv[1];
    if (strcmp(device, "dpu") == 0) {
        exit_status = dpu_send_buffer(buffer_msg, buf_size, adress);
    } else if (strcmp(device, "host") == 0) {
        exit_status = host_recv_buffer();
    } else {
        printf("\n Argument not recognised, required argument must be 'dpu' or 'host' \n");
    }

    return exit_status;
}
*/   