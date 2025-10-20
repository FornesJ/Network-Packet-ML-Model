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
    int opt = 1;
    socklen_t addrlen = sizeof(address);

}

struct dpu_socket {
    int fd;
    int wc;
    struct sockaddr_in host_addr;
}

struct logits_buffer {
    ssize_t buf_size;
    char* buffer = {0};
}

void close_host_socket(struct host_socket *socket) {
    close(socket->socket);
    free(socket);
}

void close_dpu_socket(struct dpu_socket *socket, struct logits_buffer *logits) {
    close(socket->fd);
    free(socket);
    free(logits->buffer);
    free(logits);
}

int dpu_send_buffer(char* buffer, int size, char* host_adress) {
    struct dpu_socket *socket = malloc(sizeof(struct dpu_socket));
    struct logits_buffer *logits = malloc(sizeof(struct logits_buffer));
    logits->buffer = malloc(size);
    logits->buf_size = size;
    memcpy(logits->buffer, buffer, size);

    // create dpu socket
    socket->fd = socket(AF_INET, SOCK_STREAM, 0)
    if (socket->fd < 0) {
        printf("\n Socket creation error \n");
        close_dpu_socket(socket, logits);
        return EXIT_FAILURE;
    }

    socket->host_addr.sin_family = AF_INET;
    socket->host_addr.sin_port = htons(PORT);

    // Convert IPv4 and IPv6 addresses from text to binary form
    socket->wc = inet_pton(AF_INET, host_adress, &socket->host_addr.sin_addr);
    if (socket->wc < 0) {
        printf("\n Function inet_pton failed! \n");
        close_dpu_socket(socket, logits);
        return EXIT_FAILURE;
    }
    if (socket->wc == 0) {
        printf("\n Invalid address/ Address not supported! \n");
        close_dpu_socket(socket, logits);
        return EXIT_FAILURE;
    }

    // connect to host
    socket->wc = connect(socket->fd, (struct sockaddr*)&socket->host_addr, sizeof(socket->host_addr));
    if (socket->wc < 0) {
        printf("\n Connection Failed \n");
        close_dpu_socket(socket, logits);
        return EXIT_FAILURE;
    }

    // send buffer to host
    socket->wc = send(socket->fd, logits, sizeof(struct logits_buffer) + size);
    if (socket->wc < 0) {
        printf("\n Send buffer to host failed! \n");
        close_dpu_socket(socket, logits);
        return EXIT_FAILURE;
    }
    printf("\n Buffer of size %i bytes sent to host! \n", socket->wc);

    close_dpu_socket(socket, logits);
    return EXIT_SUCCESS;
}

int host_recv_buffer() {
    struct host_socket *socket = malloc(sizeof(struct host_socket));
    struct logits_buffer *logits = malloc(sizeof(struct logits_buffer));

    socket->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket->fd < 0) {
        perror("socket failed");
        close(socket->fd);
        free(socket);
        free(logits);
        return EXIT_FAILURE;
    }

    if (setsockopt(socket->fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &socket->opt, sizeof(socket->opt))) {
        perror("setsockopt");
        close(socket->fd);
        free(socket);
        free(logits);
        return EXIT_FAILURE;
    }

    socket->address.sin_family = AF_INET;
    socket->address.sin_addr.s_addr = INADDR_ANY;
    socket->address.sin_port = htons(PORT);

    socket->wc = bind(socket->fd, (struct sockaddr*)&socket->address, sizeof(socket->address))
    if (socket->wc < 0) {
        perror("bind failed");
        close(socket->fd);
        free(socket);
        free(logits);
        return EXIT_FAILURE;
    }

    socket->wc = listen(socket->fd, 3);
    if (socket->wc < 0) {
        perror("listen");
        close(socket->fd);
        free(socket);
        free(logits);
        return EXIT_FAILURE;
    }

    socket->dpu_socket = accept(socket->fd, (struct sockaddr*)&socket->address, &socket->addrlen);
    if (socket->dpu_socket < 0) {
        perror("accept");
        close(socket->fd);
        free(socket);
        free(logits);
        return EXIT_FAILURE;
    }
    void *recv_buffer = {0};
    socket->rc = read(socket->dpu_socket, recv_buffer, sizeof(struct logits_buffer));
    if (socket->rc < 0) {
        perror("read");
        close(socket->fd);
        free(socket);
        free(logits);
    }
    memcpy(&logits, recv_buffer, sizeof(struct logits_buffer));

    logits->buffer = malloc(logits->buf_size);
    socket->rc = read(socket->dpu_socket, logits->buffer, logits->buf_size);
    if (socket->rc < 0) {
        perror("read");
        close(socket->fd);
        free(socket);
        free(logits);
    }

    printf("Recieved buffer: %s\n", logits->buffer);
    close(socket->fd);
    free(logits->buffer);
    free(logits);
    free(socket);

    return EXIT_SUCCESS;
}

/*
int socket_transfer_dpu() {
    int status, valread, client_fd;
    struct sockaddr_in serv_addr;
    char* hello = "Hello from client";
    char buffer[1024] = { 0 };

    if ((client_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return EXIT_FAILURE;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if (inet_pton(AF_INET, "10.128.14.17", &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        return EXIT_FAILURE;
    }

    if ((status = connect(client_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr))) < 0) {
        printf("\nConnection Failed \n");
        return EXIT_FAILURE;
    }

    // subtract 1 for the null
    // terminator at the end
    send(client_fd, hello, strlen(hello), 0);
    printf("Hello message sent\n");
    valread = read(client_fd, buffer, 1024 - 1);
    printf("%s\n", buffer);

    // closing the connected socket
    close(client_fd);
    return EXIT_SUCCESS;
}

int socket_transfer_host() {
    int server_fd, new_socket;
    ssize_t valread;
    struct sockaddr_in address;
    int opt = 1;
    socklen_t addrlen = sizeof(address);
    char buffer[1024] = { 0 };
    char* hello = "Hello from server";

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    if ((new_socket = accept(server_fd, (struct sockaddr*)&address, &addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    // subtract 1 for the null
    // terminator at the end
    valread = read(new_socket, buffer, 1024 - 1); 
    printf("%s\n", buffer);
    send(new_socket, hello, strlen(hello), 0);
    printf("Hello message sent\n");

    // closing the connected socket
    close(new_socket);
  
    // closing the listening socket
    close(server_fd);
    return EXIT_SUCCESS;
}
*/

int main(int argc, char* argv[]) {
    int exit_status = EXIT_FAILURE;
    char* buffer_msg = "This is a buffer containing values 1 2 4 7 10 2033 455 6777 8999";
    int buf_size = strlen(buffer_msg);
    char* adress = "10.128.14.17";

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