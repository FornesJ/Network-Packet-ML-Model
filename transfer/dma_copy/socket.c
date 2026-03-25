#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <limits.h>

#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>



struct dpu_socket* alloc_dpu_sock() {
    struct dpu_socket *s = malloc(sizeof(struct dpu_socket));
    if (!s) return NULL;
    return s;
}



void close_dpu_sock(struct dpu_socket *s) {
    if (s) {
        close(s->fd);
        free(s);
    }
}



struct host_socket* alloc_host_sock() {
    struct host_socket *s = malloc(sizeof(struct host_socket));
    if (!s) return NULL;
    return s;
}



void close_host_sock(struct host_socket *s) {
    if (s) {
        close(s->dpu_socket);
        close(s->fd);
        free(s);
    }
}



int open_dpu_socket(struct dpu_socket *socket_conf) {
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

    char* host_address = HOST_ADDR;
    socket_conf->wc = inet_pton(AF_INET, host_address, &socket_conf->host_addr.sin_addr);
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

    printf("Success, opend dpu socket and connected to host (%s) at port (%d)!\n", host_address, PORT);
    return EXIT_SUCCESS;

fail:
    close(socket_conf->fd);
    free(socket_conf);
    return EXIT_FAILURE;
}






int open_host_socket(struct host_socket *socket_conf) {
    struct timeval tv;
    socket_conf->opt = 1;
    socket_conf->addrlen = sizeof(socket_conf->address);

    // create host socket
    socket_conf->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_conf->fd < 0) {
        perror("socket failed");
        goto fail;
    }

    // set socket options
    fcntl(socket_conf->fd, F_SETFL, 0);
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



