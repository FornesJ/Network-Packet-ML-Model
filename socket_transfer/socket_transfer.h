#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

struct tensor {
    int size;
    int dim;
    int *shape;
    float *buffer;
};

struct tensor* alloc_tensor();

void free_tensor(struct tensor *t);

int dpu_send_buffer(float* buffer, int size, int dim, int* shape, char* host_adress);

int host_recv_buffer(struct tensor *new_tensor);

