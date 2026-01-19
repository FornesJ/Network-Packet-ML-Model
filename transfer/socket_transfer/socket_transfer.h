#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define PORT 8065
#define SIG_READY 1
#define LOG 1

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
    int opt;
    struct sockaddr_in host_addr;
};

struct tensor {
    int size;
    int dim;
    int *shape;
    float *buffer;
};

struct transfer_time {
    double time;
};

struct tensor* alloc_tensor();

struct dpu_socket* alloc_dpu_sock();

struct host_socket* alloc_host_sock();

struct transfer_time* alloc_transfer_time();

void free_tensor(struct tensor *t);

void close_dpu_sock(struct dpu_socket *s);

void close_host_sock(struct host_socket *s);

void free_transfer_time(struct transfer_time *t);

int open_dpu_socket(struct dpu_socket *socket_conf, char* host_adress);

int open_host_socket(struct host_socket *socket_conf);

int wait_ready_signal(struct dpu_socket *socket_conf);

int send_ready_signal(struct host_socket *socket_conf);

int send_dpu_buffer(struct dpu_socket *socket_conf, float* buffer, int size, int dim, int* shape, int log);

int recv_host_buffer(struct host_socket *socket_conf, struct tensor *new_tensor, int log);

int send_dpu_time(struct dpu_socket *socket_conf, double time);

int recv_host_time(struct host_socket *socket_conf, struct transfer_time *time);

