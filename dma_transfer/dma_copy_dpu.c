// dma_dpu.c
// Build with DOCA 2.7 libs. DPU side: create a local buffer with test data,
// connect to host server, receive export descriptor, create remote map of host buffer,
// submit DOCA DMA copy from local DPU buffer -> remote host buffer, wait completion,
// send "DONE" back to host.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <errno.h>

#include <doca_mmap.h>
#include <doca_buffer.h>
#include <doca_dev.h>
#include <doca_dma.h>
#include <doca_pe.h>
#include <doca_error.h>

#define HOST_IP getenv("HOST_IP")  // <--- replace with actual Host IP reachable from DPU
#define HOST_PORT 8888
#define BUF_SIZE 4096
#define EXPORT_MAX 65536

static void die(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    int sock = -1;
    ssize_t r;
    uint8_t *recv_desc = malloc(EXPORT_MAX);
    size_t recv_desc_len = 0;
    uint32_t net_len = 0;

    // 1) Prepare a DPU local buffer and fill it with sample data
    void *dpu_buf = NULL;
    if (posix_memalign(&dpu_buf, sysconf(_SC_PAGESIZE), BUF_SIZE) != 0) die("posix_memalign");
    for (int i = 0; i < BUF_SIZE; ++i) ((unsigned char*)dpu_buf)[i] = (unsigned char)(i & 0xff);

    // Create DOCA mmap for local DPU buffer (so we can create doca_buffer)
    doca_mmap_t *dpu_mmap = NULL;
    if (doca_mmap_create(&dpu_mmap) != 0) { fprintf(stderr,"doca_mmap_create dpu failed\n"); return -1; }
    doca_mmap_set_memrange(dpu_mmap, dpu_buf, BUF_SIZE);
    doca_mmap_set_permissions(dpu_mmap, DOCA_ACCESS_FLAG_PCI_READ_WRITE);
    if (doca_mmap_start(dpu_mmap) != 0) { fprintf(stderr,"doca_mmap_start dpu failed\n"); return -1; }
    doca_buffer_t *buf_dpu = NULL;
    if (doca_buffer_from_mmap(dpu_mmap, dpu_buf, BUF_SIZE, &buf_dpu) != 0) { fprintf(stderr,"doca_buffer_from_mmap dpu failed\n"); return -1; }

    // 2) Connect to host server (DPU initiates TCP)
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) die("socket");

    struct sockaddr_in servaddr = {0};
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(HOST_PORT);
    if (inet_pton(AF_INET, HOST_IP, &servaddr.sin_addr) != 1) {
        fprintf(stderr, "invalid HOST_IP\n");
        return -1;
    }

    printf("DPU: connecting to host %s:%d...\n", HOST_IP, HOST_PORT);
    if (connect(sock, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) die("connect");
    printf("DPU: connected\n");

    // 3) Receive export descriptor length, descriptor, host ptr and size
    if ((r = recv(sock, &net_len, sizeof(net_len), MSG_WAITALL)) != sizeof(net_len)) {
        die("recv net_len");
    }
    recv_desc_len = ntohl(net_len);
    if (recv_desc_len > EXPORT_MAX) {
        fprintf(stderr, "export descriptor too big\n");
        return -1;
    }

    if ((r = recv(sock, recv_desc, recv_desc_len, MSG_WAITALL)) != (ssize_t)recv_desc_len) die("recv desc");
    printf("DPU: received export descriptor (%zu bytes)\n", recv_desc_len);

    uint64_t host_ptr_net = 0;
    uint64_t host_size_net = 0;
    if (recv(sock, &host_ptr_net, sizeof(host_ptr_net), MSG_WAITALL) != sizeof(host_ptr_net)) die("recv host ptr");
    if (recv(sock, &host_size_net, sizeof(host_size_net), MSG_WAITALL) != sizeof(host_size_net)) die("recv host size");
    uint64_t host_ptr = be64toh(host_ptr_net);
    uint64_t host_size = be64toh(host_size_net);
    printf("DPU: host_ptr=0x%lx, host_size=%lu\n", (unsigned long)host_ptr, (unsigned long)host_size);

    // 4) Create mmap from export (import host exported region)
    doca_mmap_t *host_mmap = NULL;
    if (doca_mmap_create_from_export(recv_desc, recv_desc_len, /* dev */ NULL, &host_mmap) != 0) {
        fprintf(stderr,"doca_mmap_create_from_export failed\n");
        return -1;
    }

    doca_buffer_t *buf_host_remote = NULL;
    if (doca_buffer_from_mmap(host_mmap, (void*) (uintptr_t)host_ptr, host_size, &buf_host_remote) != 0) {
        fprintf(stderr,"doca_buffer_from_mmap (host remote) failed\n");
        return -1;
    }

    // 5) Setup DOCA DMA on DPU side: open device, create dma, submit memcpy buf_dpu -> buf_host_remote
    doca_dev_t *dev = NULL;
    if (doca_dev_open(0, &dev) != 0) {
        fprintf(stderr,"doca_dev_open failed\n");
        return -1;
    }

    doca_dma_t *dma = NULL;
    if (doca_dma_create(dev, &dma) != 0) {
        fprintf(stderr,"doca_dma_create failed\n");
        return -1;
    }

    // Create memcpy task
    doca_dma_task_memcpy *memcpy_task = NULL;
    if (doca_dma_task_memcpy_alloc_init(dma,
                                       buf_dpu,
                                       buf_host_remote,
                                       0, /* user_data */
                                       &memcpy_task) != 0) {
        fprintf(stderr,"doca_dma_task_memcpy_alloc_init failed\n");
        return -1;
    }

    doca_task_t *task = doca_dma_task_memcpy_as_task(memcpy_task);
    if (!task) { fprintf(stderr,"task creation failed\n"); return -1; }

    printf("DPU: submitting DMA memcpy task (DPU->Host)\n");
    if (doca_task_submit(task) != 0) {
        fprintf(stderr,"doca_task_submit failed\n");
        return -1;
    }

    // 6) Poll/drive progress until complete
    doca_pe_t *pe = NULL;
    doca_pe_create(&pe);
    doca_pe_connect_ctx(pe, doca_dma_as_ctx(dma));
    int completed = 0;
    while (!completed) {
        doca_pe_progress(pe);
        // doca_task_poll / check status: simplified - check task state
        if (doca_task_get_status(task) == DOCA_TASK_STATE_FINISHED) {
            completed = 1;
            break;
        }
        usleep(1000);
    }
    printf("DPU: DMA task completed\n");

    // free task
    doca_task_free(task);

    // 7) Send "DONE" back to Host
    if (send(sock, "DONE", 4, 0) != 4) perror("send DONE");
    printf("DPU: sent DONE to host\n");

    // cleanup
    doca_dma_destroy(dma);
    doca_dev_close(dev);
    doca_mmap_destroy(host_mmap);
    doca_buffer_destroy(buf_host_remote);
    doca_buffer_destroy(buf_dpu);
    doca_mmap_destroy(dpu_mmap);
    free(recv_desc);
    free(dpu_buf);
    close(sock);

    return 0;
}
