// dma_host.c
// Build with DOCA 2.7 libs. Host side: allocate destination buffer, export via doca_mmap,
// listen for DPU connection, send export descriptor & metadata, wait for "DONE".

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <fcntl.h>

#include <doca_mmap.h>
#include <doca_buffer.h>
#include <doca_error.h>

#define PORT 8888
#define BUF_SIZE 4096     // default
#define EXPORT_MAX 65536  // max export descriptor size we expect

static void die(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    int listen_fd = -1, client_fd = -1;
    void *host_buf = NULL;
    size_t buf_size = BUF_SIZE;

    // Create host destination buffer (page-aligned)
    if (posix_memalign(&host_buf, sysconf(_SC_PAGESIZE), buf_size) != 0) {
        die("posix_memalign");
    }
    memset(host_buf, 0x0, buf_size);

    // Create DOCA mmap for the host buffer and export it for PCI
    doca_mmap_t *host_mmap = NULL;
    if (doca_mmap_create(&host_mmap) != 0) {
        fprintf(stderr, "doca_mmap_create failed\n");
        return -1;
    }
    doca_mmap_set_memrange(host_mmap, host_buf, buf_size);
    doca_mmap_set_permissions(host_mmap, DOCA_ACCESS_FLAG_PCI_READ_WRITE);
    if (doca_mmap_start(host_mmap) != 0) {
        fprintf(stderr, "doca_mmap_start failed\n");
        return -1;
    }

    // Export mmap for PCI (produce export descriptor bytes)
    uint8_t *export_desc = malloc(EXPORT_MAX);
    size_t export_desc_len = EXPORT_MAX;
    int rc = doca_mmap_export_pci(host_mmap, export_desc, &export_desc_len);
    if (rc != 0) {
        fprintf(stderr, "doca_mmap_export_pci failed: %d\n", rc);
        return -1;
    }
    printf("Host: exported mmap, desc len=%zu bytes\n", export_desc_len);

    // Setup TCP server
    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) die("socket");

    int opt = 1;
    if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) die("setsockopt");

    struct sockaddr_in servaddr = {0};
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY; // listen on all interfaces; restrict as needed
    servaddr.sin_port = htons(PORT);

    if (bind(listen_fd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) die("bind");
    if (listen(listen_fd, 1) < 0) die("listen");
    printf("Host: listening on port %d\n", PORT);

    client_fd = accept(listen_fd, NULL, NULL);
    if (client_fd < 0) die("accept");
    printf("Host: DPU connected\n");

    // Send metadata: (1) export_desc_len (uint32_t network), (2) export_desc bytes, (3) host virtual pointer (uint64_t), (4) buf_size (uint64_t)
    uint32_t net_desc_len = htonl((uint32_t)export_desc_len);
    if (send(client_fd, &net_desc_len, sizeof(net_desc_len), 0) != sizeof(net_desc_len)) die("send desc len");

    ssize_t s = send(client_fd, export_desc, export_desc_len, 0);
    if (s != (ssize_t)export_desc_len) die("send export desc");

    // send the host buffer virtual address (so remote can offset into exported region)
    uint64_t host_ptr = (uint64_t)host_buf;
    uint64_t host_ptr_net = htobe64(host_ptr);
    if (send(client_fd, &host_ptr_net, sizeof(host_ptr_net), 0) != sizeof(host_ptr_net)) die("send host ptr");

    uint64_t size_net = htobe64((uint64_t)buf_size);
    if (send(client_fd, &size_net, sizeof(size_net), 0) != sizeof(size_net)) die("send buf size");

    printf("Host: sent export descriptor + metadata to DPU\n");

    // Wait for "DONE" message (4 bytes)
    char done_buf[8] = {0};
    ssize_t r = recv(client_fd, done_buf, sizeof(done_buf), 0);
    if (r <= 0) {
        fprintf(stderr, "Host: recv() failed or connection closed\n");
    } else {
        if (strncmp(done_buf, "DONE", 4) == 0) {
            printf("Host: Received DONE from DPU. Data is now in host buffer:\n");
            // For demo, hexdump first 64 bytes
            unsigned char *p = (unsigned char *)host_buf;
            for (int i = 0; i < (int) (buf_size > 64 ? 64 : buf_size); ++i) {
                if (i % 16 == 0) printf("\n%04x: ", i);
                printf("%02x ", p[i]);
            }
            printf("\n");
        } else {
            printf("Host: Received unexpected message: %s\n", done_buf);
        }
    }

    // cleanup
    close(client_fd);
    close(listen_fd);
    doca_mmap_destroy(host_mmap);
    free(export_desc);
    free(host_buf);

    return 0;
}
