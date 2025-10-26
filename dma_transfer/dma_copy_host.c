#include <doca_mmap.h>
#include <doca_dma.h>
#include <doca_comch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dma_copy.h"

int main(int argc, char **argv) {
    // 1. Parse args: file name, size, PCIe address of comm channel (‐p), etc
    // 2. init DOCA comm channel endpoint
    doca_comch_ep_t *host_ep = NULL;
    doca_comch_ep_open(/*PCI address*/, &host_ep);

    // 3. Allocate host buffer (receive buffer)
    size_t buf_size = BUFSIZE;
    void *host_buf = aligned_alloc(sysconf(_SC_PAGESIZE), buf_size);
    if (!host_buf) { perror("malloc"); return -1; }

    // 4. Export host buffer for DPU to map  
    doca_mmap_t *host_mmap = NULL;
    doca_mmap_create(&host_mmap);
    doca_mmap_set_memrange(host_mmap, host_buf, buf_size);
    doca_mmap_set_permissions(host_mmap, DOCA_ACCESS_FLAG_PCI_READ_WRITE);
    doca_mmap_start(host_mmap);

    uint8_t *export_desc = NULL;
    size_t export_desc_size = 0;
    doca_mmap_export_pci(host_mmap, /* dev pointer */,
                          &export_desc, &export_desc_size);

    // 5. Send export descriptor + host buffer address + size over comm channel
    doca_comch_send(host_ep, export_desc, export_desc_size, 0);
    doca_comch_send(host_ep, &host_buf, sizeof(host_buf), 0);
    doca_comch_send(host_ep, &buf_size, sizeof(buf_size), 0);

    // 6. Wait for DPU to signal completion
    int done = 0;
    doca_comch_recv(host_ep, &done, sizeof(done), /*timeout*/);
    if (done != 1) {
        fprintf(stderr, "DPU signalled failure\n");
        return -1;
    }

    // 7. At this point host_buf contains data from DPU — process it
    process_data(host_buf, buf_size);

    // 8. Clean up
    doca_mmap_destroy(host_mmap);
    free(host_buf);
    doca_comch_ep_close(host_ep);

    return 0;
}
