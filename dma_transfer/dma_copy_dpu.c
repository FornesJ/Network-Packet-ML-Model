#include <doca_mmap.h>
#include <doca_dma.h>
#include <doca_comch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dma_copy.h"

int main(int argc, char **argv) {
    // 1. Parse args: PCIe addresses (-p for comm channel, -r for representor on DPU), file size etc.
    doca_comch_ep_t *dpu_ep = NULL;
    char *pcie_addr = getenv("PCIe_ADDR");
    char *rep_pcie_addr = getenv("REP_PCIe_ADDR");
    doca_comch_ep_listen(pcie_addr, rep_pcie_addr, &dpu_ep);

    // 2. Receive host export descriptor + host buffer address + size
    size_t export_desc_size = BUFSIZE;  // maybe send size first
    uint8_t *export_desc = malloc(export_desc_size);
    doca_comch_recv(dpu_ep, export_desc, export_desc_size, 0);

    void *host_buf_addr = NULL;
    size_t host_buf_size = 0;
    doca_comch_recv(dpu_ep, &host_buf_addr, sizeof(host_buf_addr), 0);
    doca_comch_recv(dpu_ep, &host_buf_size, sizeof(host_buf_size), 0);

    // 3. Map host buffer as remote DOCA buffer
    doca_mmap_t *host_mmap = NULL;
    doca_mmap_create_from_export(export_desc, export_desc_size,
                                 /*dev*/, &host_mmap);
    doca_buffer_t *buf_host = NULL;
    doca_buffer_from_mmap(host_mmap, host_buf_addr,
                          host_buf_size, &buf_host);

    // 4. Allocate DPU local buffer (source of data)
    size_t dpu_buf_size = host_buf_size;
    void *dpu_buf_ptr = malloc(dpu_buf_size);
    // fill data:
    fill_data(dpu_buf_ptr, dpu_buf_size);

    doca_mmap_t *dpu_mmap = NULL;
    doca_mmap_create(&dpu_mmap);
    doca_mmap_set_memrange(dpu_mmap, dpu_buf_ptr, dpu_buf_size);
    doca_mmap_set_permissions(dpu_mmap, DOCA_ACCESS_FLAG_PCI_READ_WRITE);
    doca_mmap_start(dpu_mmap);

    doca_buffer_t *buf_dpu = NULL;
    doca_buffer_from_mmap(dpu_mmap, dpu_buf_ptr,
                          dpu_buf_size, &buf_dpu);

    // 5. Setup DMA device/context
    doca_dev_t *dev = NULL;
    open_dma_device(/* PCIe addr */, &dev);
    doca_dma_t *dma = NULL;
    doca_dma_create(dev, &dma);
    doca_ctx_t *ctx = doca_dma_as_ctx(dma);
    doca_pe_t *pe = NULL;
    doca_pe_create(&pe);
    doca_pe_connect_ctx(pe, ctx);
    doca_ctx_start(ctx);

    // 6. Issue DMA copy task: from buf_dpu → buf_host
    doca_dma_task_memcpy *memcpy_task = NULL;
    doca_dma_task_memcpy_alloc_init(dma,
                                    buf_dpu,
                                    buf_host,
                                    /* user_data */0,
                                    &memcpy_task);

    doca_task_t *task = doca_dma_task_memcpy_as_task(memcpy_task);
    doca_task_submit(task);

    // Wait for completion
    while (!task_done) {
        doca_pe_progress(pe);
    }

    doca_task_free(task);

    // 7. Signal host side done
    int done = 1;
    doca_comch_send(dpu_ep, &done, sizeof(done), 0);

    // 8. Cleanup
    doca_buffer_destroy(buf_dpu);
    doca_buffer_destroy(buf_host);
    doca_mmap_destroy(dpu_mmap);
    doca_mmap_destroy(host_mmap);
    free(dpu_buf_ptr);
    free(export_desc);
    doca_dma_destroy(dma);
    doca_dev_close(dev);
    doca_pe_destroy(pe);
    doca_comch_ep_close(dpu_ep);

    return 0;
}