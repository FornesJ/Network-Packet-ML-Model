/*
 * dma_copy_dpu.c
 * DPU side of DOCA 2.7 D2H2D DMA example (local DRAM)
 *
 * Usage:
 *   sudo ./dma_copy_dpu <device_index>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_buf.h>
#include <doca_dma.h>
#include <doca_error.h>

#define BUF_SIZE 4096

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <device_index>\n", argv[0]);
        return 1;
    }

    int dev_idx = atoi(argv[1]);
    doca_error_t rc;

    struct doca_devinfo_list *dev_list = NULL;
    struct doca_dev *dev = NULL;
    struct doca_ctx *ctx = NULL;

    rc = doca_devinfo_list_create(&dev_list);
    rc |= doca_devinfo_list_refresh(dev_list);
    rc |= doca_devinfo_list_get(dev_list, dev_idx, &dev_list);
    rc |= doca_dev_open(dev_list, &dev);
    rc |= doca_ctx_create(&ctx);
    rc |= doca_ctx_dev_add(ctx, dev);
    rc |= doca_ctx_start(ctx);
    if (rc != DOCA_SUCCESS) { fprintf(stderr,"Device/context setup failed\n"); return 2; }

    /* --- Allocate source buffer on DPU --- */
    void *src = NULL;
    posix_memalign(&src, sysconf(_SC_PAGESIZE), BUF_SIZE);
    const char msg[] = "Hello from DPU!";
    strncpy((char*)src, msg, sizeof(msg));

    struct doca_mmap *mmap_src = NULL;
    struct doca_buf *buf_src = NULL;

    doca_mmap_create(&mmap_src);
    doca_mmap_set_memrange(mmap_src, src, BUF_SIZE);
    doca_mmap_start(mmap_src);
    doca_buf_create(mmap_src, src, BUF_SIZE, &buf_src);

    /* --- Allocate destination buffer (host, simulated) --- */
    void *dst = NULL;
    posix_memalign(&dst, sysconf(_SC_PAGESIZE), BUF_SIZE);
    memset(dst, 0, BUF_SIZE);

    struct doca_mmap *mmap_dst = NULL;
    struct doca_buf *buf_dst = NULL;

    doca_mmap_create(&mmap_dst);
    doca_mmap_set_memrange(mmap_dst, dst, BUF_SIZE);
    doca_mmap_start(mmap_dst);
    doca_buf_create(mmap_dst, dst, BUF_SIZE, &buf_dst);

    /* --- Create DMA memcpy task --- */
    struct doca_task *task = NULL;
    doca_dma_task_memcpy_set_conf(ctx, NULL); // default config
    doca_task_alloc(ctx, DOCA_DMA_TASK_MEMCPY, &task);
    doca_task_memcpy_set_buffers(task, buf_src, buf_dst, BUF_SIZE);

    /* --- Submit and wait synchronously --- */
    doca_task_submit(task);
    doca_task_wait(task);

    printf("DPU copied buffer to host memory (simulated).\n");

    /* --- Cleanup --- */
    if (task) doca_task_free(task);
    if (buf_src) doca_buf_destroy(buf_src);
    if (buf_dst) doca_buf_destroy(buf_dst);
    if (mmap_src) doca_mmap_destroy(mmap_src);
    if (mmap_dst) doca_mmap_destroy(mmap_dst);
    if (src) free(src);
    if (dst) free(dst);
    if (ctx) { doca_ctx_stop(ctx); doca_ctx_destroy(ctx); }
    if (dev) doca_dev_close(dev);
    if (dev_list) doca_devinfo_list_destroy(dev_list);

    return 0;
}
