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
#include <doca_buf_inventory.h>
#include <doca_dma.h>
#include <doca_types.h>
#include <doca_error.h>

#define BUF_SIZE 4096

int main(int argc, char **argv) {
    struct doca_devinfo **dev_info_list;
    struct doca_devinfo *dev_info;
    struct doca_dev *dev;
	uint32_t nb_devs;
    char pci_addr_str[DOCA_DEVINFO_PCI_ADDR_SIZE] = {};
    struct doca_mmap *mmap;
    struct doca_buf_inventory *buf_inventory;
    struct doca_dma *dma;
    size_t num_elements = 1;

	doca_error_t result;
	size_t i;

    // get device

    // get list of pci devices
    result = doca_devinfo_create_list(&dev_info_list, &nb_devs);
	if (result != DOCA_SUCCESS) {
		printf("Failed to load doca devices list\n: %s", doca_error_get_descr(result));
		return result;
	}

    uint8_t supported = 0;
    // iterate through list of devices and get device that supports dma copy
    for (i = 0; i < nb_devs; i++) {
        result = doca_dma_cap_task_memcpy_is_supported((const struct doca_devinfo *) dev_info_list[i]);
        if (result == DOCA_SUCCESS) {
            dev_info = dev_info_list[i];
            supported++;
        }
    }

    // open device
    result = doca_dev_open(dev_info, &dev);
    if (result != DOCA_SUCCESS) {
        printf("Failed to open doca device: %s\n", doca_error_get_descr(result));
		goto fail_devinfo;
    }

    // get pci address of device
    result = doca_devinfo_get_pci_addr_str(dev_info, pci_addr_str);
    if (result != DOCA_SUCCESS) {
        printf("Failed to get pci addr from doca device: %s\n", doca_error_get_descr(result));
		goto fail_devinfo;
    }

    printf("supported device: %d\n", supported);
    printf("Number of devices: %d\n", nb_devs);
    printf("Device address: %s\n", pci_addr_str);






    // Creating DOCA Core Objects

    // create mmap
    result = doca_mmap_create(&mmap);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create mmap: %s\n", doca_error_get_descr(result));
        goto fail_dev;
    }

    // create buf inventory
    result = doca_buf_inventory_create(num_elements, &buf_inventory);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create buf inventory: %s\n", doca_error_get_descr(result));
        goto fail_mmap;
    }

    // create dma
    result = doca_dma_create(dev, &dma);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create dma: %s\n", doca_error_get_descr(result));
        goto fail_inventory;
    }

    printf("Created mmap, buf_inventory and dma!\n");





    // Initialize Core Structures




    // clean up!
    doca_dma_destroy(dma);
    doca_buf_inventory_destroy(buf_inventory);
    doca_mmap_destroy(mmap);
    doca_devinfo_destroy_list(dev_info_list);
    doca_dev_close(dev);
    return 0;

fail_inventory:
    doca_buf_inventory_destroy(buf_inventory);
fail_mmap:
    doca_mmap_destroy(mmap);
fail_dev:
    doca_dev_close(dev);
fail_devinfo:
    doca_devinfo_destroy_list(dev_info_list);

    return -1;
}

//int main(int argc, char **argv) {
//    if (argc < 2) {
//        fprintf(stderr, "Usage: %s <device_index>\n", argv[0]);
//        return 1;
//    }
//
//    int dev_idx = atoi(argv[1]);
//    doca_error_t rc;
//
//    struct doca_devinfo_list *dev_info_list = NULL;
//    struct doca_dev *dev = NULL;
//    struct doca_ctx *ctx = NULL;
//
//    rc = doca_devinfo_list_create(&dev_info_list);
//    rc |= doca_devinfo_list_refresh(dev_info_list);
//    rc |= doca_devinfo_list_get(dev_info_list, dev_idx, &dev_info_list);
//    rc |= doca_dev_open(dev_info_list, &dev);
//    rc |= doca_ctx_create(&ctx);
//    rc |= doca_ctx_dev_add(ctx, dev);
//    rc |= doca_ctx_start(ctx);
//    if (rc != DOCA_SUCCESS) { fprintf(stderr,"Device/context setup failed\n"); return 2; }
//
//    /* --- Allocate source buffer on DPU --- */
//    void *src = NULL;
//    posix_memalign(&src, sysconf(_SC_PAGESIZE), BUF_SIZE);
//    const char msg[] = "Hello from DPU!";
//    strncpy((char*)src, msg, sizeof(msg));
//
//    struct doca_mmap *mmap_src = NULL;
//    struct doca_buf *buf_src = NULL;
//
//    doca_mmap_create(&mmap_src);
//    doca_mmap_set_memrange(mmap_src, src, BUF_SIZE);
//    doca_mmap_start(mmap_src);
//    doca_buf_create(mmap_src, src, BUF_SIZE, &buf_src);
//
//    /* --- Allocate destination buffer (host, simulated) --- */
//    void *dst = NULL;
//    posix_memalign(&dst, sysconf(_SC_PAGESIZE), BUF_SIZE);
//    memset(dst, 0, BUF_SIZE);
//
//    struct doca_mmap *mmap_dst = NULL;
//    struct doca_buf *buf_dst = NULL;
//
//    doca_mmap_create(&mmap_dst);
//    doca_mmap_set_memrange(mmap_dst, dst, BUF_SIZE);
//    doca_mmap_start(mmap_dst);
//    doca_buf_create(mmap_dst, dst, BUF_SIZE, &buf_dst);
//
//    /* --- Create DMA memcpy task --- */
//    struct doca_task *task = NULL;
//    doca_dma_task_memcpy_set_conf(ctx, NULL); // default config
//    doca_task_alloc(ctx, DOCA_DMA_TASK_MEMCPY, &task);
//    doca_task_memcpy_set_buffers(task, buf_src, buf_dst, BUF_SIZE);
//
//    /* --- Submit and wait synchronously --- */
//    doca_task_submit(task);
//    doca_task_wait(task);
//
//    printf("DPU copied buffer to host memory (simulated).\n");
//
//    /* --- Cleanup --- */
//    if (task) doca_task_free(task);
//    if (buf_src) doca_buf_destroy(buf_src);
//    if (buf_dst) doca_buf_destroy(buf_dst);
//    if (mmap_src) doca_mmap_destroy(mmap_src);
//    if (mmap_dst) doca_mmap_destroy(mmap_dst);
//    if (src) free(src);
//    if (dst) free(dst);
//    if (ctx) { doca_ctx_stop(ctx); doca_ctx_destroy(ctx); }
//    if (dev) doca_dev_close(dev);
//    if (dev_info_list) doca_devinfo_list_destroy(dev_info_list);
//
//    return 0;
//}
