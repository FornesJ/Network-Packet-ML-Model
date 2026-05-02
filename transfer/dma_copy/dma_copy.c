#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

#include <doca_ctx.h>
#include <doca_pe.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_dma.h>
#include <doca_types.h>
#include <doca_error.h>

#include "dma_copy.h"







/*
 * DMA Memcpy task completed callback
 *
 * @dma_task [in]: Completed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
static void dma_memcpy_completed_callback(struct doca_dma_task_memcpy *dma_task,
					  union doca_data task_user_data,
					  union doca_data ctx_user_data)
{
    printf("callback_called!\n");
	size_t *num_remaining_tasks = (size_t *)ctx_user_data.ptr;
	doca_error_t *result = (doca_error_t *)task_user_data.ptr;

	(void)dma_task;
	/* Decrement number of remaining tasks */
	--*num_remaining_tasks;
	/* Assign success to the result */
	*result = DOCA_SUCCESS;
}

/*
 * Memcpy task error callback
 *
 * @dma_task [in]: failed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
static void dma_memcpy_error_callback(struct doca_dma_task_memcpy *dma_task,
				      union doca_data task_user_data,
				      union doca_data ctx_user_data)
{
	size_t *num_remaining_tasks = (size_t *)ctx_user_data.ptr;
	struct doca_task *task = doca_dma_task_memcpy_as_task(dma_task);
	doca_error_t *result = (doca_error_t *)task_user_data.ptr;

	/* Decrement number of remaining tasks */
	--*num_remaining_tasks;
	/* Get the result of the task */
	*result = doca_task_get_status(task);

}
















 // Open BlueField device
int open_device(struct *dev_conf config) {
    doca_error_t result;
    size_t i;

    // get list of pci devices
    result = doca_devinfo_create_list(&config->dev_info_list, &config->nb_devs);
	if (result != DOCA_SUCCESS) {
		printf("Failed to load doca devices list\n: %s", doca_error_get_descr(result));
		goto fail;
	}

    int dev_idx = -1;
    uint8_t supported = 0;
    // iterate through list of devices and get device that supports dma copy
    for (i = 0; i < config->nb_devs; i++) {
        result = doca_dma_cap_task_memcpy_is_supported((const struct doca_devinfo *) config->dev_info_list[i]);
        if (result == DOCA_SUCCESS) {
            uint8_t mmap_export;
            result = doca_mmap_cap_is_export_pci_supported((const struct doca_devinfo *) config->dev_info_list[i], &mmap_export);
            if (mmap_export > 0) {
                supported++;
                dev_idx = i;
            }
        }
    }
    if (dev_idx < 0) {
        printf("No supported devices found: %s\n", doca_error_get_descr(result));
		goto fail_dev_info;
    }
    config->dev_info = config->dev_info_list[dev_idx];

    // open device
    result = doca_dev_open(config->dev_info, &config->dev);
    if (result != DOCA_SUCCESS) {
        printf("Failed to open doca device: %s\n", doca_error_get_descr(result));
		goto fail_dev_info;
    }

    // get pci address of device
    result = doca_devinfo_get_pci_addr_str(config->dev_info, config->pci_addr_str);
    if (result != DOCA_SUCCESS) {
        printf("Failed to get pci addr from doca device: %s\n", doca_error_get_descr(result));
		goto fail_dev;
    }

    printf("supported device: %d\n", supported);
    printf("Number of devices: %d\n", config->nb_devs);
    printf("Device address: %s\n", config->pci_addr_str);

    return EXIT_SUCCESS;


fail_dev:
    doca_dev_close(config->dev);
fail_dev_info:
    doca_devinfo_destroy_list(config->dev_info_list);
fail:
    return EXIT_FAILURE;
}


















// Creating DOCA Core Objects
int dpu_create_core_objects(struct dpu_conf *config, 
                        struct dma_conf *dma_config, 
                        struct dev_conf *dev_config, 
                        struct export_conf *export) {
    doca_error_t result;

    // create local mmap
    result = doca_mmap_create(&config->local_mmap);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create local mmap: %s\n", doca_error_get_descr(result));
        goto fail;
    }

    // Import and create remote mmap from host
    result = doca_mmap_create_from_export(NULL, (const void *)export->export_desc, export->export_desc_len, dev_config->dev, &config->remote_mmap);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create remote mmap from export: %s\n", doca_error_get_descr(result));
        goto fail_local_mmap;
    }

    // create buf inventory
    result = doca_buf_inventory_create(dma_config->num_elements, &config->buf_inventory);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create buf inventory: %s\n", doca_error_get_descr(result));
        goto fail_mmap;
    }

    // create dma
    result = doca_dma_create(dev_config->dev, &dma_config->dma);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create dma: %s\n", doca_error_get_descr(result));
        goto fail_inventory;
    }

    // create progress engine
    result = doca_pe_create(&dma_config->pe);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create pe: %s\n", doca_error_get_descr(result));
        goto fail_dma;
    }

    printf("Created Core Objects!\n");
    return EXIT_SUCCESS;

fail_dma:
    doca_dma_destroy(dma_config->dma);
fail_inventory:
    doca_buf_inventory_destroy(config->buf_inventory);
fail_mmap:
    doca_mmap_destroy(config->remote_mmap);
fail_local_mmap:
    doca_mmap_destroy(config->local_mmap);
fail:
    return EXIT_FAILURE;
}



int host_create_core_objects(struct host_conf *config) {
    doca_error_t result;

    // create mmap
    result = doca_mmap_create(&config->mmap);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create mmap: %s\n", doca_error_get_descr(result));
        goto fail;
    }

    printf("Created Core Objects!\n");
    return EXIT_SUCCESS;

fail:
    return EXIT_FAILURE;
}
















// Initialize Core Structures
int dpu_init_core_objects(struct dpu_conf *config, 
                        struct dma_conf *dma_config, 
                        struct dev_conf *dev_config) {
    doca_error_t result;

    // Initialize DMA
    // create DMA Context
    dma_config->dma_ctx = doca_dma_as_ctx(dma_config->dma);
    if (dma_config->dma_ctx == NULL) {
        printf("Failed to create dma ctx: %s\n", doca_error_get_descr(result));
        goto fail;
    }

    uint32_t num_tasks = 1;
    dma_config->ctx_user_data.ptr = &num_tasks;
    result = doca_ctx_set_user_data(dma_config->dma_ctx, dma_config->ctx_user_data);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set ctx user data: %s\n", doca_error_get_descr(result));
        goto fail;
    }

    // set DMA task memcopy config
    result = doca_dma_task_memcpy_set_conf(dma_config->dma, dma_memcpy_completed_callback, 
                                            dma_memcpy_error_callback, num_tasks);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set config to dma task memcopy: %s\n", doca_error_get_descr(result));
        goto fail;
    }

    // connect Progress Engine to a Context
    result = doca_pe_connect_ctx(dma_config->pe, dma_config->dma_ctx);
    if (result != DOCA_SUCCESS) {
        printf("Failed to connect pe to dma ctx: %s\n", doca_error_get_descr(result));
        goto fail;
    }

    // start Context
    result = doca_ctx_start(dma_config->dma_ctx);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create dma: %s\n", doca_error_get_descr(result));
        goto fail;
    }







    // Initialize mmap
    // set mmap permissions based on access flags
    result = doca_mmap_set_permissions(config->local_mmap, config->mmap_access);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set permissions to mmap: %s\n", doca_error_get_descr(result));
        goto fail_ctx;
    }

    //Initialize local mmap
    result = doca_mmap_set_memrange(config->local_mmap, config->dpu_buffer, config->dpu_buffer_size);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set local mmap memrange: %s\n", doca_error_get_descr(result));
        goto fail_ctx;
    }

    // add device to mmap
    result = doca_mmap_add_dev(config->local_mmap, dev_config->dev);
    if (result != DOCA_SUCCESS) {
        printf("Failed to add device to local mmap: %s\n", doca_error_get_descr(result));
        goto fail_ctx;
    }

    // start mmap
    result = doca_mmap_start(config->local_mmap);
    if (result != DOCA_SUCCESS) {
        printf("Failed to start local mmap: %s\n", doca_error_get_descr(result));
        goto fail_ctx;
    }






    // Init buf inventory
    // Initialize doca buffers and buffer inventory
    result = doca_buf_inventory_start(config->buf_inventory);
    if (result != DOCA_SUCCESS) {
        printf("Failed to start buf_inventory: %s\n", doca_error_get_descr(result));
        goto fail_ctx;
    }

    printf("Initialized core objects!\n");

    return EXIT_SUCCESS;

fail_ctx:
    doca_ctx_stop(dma_config->dma_ctx);
fail:
    return EXIT_FAILURE;
}





int host_init_core_objects(struct dpu_conf *config,  
                        struct dev_conf *dev_config,
                        struct export_conf *export) {
    doca_error_t result;

    // allocate memory to host buffer
    config->host_buffer = (char*)malloc(config->host_buffer_size);
    if (config->host_buffer == NULL) {
        result = DOCA_ERROR_NO_MEMORY;
        printf("Failed to alloc memory to host_buffer: %s\n", doca_error_get_descr(result));
        goto fail;
    }



    // initiate export struct
    export->export_desc = NULL;
    export->export_desc_len = 0;




    // Initialize mmap
    // set memrange
    result = doca_mmap_set_memrange(config->mmap, config->host_buffer, config->host_buffer_size);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set mem range to mmap: %s\n", doca_error_get_descr(result));
        goto fail_host_buf;
    }

    // set mmap permissions based on access flags
    result = doca_mmap_set_permissions(config->mmap, config->mmap_access);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set permissions to mmap: %s\n", doca_error_get_descr(result));
        goto fail_host_buf;
    }

    // add device to mmap
    result = doca_mmap_add_dev(config->mmap, dev_config->dev);
    if (result != DOCA_SUCCESS) {
        printf("Failed to add device to mmap: %s\n", doca_error_get_descr(result));
        goto fail_host_buf;
    }

    // start mmap
    result = doca_mmap_start(config->mmap);
    if (result != DOCA_SUCCESS) {
        printf("Failed to start mmap: %s\n", doca_error_get_descr(result));
        goto fail_host_buf;
    }

    // export mmap over PCI
    result = doca_mmap_export_pci(config->mmap, dev_config->dev, (const void **)&export->export_desc, &export->export_desc_len);
    if (result != DOCA_SUCCESS) {
        printf("Failed to export mmap over pci: %s\n", doca_error_get_descr(result));
        goto fail_host_buf;
    }
    printf("Created host buffer at address: %p\n", (void*)config->host_buffer);

    return EXIT_SUCCESS;

fail_host_buf:
    free(config->host_buffer);
fail:
    return EXIT_FAILURE;
}