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

#include <doca_ctx.h>
#include <doca_pe.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_dma.h>
#include <doca_types.h>
#include <doca_error.h>

#define BUF_SIZE 4096
#define PORT 8065
#define HOST_ADDR "10.128.14.17"



struct dpu_socket {
    int fd;
    int wc;
    int rc;
    int opt;
    struct sockaddr_in host_addr;
};

struct export_conf {
    void *export_desc;
    size_t export_desc_len;
};


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




int send_sig(struct dpu_socket *socket_conf) {
    int sig_net = 1;

    sig_net = htonl(sig_net);
    socket_conf->wc = send(socket_conf->fd, &sig_net, sizeof(int), 0);
    if (socket_conf->wc < 0) {
        printf("\n Send signal to host failed! \n");
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}



//int wait_sig(struct dpu_socket *socket_conf) {
//    uint8_t sig_net;
//
//    socket_conf->wc = recv(socket_conf->fd, &sig_net, sizeof(uint8_t), 0);
//    if (socket_conf->wc < 0) {
//        printf("\n Wait for signal timed out! \n");
//        close(socket_conf->fd);
//        free(socket_conf);
//        return EXIT_FAILURE;
//    }
//    if (ntohl(sig_net) != 1) {
//        printf("\n Wait for signal failed! \n");
//        close(socket_conf->fd);
//        free(socket_conf);
//        return EXIT_FAILURE;
//    }
//
//    return EXIT_SUCCESS;
//}






int send_buffer_size(struct dpu_socket *socket_conf, size_t buffer_size) {
    size_t size_net;

    size_net = htonl(buffer_size);
    socket_conf->wc = send(socket_conf->fd, &size_net, sizeof(size_t), 0);
    if (socket_conf->wc < 0) {
        printf("\n Send buffer size to host failed! \n");
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}








int recv_export_conf(struct dpu_socket *socket_conf, struct export_conf *export_conf) {
    size_t len_net;

    socket_conf->rc = recv(socket_conf->fd, &len_net, sizeof(size_t), 0);
    if (socket_conf->rc < 0) {
        printf("\n Wait for receiving export_desc_len timed out! \n");
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }
    export_conf->export_desc_len = ntohl(len_net);

    void *recv_desc = malloc(export_conf->export_desc_len);
    socket_conf->rc = recv(socket_conf->fd, recv_desc, export_conf->export_desc_len, 0);
    if (socket_conf->rc < 0) {
        printf("\n Wait for receiving export_desc timed out! \n");
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }
    export_conf->export_desc = recv_desc;
    
    return EXIT_SUCCESS;
}













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












int main(int argc, char **argv) {
    // initial structs and variables for dev
    struct doca_devinfo **dev_info_list;
    struct doca_devinfo *dev_info;
    struct doca_dev *dev;
	uint32_t nb_devs;
    char pci_addr_str[DOCA_DEVINFO_PCI_ADDR_SIZE] = {};

    // initial structs for mmap and buffer
    struct doca_mmap *local_mmap;
    struct doca_mmap *remote_mmap;
    struct doca_buf_inventory *buf_inventory;
    struct doca_buf *src_buf;
    struct doca_buf *dst_buf;
    enum doca_access_flag mmap_access = DOCA_ACCESS_FLAG_LOCAL_READ_ONLY; // access flag for read from local buffer
    char *dpu_buffer;
    size_t dpu_buffer_size;


    // initial structs and variables for dma, context, task and progress engine
    struct doca_dma *dma;
    struct doca_ctx *dma_ctx;
    struct doca_pe *pe;
    size_t num_elements = 2;
    struct doca_dma_task_memcpy *dma_task;
    struct doca_task *task;
    union doca_data ctx_user_data = {0};
    union doca_data task_user_data = {0};

    // structs for socket communication and export conf
    struct dpu_socket *dpu_sock;
    struct export_conf export;

    // error handling and utils
	doca_error_t result, task_result, ctx_result;
    int sock_result;
	size_t i;


    struct timespec ts = {
		.tv_sec = 0,
		.tv_nsec = 10000,
	};


    // initialize message
    char* msg = "Hello there from DPU!";

    // alloc memory for dpu buffer
    dpu_buffer_size = strlen(msg) + 1;
    dpu_buffer = (char*)malloc(dpu_buffer_size);
    if (dpu_buffer == NULL) {
        result = DOCA_ERROR_NO_MEMORY;
        printf("Failed to alloc memory to host_buffer: %s\n", doca_error_get_descr(result));
        return -1;
    }








    // initialize device

    // get list of pci devices
    result = doca_devinfo_create_list(&dev_info_list, &nb_devs);
	if (result != DOCA_SUCCESS) {
		printf("Failed to load doca devices list\n: %s", doca_error_get_descr(result));
        free(dpu_buffer);
		return -1;
	}

    int dev_idx = -1;
    uint8_t supported = 0;
    // iterate through list of devices and get device that supports dma copy
    for (i = 0; i < nb_devs; i++) {
        result = doca_dma_cap_task_memcpy_is_supported((const struct doca_devinfo *) dev_info_list[i]);
        if (result == DOCA_SUCCESS) {
            uint8_t mmap_export;
            result = doca_mmap_cap_is_create_from_export_pci_supported((const struct doca_devinfo *) dev_info_list[i], &mmap_export);
            if (mmap_export > 0) {
                supported++;
                dev_idx = i;
            }
        }
    }
    if (dev_idx < 0) {
        printf("No supported devices found: %s\n", doca_error_get_descr(result));
		goto fail_devinfo;
    }
    dev_info = dev_info_list[dev_idx];

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

    uint8_t export_sup;
    result = doca_mmap_cap_is_export_pci_supported(dev_info, &export_sup);
    if (result != DOCA_SUCCESS) {
        printf("Failed to get pci support func from doca device: %s\n", doca_error_get_descr(result));
		goto fail_devinfo;
    }

    printf("supported device: %d\n", supported);
    printf("Number of devices: %d\n", nb_devs);
    printf("Device address: %s\n", pci_addr_str);
    printf("Supported pci? %d\n", export_sup);
    printf("\n");











    // Open DPU sock, connect to host and recieve mmap export_desc from host
    // initialize socket
    dpu_sock = alloc_dpu_sock(dpu_sock);
    if (dpu_sock == NULL) {
        printf("Failed to alloc dpu socket!\n");
        goto fail_dev;
    }

    // open dpu socket
    sock_result = open_dpu_socket(dpu_sock);
    if (sock_result != EXIT_SUCCESS) {
        printf("Failed to open dpu socket!\n");
        goto fail_dev;
    }

    // send dpu_buffer size
    sock_result = send_buffer_size(dpu_sock, dpu_buffer_size);
    if (sock_result != EXIT_SUCCESS) {
        printf("Failed to send dpu buffer size to host!\n");
        goto fail_dev;
    }

    // recieve export_desc and export_desc_len from host
    sock_result = recv_export_conf(dpu_sock, &export);
    if (sock_result != EXIT_SUCCESS) {
        printf("Failed to receive export_conf from host!\n");
        goto fail_dev;
    }

    printf("Recieved export_desc, export_desc_len: %ld\n", export.export_desc_len);













    // Creating DOCA Core Objects
    // create local mmap
    result = doca_mmap_create(&local_mmap);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create local mmap: %s\n", doca_error_get_descr(result));
        goto fail_dev;
    }

    // Import and create remote mmap from host
    result = doca_mmap_create_from_export(NULL, (const void *)export.export_desc, export.export_desc_len, dev, &remote_mmap);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create remote mmap from export: %s\n", doca_error_get_descr(result));
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

    // create progress engine
    result = doca_pe_create(&pe);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create pe: %s\n", doca_error_get_descr(result));
        goto fail_dma;
    }

    printf("Created Core Objects!\n");











    // Initialize Core Structures

    // Initialize DMA
    // create DMA Context
    dma_ctx = doca_dma_as_ctx(dma);
    if (dma_ctx == NULL) {
        printf("Failed to create dma ctx: %s\n", doca_error_get_descr(result));
        goto fail_buf_init;
    }

    uint32_t num_tasks = 1;
    ctx_user_data.ptr = &num_tasks;
    result = doca_ctx_set_user_data(dma_ctx, ctx_user_data);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set ctx user data: %s\n", doca_error_get_descr(result));
        goto fail_ctx;
    }

    // set DMA task memcopy config
    result = doca_dma_task_memcpy_set_conf(dma, dma_memcpy_completed_callback, 
                                            dma_memcpy_error_callback, num_tasks);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set config to dma task memcopy: %s\n", doca_error_get_descr(result));
        goto fail_buf_init;
    }

    // connect Progress Engine to a Context
    result = doca_pe_connect_ctx(pe, dma_ctx);
    if (result != DOCA_SUCCESS) {
        printf("Failed to connect pe to dma ctx: %s\n", doca_error_get_descr(result));
        goto fail_ctx;
    }

    // start Context
    result = doca_ctx_start(dma_ctx);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create dma: %s\n", doca_error_get_descr(result));
        goto fail_buf_init;
    }




    // Initialize mmap
    // set mmap permissions based on access flags
    result = doca_mmap_set_permissions(local_mmap, mmap_access);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set permissions to mmap: %s\n", doca_error_get_descr(result));
        goto fail_pe;
    }

    //Initialize local mmap
    result = doca_mmap_set_memrange(local_mmap, dpu_buffer, dpu_buffer_size);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set local mmap memrange: %s\n", doca_error_get_descr(result));
        goto fail_pe;
    }

    // add device to mmap
    result = doca_mmap_add_dev(local_mmap, dev);
    if (result != DOCA_SUCCESS) {
        printf("Failed to add device to local mmap: %s\n", doca_error_get_descr(result));
        goto fail_pe;
    }

    // start mmap
    result = doca_mmap_start(local_mmap);
    if (result != DOCA_SUCCESS) {
        printf("Failed to start local mmap: %s\n", doca_error_get_descr(result));
        goto fail_pe;
    }




    // Init buf inventory
    // Initialize doca buffers and buffer inventory
    result = doca_buf_inventory_start(buf_inventory);
    if (result != DOCA_SUCCESS) {
        printf("Failed to start buf_inventory: %s\n", doca_error_get_descr(result));
        goto fail_pe;
    }

    printf("Initialized core structures!\n");











    // Set up source and destination doca buffers

    // get host buffer address and buffer size
    void *host_buffer_addr;
    size_t host_buffer_size;
    result = doca_mmap_get_memrange(remote_mmap, &host_buffer_addr, &host_buffer_size);
    if (result != DOCA_SUCCESS) {
        printf("Failed to initialize dst_buf representing buffer on host: %s\n", doca_error_get_descr(result));
        goto fail_pe;
    }

    printf("host buffer addr: %p, host buffer size: %ld\n", host_buffer_addr, host_buffer_size);
    
    // initialize src buffer
    result = doca_buf_inventory_buf_get_by_addr(buf_inventory, local_mmap, (void*)dpu_buffer, dpu_buffer_size, &src_buf);
    if (result != DOCA_SUCCESS) {
        printf("Failed to initialize src_buf representing buffer on DPU: %s\n", doca_error_get_descr(result));
        goto fail_buf_init;
    }

    // initialize dst buffer
    result = doca_buf_inventory_buf_get_by_addr(buf_inventory, remote_mmap, host_buffer_addr, host_buffer_size, &dst_buf);
    if (result != DOCA_SUCCESS) {
        printf("Failed to initialize dst_buf representing buffer on host: %s\n", doca_error_get_descr(result));
        goto fail_buf_init;
    }

    // copy message in to buffer
    void* data;
    result = doca_buf_get_data(src_buf, &data);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set data to doca_buf: %s\n", doca_error_get_descr(result));
        goto fail_buf_init;
    }

    // copy msg to data
    memcpy(data, msg, dpu_buffer_size);
    printf("doca_buf data: %s\n", (char*)data);

    // initialize data in src buffer
    result = doca_buf_set_data(src_buf, data, dpu_buffer_size);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set data to doca_buf: %s\n", doca_error_get_descr(result));
        goto fail_buf_init;
    }





    // Start DMA task

    // init dma copy task
    task_user_data.ptr = &task_result;
    result = doca_dma_task_memcpy_alloc_init(dma, src_buf, dst_buf, task_user_data, &dma_task);
    if (result != DOCA_SUCCESS) {
        printf("Failed to init dma_task for memcpy alloc: %s\n", doca_error_get_descr(result));
        goto fail_buf_init;
    }

    // get dma memcpy task
    task = doca_dma_task_memcpy_as_task(dma_task);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create task: %s\n", doca_error_get_descr(result));
        goto fail_buf_init;
    }


    // Submit dma task and wait for pe to finish
    result = doca_task_submit(task);
	if (result != DOCA_SUCCESS) {
		printf("Failed to submit DMA task: %s", doca_error_get_descr(result));
		goto fail_task;
	}

    printf("task submit!\n");

    while (true) {
        if (doca_pe_progress(pe) == 0) {
            nanosleep(&ts, &ts);
        } else {
            printf("true, calls: %ld\n", i);
            break;
        }
    }


    if (task_result == DOCA_SUCCESS) {
        printf("DMA copy Success!\n");
    } else {
        printf("DMA memcpy task failed: %s\n", doca_error_get_descr(task_result));
    }

    sock_result = send_sig(dpu_sock);
    if (sock_result != EXIT_SUCCESS) {
        printf("Failed to send signal to host!\n");
        goto fail_task;
    }

    result = task_result;





    // clean up!
    //doca_buf_dec_refcount(src_buf, NULL);
    doca_task_free(task);
    doca_ctx_stop(dma_ctx);
    //doca_buf_dec_refcount(src_buf, NULL);
    doca_buf_inventory_stop(buf_inventory);
    doca_pe_destroy(pe);
    doca_dma_destroy(dma);
    doca_buf_inventory_destroy(buf_inventory);
    doca_mmap_destroy(remote_mmap);
    doca_mmap_destroy(local_mmap);
    close_dpu_sock(dpu_sock);
    doca_devinfo_destroy_list(dev_info_list);
    doca_dev_close(dev);
    free(dpu_buffer);

    return 0;


fail_task:
    doca_task_free(task);
fail_ctx:
    doca_ctx_stop(dma_ctx);
fail_buf_init:
    doca_buf_inventory_stop(buf_inventory);
fail_pe:
    doca_pe_destroy(pe);
fail_dma:
    doca_dma_destroy(dma);
fail_inventory:
    doca_buf_inventory_destroy(buf_inventory);
fail_mmap:
    doca_mmap_destroy(remote_mmap);
    doca_mmap_destroy(local_mmap);
fail_dev:
    doca_dev_close(dev);
fail_devinfo:
    doca_devinfo_destroy_list(dev_info_list);
    free(dpu_buffer);

    return -1;
}