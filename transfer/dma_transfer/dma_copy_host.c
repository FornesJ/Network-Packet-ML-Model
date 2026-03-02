#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>

#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <fcntl.h>

#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_buf.h>
#include <doca_dma.h>
#include <doca_error.h>

#define BUF_SIZE 4096
#define PORT 8065
#define SIZE 8


struct host_socket {
    int fd;
    int wc;
    int dpu_socket;
    ssize_t rc;
    struct sockaddr_in address;
    int opt;
    socklen_t addrlen;
};

struct export_conf {
    void *export_desc;
    size_t export_desc_len;
};


struct host_socket* alloc_host_sock() {
    struct host_socket *s = malloc(sizeof(struct host_socket));
    if (!s) return NULL;
    return s;
}



void close_host_sock(struct host_socket *s) {
    if (s) {
        close(s->dpu_socket);
        close(s->fd);
        free(s);
    }
}









int open_host_socket(struct host_socket *socket_conf) {
    struct timeval tv;
    socket_conf->opt = 1;
    socket_conf->addrlen = sizeof(socket_conf->address);

    // create host socket
    socket_conf->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_conf->fd < 0) {
        perror("socket failed");
        goto fail;
    }

    // set socket options
    fcntl(socket_conf->fd, F_SETFL, 0);
    if (setsockopt(socket_conf->fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &socket_conf->opt, sizeof(socket_conf->opt))) {
        perror("setsockopt");
        goto fail;
    }

    socket_conf->address.sin_family = AF_INET;
    socket_conf->address.sin_addr.s_addr = INADDR_ANY;
    socket_conf->address.sin_port = htons(PORT);

    // bind socket to adress
    socket_conf->wc = bind(socket_conf->fd, (struct sockaddr*)&socket_conf->address, sizeof(socket_conf->address));
    if (socket_conf->wc < 0) {
        perror("bind failed");
        goto fail;
    }

    // listen on port 8086
    socket_conf->wc = listen(socket_conf->fd, 3);
    if (socket_conf->wc < 0) {
        perror("listen");
        goto fail;
    }
    printf("Success, opend host socket and listening on port (%d)!\n", PORT);

    // accept connection from dpu socket
    socket_conf->dpu_socket = accept(socket_conf->fd, (struct sockaddr*)&socket_conf->address, &socket_conf->addrlen);
    if (socket_conf->dpu_socket < 0) {
        perror("accept");
        close(socket_conf->dpu_socket);
        goto fail;
    }

    printf("Success, accepted connection from dpu!\n");
    return EXIT_SUCCESS;

fail:
    close(socket_conf->fd);
    free(socket_conf);
    return EXIT_FAILURE;
}




//int send_sig(struct host_socket *socket_conf) {
//    uint8_t sig_net = 1;
//
//    sig_net = htonl(sig_net);
//    socket_conf->wc = send(socket_conf->fd, &sig_net, sizeof(uint8_t), 0);
//    if (socket_conf->wc < 0) {
//        printf("\n Send signal to dpu failed! \n");
//        close(socket_conf->fd);
//        free(socket_conf);
//        return EXIT_FAILURE;
//    }
//
//    return EXIT_SUCCESS;
//}



int wait_sig(struct host_socket *socket_conf) {
    int sig_net;

    socket_conf->rc = recv(socket_conf->dpu_socket, &sig_net, sizeof(int), 0);
    if (socket_conf->rc < 0) {
        printf("\n Wait for signal timed out! \n");
        close(socket_conf->dpu_socket);
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }
    if (ntohl(sig_net) != 1) {
        printf("data: %d", sig_net);
        printf("\n Wait for signal failed! \n");
        close(socket_conf->dpu_socket);
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}




int recv_buffer_size(struct host_socket *socket_conf, size_t *buffer_size) {
    size_t size_net;

    socket_conf->rc = recv(socket_conf->dpu_socket, &size_net, sizeof(size_t), 0);
    if (socket_conf->rc < 0) {
        printf("\n Wait for receiving export_desc_len timed out! \n");
        close(socket_conf->dpu_socket);
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }
    size_net = ntohl(size_net);
    memcpy(buffer_size, &size_net, sizeof(size_t));

    return EXIT_SUCCESS;
}





int send_export_desc(struct host_socket *socket_conf, struct export_conf *export_conf) {
    size_t len_net;
    
    len_net = htonl(export_conf->export_desc_len);
    socket_conf->wc = send(socket_conf->dpu_socket, &len_net, sizeof(size_t), 0);
    if (socket_conf->wc < 0) {
        printf("\n Send export_desc_len to dpu failed! \n");
        close(socket_conf->dpu_socket);
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }

    socket_conf->wc = send(socket_conf->dpu_socket, export_conf->export_desc, export_conf->export_desc_len, 0);
    if (socket_conf->wc < 0) {
        printf("\n Send export_desc to dpu failed! \n");
        close(socket_conf->dpu_socket);
        close(socket_conf->fd);
        free(socket_conf);
        return EXIT_FAILURE;
    }

    printf("Sent export_desc to DPU!\n");

    return EXIT_SUCCESS;
}









int main(int argc, char **argv) {
    // initial structs and variables for dev
    struct doca_devinfo **dev_info_list;
    struct doca_devinfo *dev_info;
    struct doca_dev *dev;
	uint32_t nb_devs;
    char pci_addr_str[DOCA_DEVINFO_PCI_ADDR_SIZE] = {};

    // initial structs for mmap and buffer
    struct doca_mmap *mmap;
    //struct doca_buf_inventory *buf_inventory;
    enum doca_access_flag mmap_access = DOCA_ACCESS_FLAG_PCI_READ_WRITE; // access flag for pci read/write to from device
    char *host_buffer;
    size_t host_buffer_size;

    // initial structs and variables for dma, context, task and progress engine
    //struct doca_dma *dma;
    //size_t num_elements = 1;

    // structs for socket communication and export conf
    struct host_socket *host_sock;
    struct export_conf export;

    // error handling and utils
	doca_error_t result;
    int sock_result;
	size_t i;






    // Open BlueField device

    // get list of pci devices
    result = doca_devinfo_create_list(&dev_info_list, &nb_devs);
	if (result != DOCA_SUCCESS) {
		printf("Failed to load doca devices list\n: %s", doca_error_get_descr(result));
		return -1;
	}
    int dev_idx = -1;
    uint8_t supported = 0;
    // iterate through list of devices and get device that supports dma copy
    for (i = 0; i < nb_devs; i++) {
        result = doca_dma_cap_task_memcpy_is_supported((const struct doca_devinfo *) dev_info_list[i]);
        if (result == DOCA_SUCCESS) {
            uint8_t mmap_export;
            result = doca_mmap_cap_is_export_pci_supported((const struct doca_devinfo *) dev_info_list[i], &mmap_export);
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

    printf("supported device: %d\n", supported);
    printf("Number of devices: %d\n", nb_devs);
    printf("Device address: %s\n", pci_addr_str);







    // open host socket, wait for connection from dpu and send mmap export desc to dpu
    // alloc host socket
    host_sock = alloc_host_sock();
    if (host_sock == NULL) {
        printf("Failed to alloc host socket!\n");
        goto fail_dev;
    }

    sock_result = open_host_socket(host_sock);
    if (sock_result != EXIT_SUCCESS) {
        printf("Failed to open host socket!\n");
        goto fail_dev;
    }

    // recieve dpu buffer size
    sock_result = recv_buffer_size(host_sock, &host_buffer_size);
    if (sock_result != EXIT_SUCCESS) {
        printf("Failed to recieve buffer length from dpu!\n");
        goto fail_dev;
    }
    printf("Recieved buffer size from DPU, buf_size: %ld\n", host_buffer_size);







    // Creating DOCA Core Objects

    // create mmap
    result = doca_mmap_create(&mmap);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create mmap: %s\n", doca_error_get_descr(result));
        goto fail_dev;
    }

    printf("Created Core Objects!\n");









    // Initialize Core Structures

    // Initialize mmap
    // allocate memory to host buffer
    host_buffer = (char*)malloc(host_buffer_size);
    if (host_buffer == NULL) {
        result = DOCA_ERROR_NO_MEMORY;
        printf("Failed to alloc memory to host_buffer: %s\n", doca_error_get_descr(result));
        goto fail_mmap;
    }

    // initiate export struct
    export.export_desc = NULL;
    export.export_desc_len = 0;

    // set memrange
    result = doca_mmap_set_memrange(mmap, host_buffer, host_buffer_size);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set mem range to mmap: %s\n", doca_error_get_descr(result));
        goto fail_host_buf;
    }

    // set mmap permissions based on access flags
    result = doca_mmap_set_permissions(mmap, mmap_access);
    if (result != DOCA_SUCCESS) {
        printf("Failed to set permissions to mmap: %s\n", doca_error_get_descr(result));
        goto fail_host_buf;
    }

    // add device to mmap
    result = doca_mmap_add_dev(mmap, dev);
    if (result != DOCA_SUCCESS) {
        printf("Failed to add device to mmap: %s\n", doca_error_get_descr(result));
        goto fail_host_buf;
    }

    // start mmap
    result = doca_mmap_start(mmap);
    if (result != DOCA_SUCCESS) {
        printf("Failed to start mmap: %s\n", doca_error_get_descr(result));
        goto fail_host_buf;
    }

    // export mmap over PCI
    result = doca_mmap_export_pci(mmap, dev, (const void **)&export.export_desc, &export.export_desc_len);
    if (result != DOCA_SUCCESS) {
        printf("Failed to export mmap over pci: %s\n", doca_error_get_descr(result));
        goto fail_host_buf;
    }
    printf("Created host buffer at address: %p\n", (void*)host_buffer);

    printf("Sending export_desc to DPU, export_desc_len: %ld\n", export.export_desc_len);

    // send export_desc to dpu
    sock_result = send_export_desc(host_sock, &export);
    if (sock_result != EXIT_SUCCESS) {
        printf("Failed to send export_desc to DPU!\n");
        goto fail_host_buf;
    }

    sock_result = wait_sig(host_sock);
    if (sock_result != EXIT_SUCCESS) {
        printf("Wait for sig failed!\n");
        goto fail_host_buf;
    }

    printf("From DMA copy: %s\n", host_buffer);






    // clean up!
    //doca_dma_destroy(dma);
    //doca_buf_inventory_destroy(buf_inventory);
    //doca_mmap_destroy(test_mmap);
    free(host_buffer);
    doca_mmap_destroy(mmap);
    close_host_sock(host_sock);
    doca_devinfo_destroy_list(dev_info_list);
    doca_dev_close(dev);
    return 0;
fail_host_buf:
    free(host_buffer);
fail_mmap:
    doca_mmap_destroy(mmap);
fail_dev:
    doca_dev_close(dev);
fail_devinfo:
    doca_devinfo_destroy_list(dev_info_list);

    return -1;
}