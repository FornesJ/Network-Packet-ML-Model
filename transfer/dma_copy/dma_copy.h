#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

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



struct dpu_socket {
    int fd;
    int wc;
    ssize_t rc;
    int opt;
    struct sockaddr_in host_addr;
};




struct host_socket {
    int fd;
    int wc;
    ssize_t rc;
    int opt;
    int dpu_socket;
    struct sockaddr_in address;
    socklen_t addrlen;
};



struct tensor {
    int num_elements;
    int dim;
    int *shape;
    float *buffer;
};

struct export_conf {
    size_t export_desc_len;
    void *export_desc;
};

struct dma_status {
    Type type;
    Signal signal;
    size_t size;
};

typedef enum {
    EXPORT_CONF,
    TENSOR,
    DMA_STATUS,
    NONE
} Type;

typedef enum {
    READY,
    WAITE,
    DONE,
    ERROR
} Signal;



struct dev_conf {
    struct doca_devinfo **dev_info_list;
    struct doca_devinfo *dev_info;
    struct doca_dev *dev;
	uint32_t nb_devs;
    char pci_addr_str[DOCA_DEVINFO_PCI_ADDR_SIZE] = {};
};



struct dpu_conf {
    struct doca_mmap *local_mmap;
    struct doca_mmap *remote_mmap;
    struct doca_buf_inventory *buf_inventory;
    struct doca_buf *src_buf;
    struct doca_buf *dst_buf;
    enum doca_access_flag mmap_access = DOCA_ACCESS_FLAG_LOCAL_READ_ONLY; // access flag for read from local buffer
    size_t dpu_buffer_size;
    char *dpu_buffer;
};



struct host_conf {
    struct doca_mmap *mmap;
    enum doca_access_flag mmap_access = DOCA_ACCESS_FLAG_PCI_READ_WRITE; // access flag for pci read/write to from device
    size_t host_buffer_size;
    char *host_buffer;
};



struct dma_conf {
    struct doca_dma *dma;
    struct doca_ctx *dma_ctx;
    struct doca_pe *pe;
    size_t num_elements = 2;
    struct doca_dma_task_memcpy *dma_task;
    struct doca_task *task;
    union doca_data ctx_user_data = {0};
    union doca_data task_user_data = {0};
};






struct buf_conf {
    size_t buf_size;
    uint8_t *buffer;
};






/* Functions for handling tensors */
struct tensor* create_tensor();

void delete_tensor(struct tensor *t);

int set_tensor_data(struct tensor *t, int dim, int *shape, float *buffer);

int get_tensor_dim(struct tensor *t);

int* get_tensor_shape(struct tensor *t);

float* get_tensor_buffer(struct tensor *t);





/* Functions for serialization / deserialization */
struct buf_conf* serialize(void *data, Type type);

void* deserialize(struct buf_conf *buf, Type type);

/* Functions for allocating and freeing buf_conf*/
struct buf_conf* alloc_buf_conf();

void delete_buf_conf(struct buf_conf *buf);








/* Functions for DPU and host Socket */
struct dpu_socket* alloc_dpu_sock();

void close_dpu_sock(struct dpu_socket *s);

struct host_socket* alloc_host_sock();

void close_host_sock(struct host_socket *s);

int open_dpu_socket(struct dpu_socket *socket_conf);

int open_host_socket(struct host_socket *socket_conf);

int dpu_send(struct dpu_socket *socket_conf, void* data, Type type);

int dpu_recv(struct dpu_socket *socket_conf, void* data, Type type);

int host_send(struct host_socket *socket_conf, void* data, Type type);

int host_recv(struct host_socket *socket_conf, void* data, Type type);






/* Functions for handling DPU and host DMA transfer */


// open device
int open_device(struct *dev_conf config);


// create core objects
int dpu_create_core_objects(struct dpu_conf *config, 
                        struct dma_conf *dma_config, 
                        struct dev_conf *dev_config, 
                        struct export_conf *export);

int host_create_core_objects(struct host_conf *config);


// Initialize Core Structures
int dpu_init_core_objects(struct dpu_conf *config, 
                        struct dma_conf *dma_config, 
                        struct dev_conf *dev_config);

int host_init_core_objects(struct dpu_conf *config,  
                        struct dev_conf *dev_config,
                        struct export_conf *export);                        