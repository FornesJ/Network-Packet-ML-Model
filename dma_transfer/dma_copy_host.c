#include <doca_ctx.h>
#include <doca_pe.h>
#include <doca_mmap.h>
#include <doca_dma.h>
#include <doca_error.h>
#include <doca_argp.h>
#include <doca_dev.h>
#include <doca_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_USER_ARG_SIZE 256		     /* Maximum size of user input argument */
#define MAX_ARG_SIZE (MAX_USER_ARG_SIZE + 1) /* Maximum size of input argument */
#define MAX_USER_TXT_SIZE 4096		     /* Maximum size of user input text */
#define MAX_TXT_SIZE (MAX_USER_TXT_SIZE + 1) /* Maximum size of input text */
#define PAGE_SIZE sysconf(_SC_PAGESIZE)	     /* Page size */
#define NUM_DMA_TASKS (1)		     /* DMA tasks number */

/* DOCA core objects used by the samples / applications */
struct program_core_objects {
	struct doca_dev *dev;		    /* doca device */
	struct doca_mmap *src_mmap;	    /* doca mmap for source buffer */
	struct doca_mmap *dst_mmap;	    /* doca mmap for destination buffer */
	struct doca_buf_inventory *buf_inv; /* doca buffer inventory */
	struct doca_ctx *ctx;		    /* doca context */
	struct doca_pe *pe;		    /* doca progress engine */
};

/* Configuration struct */
struct dma_config {
	char pci_address[DOCA_DEVINFO_PCI_ADDR_SIZE]; /* PCI device address */
	char cpy_txt[MAX_TXT_SIZE];		      /* Text to copy between the two local buffers */
	char export_desc_path[MAX_ARG_SIZE];	      /* Path to save/read the exported descriptor file */
	char buf_info_path[MAX_ARG_SIZE];	      /* Path to save/read the buffer information file */
	int num_src_buf;			      /* Number of linked_list doca_buf element for the source buffer */
	int num_dst_buf; /* Number of linked_list doca_buf element for the destination buffer */
};

struct dma_resources {
	struct program_core_objects state; /* Core objects that manage our "state" */
	struct doca_dma *dma_ctx;	   /* DOCA DMA context */
	size_t num_remaining_tasks;	   /* Number of remaining tasks to process */
	bool run_pe_progress;		   /* Should we keep on progressing the PE? */
};

doca_error_t dma_copy_host(const char *pcie_addr,
			   char *src_buffer,
			   size_t src_buffer_size,
			   char *export_desc_file_path,
			   char *buffer_info_file_name)
{
    struct program_core_objects state = {0};
	const void *export_desc;
	size_t export_desc_len;
	int enter = 0;
	doca_error_t result, tmp_result;

    return result;
}

int main(int argc, char **argv) {
    struct dma_config dma_conf;
	char *src_buffer;
	size_t length;
	doca_error_t result;
	int exit_status = EXIT_FAILURE;

    /* Set the default configuration values (Example values) */
	strcpy(dma_conf.pci_address, "c1:00.0"); // getenv("PCIe_ADDR")
	strcpy(dma_conf.cpy_txt, "This is a sample piece of text");
	strcpy(dma_conf.export_desc_path, "/tmp/export_desc.txt");
	strcpy(dma_conf.buf_info_path, "/tmp/buffer_info.txt");

	length = strlen(dma_conf.cpy_txt) + 1;
	src_buffer = (char *)malloc(length);
	if (src_buffer == NULL) {
		printf("Source buffer allocation failed");
		return EXIT_FAILURE;
	}

	memcpy(src_buffer, dma_conf.cpy_txt, length);

	printf("%s\n", src_buffer);

    free(src_buffer);
    return EXIT_SUCCESS;
}
