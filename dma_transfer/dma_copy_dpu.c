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

#define RECV_BUF_SIZE (512)	   /* Buffer which contains config information */
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

int main(int argc, char **argv) {
    struct dma_config dma_conf;
	doca_error_t result;
	int exit_status = EXIT_FAILURE;

	/* Set the default configuration values (Example values) */
	strcpy(dma_conf.pci_address, "03:00.0"); //getenv("REP_PCIe_ADDR")
	strcpy(dma_conf.export_desc_path, "/tmp/export_desc.txt");
	strcpy(dma_conf.buf_info_path, "/tmp/buffer_info.txt");
	/* No need to set cpy_txt because we get it from the host */
	dma_conf.cpy_txt[0] = '\0';
	dma_conf.num_dst_buf = 1;
	dma_conf.num_src_buf = 1;

    char buffer[RECV_BUF_SIZE];

	printf("hello\n");

    return 0;
}