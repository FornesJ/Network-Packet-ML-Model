#ifndef DMA_COPY_CORE_H_
#define DMA_COPY_CORE_H_

#include <stdbool.h>

#include <doca_argp.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_pe.h>

#include "comch_utils.h"

#define MAX_ARG_SIZE 128	                /* PCI address and file path maximum length */
#define SERVER_NAME "dma copy server"       /* Comm Channel service name */
#define NUM_DMA_TASKS (1)	                /* DMA tasks number */



enum dma_copy_mode {
	DMA_COPY_MODE_HOST,                     /* Run endpoint in Host */
	DMA_COPY_MODE_DPU                       /* Run endpoint in DPU */
};

enum dma_comch_state {
	COMCH_NEGOTIATING,                      /* DMA metadata is being negotiated */
	COMCH_COMPLETE,	                        /* DMA metadata successfully passed */
	COMCH_ERROR,	                        /* An error was detected DMA metadata negotiation */
};

struct comch_conf {
	void *app_user_data;  					/* User-data supplied by the app */
	struct doca_pe *pe;   					/* Progress engine for comch */
	struct doca_ctx *ctx; 					/* Doca context of the client/server */
	union {
		struct doca_comch_server *server; 	/* Server context (DPU only) */
		struct doca_comch_client *client; 	/* Client context (x86 host only) */
	};
	struct doca_comch_connection *active_connection; /* Single connection active on the channel */
	struct doca_dev *dev;				 	/* Device in use */
	struct doca_dev_rep *dev_rep;			/* Representor in use (DPU only) */
	uint32_t max_buf_size;				 	/* Maximum size of message on channel */
	uint8_t is_server;				 		/* Indicator of client or server */
};


struct dma_copy_conf {
    enum dma_copy_mode mode;                /* Node running mode {host, dpu} */
	char cc_dev_pci_addr[DOCA_DEVINFO_PCI_ADDR_SIZE];	  /* Comm Channel DOCA device PCI address */
	char cc_dev_rep_pci_addr[DOCA_DEVINFO_REP_PCI_ADDR_SIZE]; /* Comm Channel DOCA device representor PCI address */
    uint64_t buf_size;					    /* Buffer size in bytes */
    char *dma_copy_buf:                     /* Buffer to store field to send or file to receive */
    struct doca_dev *dev;                   /* Doca device used for DMA */
    struct doca_mmap *buf_mmap;			    /* Mmap associated with the file buffer */
    uint64_t max_dma_buf_size;              /* Max size DMA supported */

    /* DPU side only field */
	uint8_t *exported_mmap;	                /* Exported mmap sent from host to DPU */
	size_t exported_mmap_len;               /* Length of exported mmap */
	uint8_t *host_addr;	                    /* Host address of file to be used with exported mmap */

    /* Comch connection info */
	uint32_t max_comch_buf;	                /* Max buffer size the comch is configure for */
	enum dma_comch_state comch_state;       /* Current state of DMA metadata negotiation on the comch */
}



void host_dma_copy();

void dpu_dma_copy();
