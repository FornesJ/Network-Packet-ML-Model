#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_dma.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_mmap.h>


/* DOCA core objects used by the samples / applications */
struct program_core_objects {
	struct doca_dev *dev;		    /* doca device */
	struct doca_mmap *src_mmap;	    /* doca mmap for source buffer */
	struct doca_mmap *dst_mmap;	    /* doca mmap for destination buffer */
	struct doca_buf_inventory *buf_inv; /* doca buffer inventory */
	struct doca_ctx *ctx;		    /* doca context */
	struct doca_pe *pe;		    /* doca progress engine */
};










