#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_dma.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_mmap.h>

#include "dma_transfer.h"

doca_error_t comch_utils_init(const char *server_name,
			      const char *pci_addr,
			      const char *rep_pci_addr,
			      void *user_data,
			      struct comch_conf **comch_config)
{
    enum doca_ctx_states state;
	union doca_data comch_user_data = {0};
	struct timespec ts = {
		.tv_nsec = SLEEP_IN_NANOS,
	};
	struct comch_conf *config;
	doca_error_t result;

    if (server_name == NULL) {
		DOCA_LOG_ERR("Init: server name is NULL");
		return DOCA_ERROR_INVALID_VALUE;
	}

	if (pci_addr == NULL) {
		DOCA_LOG_ERR("Init: PCIe address is NULL");
		return DOCA_ERROR_INVALID_VALUE;
	}

	if (comch_cfg == NULL) {
		DOCA_LOG_ERR("Init: configuration object is NULL");
		return DOCA_ERROR_INVALID_VALUE;
	}



    /*
	return comch_utils_fast_path_init(server_name,
					  pci_addr,
					  rep_pci_addr,
					  user_data,
					  client_recv_event_cb,
					  server_recv_event_cb,
					  NULL,
					  NULL,
					  comch_cfg);
    */
    return result;
}


void host_dma_copy() {
    doca_error_t result;

    return result;
}

void dpu_dma_copy() {
    doca_error_t result;
    struct comch_conf *comch_config;
	struct dma_copy_cfg dma_config = {0};
    int exit_status = EXIT_FAILURE;

    

    result = comch_utils_init(SERVER_NAME,
				  dma_config.cc_dev_pci_addr,
				  dma_config.cc_dev_rep_pci_addr,
				  &dma_config,
				  &comch_config);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to initialize a comch: %s", doca_error_get_descr(result));
		/* ARGP destroy_resources */
        doca_argp_destroy();
        return exit_status;
	}

    return exit_status;
}

int main(int argc, char **argv) {
    

	return EXIT_SUCCESS;
}







