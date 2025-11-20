DOCA_ROOT ?= /opt/mellanox/doca
DOCA_PKG := $(shell pkg-config --exists doca && echo yes || echo no)

ifeq ($(DOCA_PKG),yes)
    CFLAGS  := -g -O2 $(shell pkg-config --cflags doca)
    LDFLAGS := $(shell pkg-config --libs-only-L doca)
    LDLIBS  := $(shell pkg-config --libs-only-l doca)
else
    CFLAGS  := -g -O2 -I$(DOCA_ROOT)/include
    LDFLAGS := -L$(DOCA_ROOT)/lib
    LDLIBS  := -ldoca
endif

SRC_HOST = dma_copy_host.c
SRC_DPU  = dma_copy_dpu.c
BIN_HOST = dma_copy_host
BIN_DPU  = dma_copy_dpu

all: $(BIN_HOST) $(BIN_DPU)

$(BIN_HOST): $(SRC_HOST)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(BIN_DPU): $(SRC_DPU)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(BIN_HOST) $(BIN_DPU)
