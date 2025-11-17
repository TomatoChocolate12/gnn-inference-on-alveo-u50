# ============================================================
# Kernel Makefile (invoked from top-level)
# Supports TARGET=hw|hw_emu and PLATFORM=<path/.xpfm>
# Outputs: build/gcn_layer_hls.<target>.{xo,xclbin}
# ============================================================

TARGET ?= hw
PLATFORM ?= xilinx_u50_gen3x16_xdma_5_202210_1
KERNEL_NAME ?= gnn
VPP ?= v++
XO_DIR := build
KERNEL_SRC := gnn_hls.cpp

XO_FILE := $(XO_DIR)/gcn_layer_hls.$(TARGET).xo
XCLBIN_FILE := $(XO_DIR)/gcn_layer_hls.$(TARGET).xclbin

.PHONY: all xclbin xo clean

all: xclbin

xclbin: $(XCLBIN_FILE)

xo: $(XO_FILE)

$(XO_DIR):
	@mkdir -p $(XO_DIR)

$(XO_FILE): $(KERNEL_SRC) gnn_hls.h | $(XO_DIR)
	$(VPP) -c -k $(KERNEL_NAME) -t $(TARGET) --platform $(PLATFORM) \
		-I../kernel \
		--report_level 2 \
		--save-temps \
		-o $@ $<

$(XCLBIN_FILE): $(XO_FILE) link.cfg | $(XO_DIR)
	$(VPP) -l -t $(TARGET) --platform $(PLATFORM) \
		--config link.cfg \
		-o $@ $(XO_FILE)

clean:
	@echo "[kernel] Cleaning build artifacts"
	rm -rf $(XO_DIR) *.log *.compile_summary *.link_summary *.ltx *.protoinst

