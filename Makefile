# ============================================================
# Top-Level Makefile for GNN Project
#
# Usage:
#   make csim                # HLS C simulation
#   make csynth              # HLS synthesis -> generates RTL + .xo
#   make cosim               # HLS cosim (runs csynth first)
#   make kernel TARGET=...   # link to gcn_layer_hls.xclbin (needs PLATFORM)
#   make host                # build host
#   make all TARGET=...      # csynth + kernel + host
#   make run TARGET=...      # run host (hw or hw_emu)
#
# Env:
#   PLATFORM=<your .xpfm> (e.g., xilinx_u50_gen3x16_xdma_5_202210_1)
#   TARGET=hw|hw_emu      (default hw)
# ============================================================

TARGET ?= hw  # hw_emu | hw
PLATFORM ?= xilinx_u50_gen3x16_xdma_5_202210_1
XCLBIN ?= kernel/build/gcn_layer_hls.$(TARGET).xclbin
HLS_CFG ?= hls_csim.cfg
WORK_HLS ?= build/hls

.PHONY: all csim csynth cosim kernel host run clean

# -------------------------
# HLS: C-simulation
# -------------------------
csim:
	@echo "=== HLS C-simulation ==="
	@mkdir -p $(WORK_HLS)
	vitis-run --mode hls --csim --config $(HLS_CFG) --work_dir $(WORK_HLS)

# -------------------------
# HLS: Synthesis (produces hls/syn + .xo in $(WORK_HLS))
# -------------------------
csynth:
	@echo "=== HLS Synthesis ==="
	@mkdir -p $(WORK_HLS)
	v++ -c --mode hls --config $(HLS_CFG) --work_dir $(WORK_HLS)

# -------------------------
# HLS: Co-simulation (auto-runs csynth first, same work_dir)
# -------------------------
cosim: csynth
	@echo "=== HLS Co-simulation ==="
	vitis-run --mode hls --cosim --config $(HLS_CFG) --work_dir $(WORK_HLS)

# -------------------------
# Vitis kernel build (needs PLATFORM, respects TARGET=hw|hw_emu)
# -------------------------
kernel:
	@echo "=== Building Kernel XCLBIN for TARGET=$(TARGET) ==="
	$(MAKE) -C kernel TARGET=$(TARGET) PLATFORM=$(PLATFORM) xclbin

# -------------------------
# Host build
# -------------------------
host:
	@echo "=== Building Host ==="
	$(MAKE) -C host all

# -------------------------
# Full build
# -------------------------
all: kernel host

# -------------------------
# Run (hw or hw_emu)
# -------------------------
run: all
	@echo "=== Running Host for TARGET=$(TARGET) ==="
ifeq ($(strip $(TARGET)),hw_emu)
	@echo "=== Starting Hardware Emulation ==="
	emconfigutil --platform $(PLATFORM) --od build
	export XCL_EMULATION_MODE=hw_emu; \
	./host/build/host.exe --xclbin_file $(XCLBIN) --device_id 0
else
	@echo "=== Starting Hardware Execution ==="
	env -u XCL_EMULATION_MODE \
	./host/build/host.exe --xclbin_file $(XCLBIN) --device_id 0
endif

# -------------------------
# Clean
# -------------------------
clean:
	@echo "=== Cleaning ==="
	$(MAKE) -C kernel clean || true
	$(MAKE) -C host clean || true
	rm -rf build .Xil .vitis-run


