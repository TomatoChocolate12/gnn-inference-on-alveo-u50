# ============================================================
# Top-Level Makefile for GNN Project
# ============================================================

TARGET ?= hw  # hw_emu | hw
PLATFORM ?= xilinx_u50_gen3x16_xdma_5_202210_1
XCLBIN ?= kernel/build/gcn_layer_hls.$(TARGET).xclbin
HLS_CFG ?= hls_csim.cfg
WORK_HLS ?= build/hls

# --- LOGIC TO FORCE TEST MODE FOR EMULATION ---
# If we are running hw_emu, we define TEST_MODE.
# We also pass this to 'make cosim' manually if desired.
COMMON_FLAGS :=
ifeq ($(TARGET),hw_emu)
    COMMON_FLAGS += -DTEST_MODE
endif

.PHONY: all csim csynth cosim kernel host run clean

# -------------------------
# HLS: C-simulation
# -------------------------
csim:
	@echo "=== HLS C-simulation ==="
	@mkdir -p $(WORK_HLS)
	# Pass TEST_MODE to CFLAGS so the simulation uses small graph
	vitis-run --mode hls --csim --config $(HLS_CFG) --work_dir $(WORK_HLS) --cflags "-DTEST_MODE"

# -------------------------
# HLS: Synthesis
# -------------------------
csynth:
	@echo "=== HLS Synthesis ==="
	@mkdir -p $(WORK_HLS)
	v++ -c --mode hls --config $(HLS_CFG) --work_dir $(WORK_HLS)

# -------------------------
# HLS: Co-simulation
# -------------------------
cosim: 
	@echo "=== HLS Co-simulation ==="
	# We force synthesis + cosim here with the flag to ensure small dataset
	@mkdir -p $(WORK_HLS)
	v++ -c --mode hls --config $(HLS_CFG) --work_dir $(WORK_HLS) --cflags "-DTEST_MODE"
	vitis-run --mode hls --cosim --config $(HLS_CFG) --work_dir $(WORK_HLS)

# -------------------------
# Vitis kernel build
# -------------------------
kernel:
	@echo "=== Building Kernel XCLBIN for TARGET=$(TARGET) ==="
	# Pass COMMON_FLAGS (contains -DTEST_MODE if hw_emu) to kernel makefile
	$(MAKE) -C kernel TARGET=$(TARGET) PLATFORM=$(PLATFORM) EXTRA_FLAGS="$(COMMON_FLAGS)" xclbin

# -------------------------
# Host build
# -------------------------
host:
	@echo "=== Building Host ==="
	# Pass COMMON_FLAGS to host makefile so header constants match kernel
	$(MAKE) -C host EXTRA_FLAGS="$(COMMON_FLAGS)" all

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
	./host/build/host.exe --xclbin_file $(XCLBIN) --device_id 0 -s
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