#!/bin/bash
# Script to run hw_emu with full ogbn-products dataset
# This overrides TEST_MODE to use full dataset dimensions

set -e

echo "=========================================="
echo "Building for hw_emu with FULL ogbn-products dataset"
echo "=========================================="
echo ""
echo "WARNING: This will use the full dataset (2.4M nodes, 61.9M edges)"
echo "Hardware emulation will be VERY SLOW (potentially hours/days)"
echo ""

TARGET=hw_emu
PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1
XCLBIN=kernel/build/gcn_layer_hls.${TARGET}.xclbin
DATA_DIR=data/ogbn-products

# Clean previous builds (optional, comment out if you want to keep previous builds)
# echo "Cleaning previous builds..."
# make clean

echo ""
echo "Step 1: Building kernel WITHOUT TEST_MODE..."
# Use DISABLE_TEST_MODE=1 to override TEST_MODE
make kernel TARGET=${TARGET} PLATFORM=${PLATFORM} DISABLE_TEST_MODE=1

echo ""
echo "Step 2: Building host WITHOUT TEST_MODE..."
make host TARGET=${TARGET} DISABLE_TEST_MODE=1

echo ""
echo "Step 3: Generating emconfig.json..."
emconfigutil --platform ${PLATFORM} --od build

echo ""
echo "Step 4: Running hw_emu with full dataset..."
export XCL_EMULATION_MODE=hw_emu
./host/build/host.exe \
    --xclbin_file ${XCLBIN} \
    --device_id 0 \
    --data_dir ${DATA_DIR}

echo ""
echo "Done!"

