# Quick Start: Run hw_emu with Full ogbn-products Dataset

## Prerequisites

1. Ensure you have the data files in `data/ogbn-products/`:
   ```bash
   ls -lh data/ogbn-products/*.bin
   ```

2. Set up Vitis and XRT environment:
   ```bash
   source /opt/xilinx/xrt/setup.sh
   source /opt/Xilinx/Vitis/2023.2/settings64.sh  # Adjust version as needed
   ```

## Method 1: Using the Script (Easiest)

```bash
./run_hw_emu_full.sh
```

This script will:
- Build kernel and host without TEST_MODE
- Generate emconfig.json
- Run hw_emu with the full dataset

## Method 2: Manual Commands

### Step 1: Build Kernel (without TEST_MODE)

```bash
make kernel TARGET=hw_emu DISABLE_TEST_MODE=1
```

### Step 2: Build Host (without TEST_MODE)

```bash
make host TARGET=hw_emu DISABLE_TEST_MODE=1
```

### Step 3: Generate emconfig.json

```bash
emconfigutil --platform xilinx_u50_gen3x16_xdma_5_202210_1 --od build
```

### Step 4: Run hw_emu

```bash
export XCL_EMULATION_MODE=hw_emu
./host/build/host.exe \
    --xclbin_file kernel/build/gcn_layer_hls.hw_emu.xclbin \
    --device_id 0 \
    --data_dir data/ogbn-products
```

## Method 3: Build Everything at Once

```bash
make all TARGET=hw_emu DISABLE_TEST_MODE=1
emconfigutil --platform xilinx_u50_gen3x16_xdma_5_202210_1 --od build
export XCL_EMULATION_MODE=hw_emu
./host/build/host.exe \
    --xclbin_file kernel/build/gcn_layer_hls.hw_emu.xclbin \
    --device_id 0 \
    --data_dir data/ogbn-products
```

## Important Notes

⚠️ **Performance Warning**: 
- hw_emu with the full dataset (2.4M nodes, 61.9M edges) will be **VERY SLOW**
- It may take hours or even days to complete
- For faster results, consider using real hardware (`TARGET=hw`)

⚠️ **Memory Requirements**:
- Ensure you have enough disk space for emulation artifacts
- The emulation may use significant memory

## Verification

After building, verify the xclbin was created:
```bash
ls -lh kernel/build/gcn_layer_hls.hw_emu.xclbin
```

Verify data files exist:
```bash
ls -lh data/ogbn-products/*.bin
```

Expected file sizes:
- `features.bin`: ~935 MB
- `weights.bin`: ~19 KB
- `adj_values.bin`: ~236 MB
- `adj_col_indices.bin`: ~236 MB
- `adj_row_ptr.bin`: ~9.4 MB

## Troubleshooting

### "Size mismatch" error
- Ensure you built with `DISABLE_TEST_MODE=1`
- Check that data files match expected dimensions

### Build fails
- Verify Vitis and XRT are sourced correctly
- Check platform path is correct

### Emulation too slow
- Consider using real hardware: `make all TARGET=hw`
- Or test with smaller subset first

## Alternative: Use Real Hardware

For the full dataset, real hardware is much faster:

```bash
make all TARGET=hw
./host/build/host.exe \
    --xclbin_file kernel/build/gcn_layer_hls.hw.xclbin \
    --device_id 0 \
    --data_dir data/ogbn-products
```

