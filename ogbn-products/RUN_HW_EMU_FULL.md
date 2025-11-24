# Running hw_emu with Full ogbn-products Dataset

## Important Note

By default, `hw_emu` uses `TEST_MODE` which limits the dataset to small dimensions (16 nodes, 32 edges). To run the **full ogbn-products dataset** (2.4M nodes, 61.9M edges) in hw_emu, you need to override the `TEST_MODE` flag.

**Warning**: Hardware emulation with the full dataset will be **very slow** (potentially hours or days). Consider using real hardware (`TARGET=hw`) for the full dataset.

## Method 1: Override TEST_MODE (Recommended for Full Dataset)

Build without TEST_MODE by explicitly clearing it:

```bash
# Clean previous builds
make clean

# Build kernel without TEST_MODE (full dataset)
make kernel TARGET=hw_emu EXTRA_FLAGS=""

# Build host without TEST_MODE
make host TARGET=hw_emu EXTRA_FLAGS=""

# Or build both at once
make all TARGET=hw_emu EXTRA_FLAGS=""
```

Then run:
```bash
# Generate emconfig.json
emconfigutil --platform xilinx_u50_gen3x16_xdma_5_202210_1 --od build

# Set emulation mode and run
export XCL_EMULATION_MODE=hw_emu
./host/build/host.exe \
    --xclbin_file kernel/build/gcn_layer_hls.hw_emu.xclbin \
    --device_id 0 \
    --data_dir data/ogbn-products
```

## Method 2: Temporarily Modify Makefile

Edit `Makefile` and comment out the TEST_MODE line:

```makefile
# --- LOGIC TO FORCE TEST MODE FOR EMULATION ---
COMMON_FLAGS :=
# ifeq ($(TARGET),hw_emu)
#     COMMON_FLAGS += -DTEST_MODE
# endif
```

Then build normally:
```bash
make all TARGET=hw_emu
make run TARGET=hw_emu
```

**Remember to revert the Makefile change after testing!**

## Method 3: Use Real Hardware (Fastest for Full Dataset)

For the full dataset, real hardware is much faster:

```bash
# Build for hardware
make all TARGET=hw

# Run on hardware
./host/build/host.exe \
    --xclbin_file kernel/build/gcn_layer_hls.hw.xclbin \
    --device_id 0 \
    --data_dir data/ogbn-products
```

## Verification

Before running, verify your data files:

```bash
ls -lh data/ogbn-products/*.bin
```

Expected file sizes:
- `features.bin`: ~467 MB (2449029 * 100 * 2 bytes)
- `weights.bin`: ~9.4 KB (100 * 47 * 2 bytes)
- `adj_values.bin`: ~118 MB (61859140 * 2 bytes)
- `adj_col_indices.bin`: ~236 MB (61859140 * 4 bytes)
- `adj_row_ptr.bin`: ~9.4 MB (2449030 * 4 bytes)

## Troubleshooting

### "Size mismatch" errors
- Ensure you built without TEST_MODE
- Verify data files match expected dimensions

### Emulation is too slow
- Consider using real hardware instead
- Or use a subset of the data for testing

### "Cannot find features.bin"
- Check the `--data_dir` path
- Ensure all 5 `.bin` files exist

