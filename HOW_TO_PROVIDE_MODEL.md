# How to Provide Model and Graph Data

## Quick Answer

You have **3 options** to provide your model and graph:

### Option 1: PyTorch Geometric (Cora) - Recommended
```bash
python scripts/extract_pytorch_model.py \
    --model_path your_model.pth \
    --output_dir data/cora
```

### Option 2: Manual (NumPy/Scipy)
```bash
python scripts/prepare_graph_data.py \
    --features features.npy \
    --weights weights.npy \
    --adjacency adjacency.npz \
    --out_dir data/cora
```

### Option 3: Synthetic (Testing)
```bash
./host/build/host.exe --use_synthetic
```

## What You Need

The inference pipeline needs **5 binary files**:
1. `features.bin` - Node features (float32)
2. `weights.bin` - GCN layer weights (float32)  
3. `adj_values.bin` - CSR adjacency values (float32)
4. `adj_col_indices.bin` - CSR column indices (int32)
5. `adj_row_ptr.bin` - CSR row pointers (int32)

The extraction scripts automatically create these from your model.

## Complete Workflow Example

```bash
# 1. Extract single layer from your PyTorch model (Cora)
python scripts/extract_pytorch_model.py \
    --model_path models/gcn.pth \
    --output_dir data/cora

# 2. Build the kernel and host
make all TARGET=hw_emu

# 3. Run inference
./host/build/host.exe \
    --xclbin_file kernel/build/gcn_layer_hls.hw_emu.xclbin \
    --data_dir data/cora \
    --enable_golden_check
```

## Important Notes

- **Dimensions must match Cora**: The HLS kernel is configured for Cora dataset:
  - `NUM_NODES = 2708`
  - `IN_FEATURES = 1433`
  - `OUT_FEATURES = 16` (single layer output)
  - `NUM_EDGES_NNZ = 10556`

- The script extracts only the **first GCN layer** (1433 -> 16) as the HLS kernel expects a single layer.

- If you need different dimensions, update `kernel/gnn_hls.h` and rebuild.

- The extraction scripts will warn you if dimensions don't match.

## More Details

- See [`scripts/QUICKSTART.md`](scripts/QUICKSTART.md) for detailed step-by-step instructions
- See [`data/README.md`](data/README.md) for binary file format details
- See [`README.md`](README.md) for full build instructions

