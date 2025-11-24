# Complete Guide: Creating Model and Graph Data for ogbn-products

This guide walks you through the complete process of creating a trained model and preparing graph data for ogbn-products dataset to run on Alveo U50.

## Prerequisites

Install required Python packages:
```bash
pip install torch torch-geometric ogb numpy scipy
```

## Step-by-Step Process

### Step 1: Train the GCN Model

Train a GCN model with the first layer matching HLS kernel dimensions (100 -> 16):

```bash
python scripts/train.py --output models/gcn_model.pth --epochs 100
```

**What this does:**
- Loads ogbn-products dataset (~2.4M nodes, 100 features)
- Creates a GCN model with `conv1: 100 -> 16` (matches HLS kernel)
- Trains the model for specified epochs
- Saves the trained model to `models/gcn_model.pth`

**Expected output:**
```
Loading ogbn-products dataset...
  Nodes: 2449029
  Edges: 61859140
  Features: 100
  Classes: 47

Creating model: 100 -> 16 -> 47
Training for 100 epochs...
Epoch 10/100, Loss: 2.3456
...
Saving model to models/gcn_model.pth
✓ Model saved successfully!
```

### Step 2: Extract Model Weights and Graph Data

Extract the first GCN layer weights and prepare the graph data:

```bash
python scripts/extract_pytorch_model.py \
    --model_path models/gcn_model.pth \
    --output_dir data/ogbn-products \
    --normalization sym
```

**What this does:**
- Loads the trained model
- Extracts weights from `conv1` layer (100 -> 16)
- Loads ogbn-products graph data
- Creates normalized adjacency matrix (symmetric normalization)
- Verifies dimensions match kernel constants
- Saves intermediate `.npy` files
- Converts to binary format (`.bin` files)

**Expected output:**
```
Loading model from models/gcn_model.pth
Extracting weights from first GCN layer 'conv1'
  Note: HLS kernel expects a single layer (100 -> 16 features for ogbn-products)
  Weight shape: (100, 16) (should be [100, 16])
Loading ogbn-products dataset from data
  Nodes: 2449029, Features: 100, Classes: 47
Preparing adjacency matrix (normalization: sym)
  Adjacency shape: (2449029, 2449029), NNZ: 61859140

Verifying dimensions match HLS kernel constants (ogbn-products):
  ✓ All dimensions match!

Saving intermediate .npy files...
  Saved data/ogbn-products/features.npy
  Saved data/ogbn-products/weights.npy
  Saved data/ogbn-products/adjacency.npz

Converting to binary format...
[prepare_graph_data] wrote 244902900 values to data/ogbn-products/features.bin
[prepare_graph_data] wrote 1600 values to data/ogbn-products/weights.bin
[prepare_graph_data] wrote 61859140 values to data/ogbn-products/adj_values.bin
[prepare_graph_data] wrote 61859140 values to data/ogbn-products/adj_col_indices.bin
[prepare_graph_data] wrote 2449030 values to data/ogbn-products/adj_row_ptr.bin

✓ Successfully prepared data in data/ogbn-products
```

### Step 3: Verify Generated Files

Check that all required binary files are created:

```bash
ls -lh data/ogbn-products/*.bin
```

You should see:
- `features.bin` (~467 MB) - Node features
- `weights.bin` (~6 KB) - GCN layer weights
- `adj_values.bin` (~118 MB) - Adjacency values
- `adj_col_indices.bin` (~236 MB) - Column indices
- `adj_row_ptr.bin` (~9 MB) - Row pointers

### Step 4: Build and Run (hw_emu)

For hardware emulation (uses small test dataset automatically):

```bash
# Build kernel and host
make all TARGET=hw_emu

# Run with synthetic data (for quick test)
make run TARGET=hw_emu
```

For full ogbn-products dataset on hardware:

```bash
# Build kernel and host
make all TARGET=hw

# Run with ogbn-products data
export XCL_EMULATION_MODE=""  # Clear emulation mode
./host/build/host.exe \
    --xclbin_file kernel/build/gcn_layer_hls.hw.xclbin \
    --device_id 0 \
    --data_dir data/ogbn-products
```

## Alternative: Manual Data Preparation

If you already have NumPy arrays, you can skip training and use:

```bash
python scripts/prepare_graph_data.py \
    --features features.npy \
    --weights weights.npy \
    --adjacency adjacency.npz \
    --out_dir data/ogbn-products
```

**Required formats:**
- `features.npy`: shape `(2449029, 100)`, dtype `float32`
- `weights.npy`: shape `(100, 16)`, dtype `float32`
- `adjacency.npz`: scipy.sparse CSR matrix, saved via `scipy.sparse.save_npz()`

## Troubleshooting

### "ModuleNotFoundError: No module named 'ogb'"
```bash
pip install ogb
```

### "Expected 100 input features, got X"
- The ogbn-products dataset should have 100 features
- Check your dataset loading code

### "Dimension mismatches detected"
- Verify your model's first layer is `100 -> 16`
- Check that the dataset matches ogbn-products dimensions
- Ensure kernel constants in `kernel/gnn_hls.h` match

### "Failed to open file: features.bin"
- Make sure you ran the extraction script successfully
- Check the `--output_dir` path
- Verify all 5 `.bin` files exist

## File Structure After Setup

```
data/ogbn-products/
├── features.bin          # Node features (2449029 * 100 * 4 bytes)
├── weights.bin           # GCN weights (100 * 16 * 4 bytes)
├── adj_values.bin        # Adjacency values (61859140 * 4 bytes)
├── adj_col_indices.bin   # Column indices (61859140 * 4 bytes)
├── adj_row_ptr.bin       # Row pointers (2449030 * 4 bytes)
├── features.npy          # Intermediate format
├── weights.npy           # Intermediate format
└── adjacency.npz         # Intermediate format

models/
└── gcn_model.pth         # Trained PyTorch model
```

## Next Steps

1. **Verify data**: Check file sizes match expected values
2. **Build kernel**: `make kernel TARGET=hw`
3. **Run inference**: Use the host application with `--data_dir data/ogbn-products`
4. **Compare results**: Use `--enable_golden_check` to verify correctness

## Notes

- The HLS kernel only uses the **first GCN layer** (100 -> 16)
- The second layer in the model is only used during training
- Adjacency normalization should match what was used during training (symmetric by default)
- For hw_emu, the kernel automatically uses TEST_MODE with smaller dimensions
- For full dataset, use `TARGET=hw` (hardware build)

