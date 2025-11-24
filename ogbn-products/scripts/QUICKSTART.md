# Quick Start: Providing Model and Graph Data

This guide shows you how to provide your trained GNN model and graph data to the Alveo U50 inference pipeline.

## Overview

The inference pipeline expects **5 binary files** in a directory:
- `features.bin` - Node features (float32, row-major)
- `weights.bin` - GCN layer weights (float32)
- `adj_values.bin` - CSR adjacency values (float32)
- `adj_col_indices.bin` - CSR column indices (int32)
- `adj_row_ptr.bin` - CSR row pointers (int32)

## Method 1: From PyTorch Geometric Model (Cora)

If you have a trained PyTorch Geometric model with a single GCN layer:

```bash
# 1. Extract first GCN layer weights and Cora graph data
python scripts/extract_pytorch_model.py \
    --model_path models/my_gcn_model.pth \
    --output_dir data/cora

# 2. Run inference
./host/build/host.exe \
    --xclbin_file kernel/build/gcn_layer_hls.hw.xclbin \
    --data_dir data/cora \
    --enable_golden_check
```

**Note:** The HLS kernel expects a single GCN layer with dimensions 1433 -> 16 (Cora dataset).
The script automatically extracts the first GCN layer (`conv1` by default).

## Method 2: Manual Conversion from NumPy/Scipy

If you already have your data as NumPy arrays:

```bash
# 1. Prepare your data:
#    - features.npy: shape (NUM_NODES, IN_FEATURES), float32
#    - weights.npy: shape (IN_FEATURES, OUT_FEATURES), float32
#    - adjacency.npz: scipy.sparse CSR matrix, saved via scipy.sparse.save_npz()

# 2. Convert to binary format
python scripts/prepare_graph_data.py \
    --features features.npy \
    --weights weights.npy \
    --adjacency adjacency.npz \
    --out_dir data/my_graph

# 3. Run inference
./host/build/host.exe \
    --xclbin_file kernel/build/gcn_layer_hls.hw.xclbin \
    --data_dir data/my_graph
```

## Method 3: Synthetic Data (Testing Only)

For quick testing without real data:

```bash
# Use built-in synthetic data generator
./host/build/host.exe \
    --xclbin_file kernel/build/gcn_layer_hls.hw.xclbin \
    --use_synthetic \
    --enable_golden_check
```

## Step-by-Step: PyTorch Example (Cora)

Here's a complete example for PyTorch Geometric with Cora:

### Step 1: Train and Save Your Model

```python
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Define model (HLS expects single layer: 1433 -> 16)
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)  # First layer: 1433 -> 16
        self.conv2 = GCNConv(16, num_classes)    # Second layer (not used by HLS)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Load Cora dataset
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]

# Train model (your training code here)
model = GCN(dataset.num_node_features, dataset.num_classes)
# ... training loop ...

# Save model
torch.save(model, 'models/gcn_model.pth')
```

### Step 2: Extract Single Layer for Inference

```bash
python scripts/extract_pytorch_model.py \
    --model_path models/gcn_model.pth \
    --output_dir data/cora
```

This will:
- Extract the **first GCN layer weights** (`conv1`: 1433 -> 16) - this is what HLS expects
- Extract node features from Cora dataset
- Build normalized adjacency matrix (symmetric normalization)
- Verify dimensions match kernel constants
- Convert everything to binary format

### Step 3: Verify Dimensions

The kernel has hardcoded dimensions in `kernel/gnn_hls.h`:
- `NUM_NODES = 2708`
- `IN_FEATURES = 1433`
- `OUT_FEATURES = 16`
- `NUM_EDGES_NNZ = 10556`

If your graph has different dimensions, you'll need to:
1. Update these constants in `kernel/gnn_hls.h`
2. Rebuild the kernel: `make csynth cosim kernel`

### Step 4: Run Inference

```bash
# For hardware emulation
make run TARGET=hw_emu

# For real hardware
make run TARGET=hw
```

Or manually:

```bash
./host/build/host.exe \
    --xclbin_file kernel/build/gcn_layer_hls.hw.xclbin \
    --data_dir data/cora \
    --enable_golden_check \
    --device_id 0
```

## Troubleshooting

### "Failed to open file: features.bin"
- Make sure you've run the extraction script and the `data_dir` path is correct
- Check that all 5 `.bin` files exist in the directory

### "Unexpected element count"
- Your graph dimensions don't match the kernel constants
- Either update the kernel constants or use a graph that matches

### "Could not find GCN layer"
- The extraction script couldn't find a GCN layer in your model
- Try specifying `--layer_name` or `--layer_index` explicitly
- Check your model architecture

### Dimension Mismatches
- The kernel is compiled with specific dimensions
- If your model has different dimensions, you must recompile the kernel
- See `kernel/gnn_hls.h` for the current constants

## Next Steps

- See `README.md` for full build and run instructions
- See `data/README.md` for binary file format details
- Use `--enable_golden_check` to verify correctness against CPU reference

