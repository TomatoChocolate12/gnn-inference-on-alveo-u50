#!/usr/bin/env python3
"""
Extract single GCN layer weights and Cora graph data from a PyTorch Geometric model.

This script extracts the first GCN layer from a trained model and prepares the Cora
dataset for inference on Alveo U50. The HLS kernel expects a single layer with
specific dimensions (1433 -> 16 features).

Example usage:
    python scripts/extract_pytorch_model.py \
        --model_path models/gcn_model.pth \
        --output_dir data/cora
"""

import argparse
import pathlib

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_scipy_sparse_matrix


def extract_gcn_weights(model, layer_name="conv1"):
    """
    Extract weights from a GCN layer.
    
    Args:
        model: PyTorch model with GCN layers
        layer_name: Name of the GCN layer to extract (default: "conv1")
    
    Returns:
        numpy array of shape (in_features, out_features)
    """
    if not hasattr(model, layer_name):
        # Try to find first GCNConv layer
        for name, module in model.named_modules():
            if isinstance(module, GCNConv):
                layer = module
                break
        else:
            raise ValueError("Could not find GCNConv layer in model")
    else:
        layer = getattr(model, layer_name)
    
    # Extract weight matrix (shape: [out_features, in_features])
    weight = layer.weight.data.cpu().numpy()
    # Transpose to [in_features, out_features] for our kernel
    weight = weight.T
    return weight.astype(np.float32)


def load_cora_dataset(root="data"):
    """Load the Cora dataset."""
    dataset = Planetoid(root=root, name="Cora", transform=NormalizeFeatures())
    data = dataset[0]
    return data, dataset.num_node_features, dataset.num_classes


def prepare_adjacency(edge_index, num_nodes, normalization="sym"):
    """
    Convert edge_index to normalized CSR adjacency matrix.
    
    Args:
        edge_index: torch.Tensor of shape [2, num_edges]
        num_nodes: Number of nodes in the graph
        normalization: "sym" (symmetric) or "row" (row-normalized)
    
    Returns:
        scipy.sparse.csr_matrix with normalized adjacency
    """
    # Convert to scipy sparse matrix
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    
    # Normalize
    if normalization == "sym":
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5, where=deg != 0)
        deg_inv_sqrt[deg == 0] = 0
        deg_matrix = sp.diags(deg_inv_sqrt)
        adj = deg_matrix @ adj @ deg_matrix
    elif normalization == "row":
        # Row normalization: D^(-1) A
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv = np.power(deg, -1, where=deg != 0)
        deg_inv[deg == 0] = 0
        deg_matrix = sp.diags(deg_inv)
        adj = deg_matrix @ adj
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    
    return adj.astype(np.float32).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="Path to saved PyTorch model (.pth file)")
    parser.add_argument("--dataset_root", default="data", help="Root directory for Cora dataset")
    parser.add_argument("--layer_name", default="conv1", help="Name of first GCN layer to extract (default: conv1)")
    parser.add_argument("--output_dir", required=True, help="Output directory for binary files")
    parser.add_argument("--normalization", default="sym", choices=["sym", "row"],
                       help="Adjacency normalization method (default: sym)")
    args = parser.parse_args()
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = torch.load(args.model_path, map_location="cpu")
    model.eval()
    
    # Extract weights from first GCN layer (single layer as HLS expects)
    print(f"Extracting weights from first GCN layer '{args.layer_name}'")
    print("  Note: HLS kernel expects a single layer (1433 -> 16 features)")
    weights = extract_gcn_weights(model, args.layer_name)
    print(f"  Weight shape: {weights.shape} (should be [1433, 16])")
    
    # Load Cora dataset
    print(f"Loading Cora dataset from {args.dataset_root}")
    data, num_features, num_classes = load_cora_dataset(args.dataset_root)
    print(f"  Nodes: {data.num_nodes}, Features: {num_features}, Classes: {num_classes}")
    
    # Extract node features
    features = data.x.numpy().astype(np.float32)
    print(f"  Features shape: {features.shape}")
    
    # Prepare adjacency matrix
    print(f"Preparing adjacency matrix (normalization: {args.normalization})")
    adj = prepare_adjacency(data.edge_index, data.num_nodes, args.normalization)
    print(f"  Adjacency shape: {adj.shape}, NNZ: {adj.nnz}")
    
    # Verify dimensions match kernel constants (Cora-specific)
    # These should match kernel/gnn_hls.h for Cora dataset
    NUM_NODES = 2708
    IN_FEATURES = 1433
    OUT_FEATURES = 16
    NUM_EDGES_NNZ = 10556
    
    print("\nVerifying dimensions match HLS kernel constants (Cora):")
    errors = []
    if features.shape[0] != NUM_NODES:
        errors.append(f"Nodes: {features.shape[0]} != {NUM_NODES}")
    if features.shape[1] != IN_FEATURES:
        errors.append(f"Input features: {features.shape[1]} != {IN_FEATURES}")
    if weights.shape[0] != IN_FEATURES:
        errors.append(f"Weight input dim: {weights.shape[0]} != {IN_FEATURES}")
    if weights.shape[1] != OUT_FEATURES:
        errors.append(f"Output features: {weights.shape[1]} != {OUT_FEATURES}")
    if adj.nnz != NUM_EDGES_NNZ:
        errors.append(f"Adjacency NNZ: {adj.nnz} != {NUM_EDGES_NNZ}")
    
    if errors:
        print("  ERROR: Dimension mismatches detected!")
        for err in errors:
            print(f"    - {err}")
        print("  You need to update kernel/gnn_hls.h and rebuild the kernel")
        return 1
    else:
        print("  ✓ All dimensions match!")
    
    # Save as numpy arrays (intermediate format)
    print("\nSaving intermediate .npy files...")
    features_path = output_dir / "features.npy"
    weights_path = output_dir / "weights.npy"
    adj_path = output_dir / "adjacency.npz"
    
    np.save(features_path, features)
    np.save(weights_path, weights)
    sp.save_npz(adj_path, adj)
    
    print(f"  Saved {features_path}")
    print(f"  Saved {weights_path}")
    print(f"  Saved {adj_path}")
    
    # Convert to binary format using prepare_graph_data.py
    print("\nConverting to binary format...")
    import subprocess
    import sys
    
    script_path = pathlib.Path(__file__).parent / "prepare_graph_data.py"
    subprocess.run([
        sys.executable, str(script_path),
        "--features", str(features_path),
        "--weights", str(weights_path),
        "--adjacency", str(adj_path),
        "--out_dir", str(output_dir)
    ], check=True)
    
    print(f"\n✓ Successfully prepared data in {output_dir}")
    print(f"  You can now run inference with:")
    print(f"    ./host/build/host.exe --xclbin_file <path> --data_dir {output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

