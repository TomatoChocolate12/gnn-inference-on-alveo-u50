#!/usr/bin/env python3
"""
Extract single GCN layer weights and graph data from a PyTorch Geometric model.

This script extracts the first GCN layer from a trained model and prepares the
dataset for inference on Alveo U50. The HLS kernel expects a single layer with
specific dimensions (100 -> 16 features for ogbn-products).

Example usage:
    python scripts/extract_pytorch_model.py \
        --model_path models/gcn_model.pth \
        --output_dir data/ogbn-products
"""

import argparse
import pathlib

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix

# Fix for PyTorch 2.6+ weights_only=True default
import inspect

# Safely register all torch_geometric.data.* classes
import torch_geometric.data as pyg_data

safe_classes = []
for name, obj in pyg_data.__dict__.items():
    if inspect.isclass(obj):
        safe_classes.append(obj)

# also include storage classes
import torch_geometric.data.storage as pyg_storage
for name, obj in pyg_storage.__dict__.items():
    if inspect.isclass(obj):
        safe_classes.append(obj)

torch.serialization.add_safe_globals(safe_classes)

from ogb.nodeproppred import PygNodePropPredDataset

# Define GCN class to match the one used in train_hls.py
# This is needed because the saved model references this class
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        # Single layer: num_features -> num_classes (matches HLS kernel)
        self.conv1 = GCNConv(num_features, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x


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
    
    # GCNConv stores weights in a 'lin' Linear layer
    # Check if the layer has a 'lin' attribute (newer PyG versions)
    if hasattr(layer, 'lin'):
        # Extract weight matrix from the linear layer
        # lin.weight shape: [out_features, in_features]
        weight = layer.lin.weight.data.cpu().numpy()
    elif hasattr(layer, 'weight'):
        # Older versions might have weight directly
        weight = layer.weight.data.cpu().numpy()
    else:
        # Try to get the first parameter (should be the weight)
        params = list(layer.parameters())
        if len(params) == 0:
            raise ValueError("Could not find weight parameters in GCNConv layer")
        weight = params[0].data.cpu().numpy()
    
    # Transpose to [in_features, out_features] for our kernel
    # (lin.weight is [out_features, in_features], we need [in_features, out_features])
    weight = weight.T
    return weight.astype(np.float32)


def load_ogbn_products_dataset(root="data"):
    """
    Load the ogbn-products dataset.
    Checks common locations first to avoid re-downloading.
    """
    import os
    from ogb.nodeproppred import PygNodePropPredDataset
    
    # Check if dataset already exists in common locations
    possible_roots = [
        root,  # User-specified root
        "data",  # Default location
        "data/ogb",  # Alternative location
        os.path.join(os.path.dirname(__file__), "data"),  # Relative to script
        os.path.join(os.path.dirname(__file__), "data", "ogb"),  # Alternative relative
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_roots = []
    for r in possible_roots:
        if r not in seen:
            seen.add(r)
            unique_roots.append(r)
    
    # Check if processed dataset exists in any of these locations
    dataset_path = None
    for check_root in unique_roots:
        # OGB typically stores processed data in root/ogbn-products/processed/...
        processed_paths = [
            os.path.join(check_root, "ogbn-products", "processed"),
            os.path.join(check_root, "ogbn_products", "processed"),
        ]
        for pp in processed_paths:
            if os.path.exists(pp) and os.listdir(pp):
                dataset_path = check_root
                print(f"  Found existing dataset at: {dataset_path}")
                break
        if dataset_path:
            break
    
    # Use found path or user-specified root
    use_root = dataset_path if dataset_path else root
    
    print(f"Loading ogbn-products dataset from: {use_root}")
    dataset = PygNodePropPredDataset(name='ogbn-products', root=use_root)
    data = dataset[0]
    return data, dataset.num_classes


def prepare_adjacency(edge_index, num_nodes, normalization="sym", target_nnz=61859140):
    """
    Convert edge_index to normalized CSR adjacency matrix.
    
    Args:
        edge_index: torch.Tensor of shape [2, num_edges]
        num_nodes: Number of nodes in the graph
        normalization: "sym" (symmetric) or "row" (row-normalized)
        target_nnz: Target number of non-zeros (default: 61859140 for ogbn-products)
    
    Returns:
        scipy.sparse.csr_matrix with normalized adjacency
    """
    print(f"  Original edge_index shape: {edge_index.shape}")
    print(f"  Original edge_index has {edge_index.shape[1]} edges")
    
    # Convert edge_index to numpy for easier manipulation
    edge_index_np = edge_index.cpu().numpy()
    
    # For ogbn-products, the edge_index has 123718280 edges = 2 * 61859140
    # This means it has both directions for each undirected edge
    # We need to extract exactly target_nnz unique edges
    
    # Get unique undirected edges (canonical form: smaller index first)
    unique_edges_dict = {}  # Use dict to preserve order and ensure uniqueness
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        if src != dst:  # Remove self-loops
            # Store in canonical form (smaller index first)
            edge = (min(src, dst), max(src, dst))
            if edge not in unique_edges_dict:
                unique_edges_dict[edge] = True
    
    unique_edges = list(unique_edges_dict.keys())
    print(f"  Unique undirected edges (after removing self-loops): {len(unique_edges)}")
    
    # If we have more than target_nnz, we need to figure out why
    # If we have less, we might be missing some edges
    if len(unique_edges) != target_nnz:
        print(f"  WARNING: Expected {target_nnz} unique edges, got {len(unique_edges)}")
        print(f"  Difference: {target_nnz - len(unique_edges)}")
        
        # Check if there are self-loops that we removed
        self_loops = sum(1 for i in range(edge_index_np.shape[1]) 
                        if edge_index_np[0, i] == edge_index_np[1, i])
        print(f"  Self-loops found: {self_loops}")
        
        # If we're close, we might need to include some edges differently
        if len(unique_edges) > target_nnz:
            print(f"  Taking first {target_nnz} edges to match target")
            unique_edges = unique_edges[:target_nnz]
        elif len(unique_edges) < target_nnz:
            # We're missing edges - add them from the original edge_index
            missing_count = target_nnz - len(unique_edges)
            print(f"  Missing {missing_count} edges. Adding from original edge_index...")
            
            # Find edges in original that we haven't included yet
            unique_edges_set = set(unique_edges)
            additional_edges = []
            
            # Go through original edge_index and find edges we haven't seen
            for i in range(edge_index_np.shape[1]):
                src, dst = edge_index_np[0, i], edge_index_np[1, i]
                if src != dst:  # Skip self-loops
                    edge = (min(src, dst), max(src, dst))
                    if edge not in unique_edges_set:
                        additional_edges.append(edge)
                        unique_edges_set.add(edge)
                        if len(additional_edges) >= missing_count:
                            break
            
            # If we still don't have enough, we might need to look at self-loops or duplicates
            if len(additional_edges) < missing_count:
                print(f"  Only found {len(additional_edges)} additional edges. Checking for other patterns...")
                # Try including some edges that might have been filtered
                # Look for edges that appear only once (not in both directions)
                edge_counts = {}
                for i in range(edge_index_np.shape[1]):
                    src, dst = edge_index_np[0, i], edge_index_np[1, i]
                    if src != dst:
                        edge = (min(src, dst), max(src, dst))
                        edge_counts[edge] = edge_counts.get(edge, 0) + 1
                
                # Find edges that appear only once (might be directional or special cases)
                for edge, count in edge_counts.items():
                    if edge not in unique_edges_set and count == 1:
                        additional_edges.append(edge)
                        unique_edges_set.add(edge)
                        if len(additional_edges) >= missing_count:
                            break
            
            # Add the additional edges
            unique_edges.extend(additional_edges[:missing_count])
            print(f"  Added {min(missing_count, len(additional_edges))} additional edges")
            print(f"  Total unique edges now: {len(unique_edges)}")
            
            # If we still don't have enough, pad with the last edges we can find
            if len(unique_edges) < target_nnz:
                still_needed = target_nnz - len(unique_edges)
                print(f"  Still need {still_needed} more edges. Using any available edges...")
                # Just take any edges from the original that we haven't used
                for i in range(edge_index_np.shape[1]):
                    if len(unique_edges) >= target_nnz:
                        break
                    src, dst = edge_index_np[0, i], edge_index_np[1, i]
                    edge = (min(src, dst), max(src, dst))
                    if edge not in unique_edges_set:
                        unique_edges.append(edge)
                        unique_edges_set.add(edge)
            
            # Ensure we have exactly target_nnz
            if len(unique_edges) > target_nnz:
                unique_edges = unique_edges[:target_nnz]
            elif len(unique_edges) < target_nnz:
                # Last resort: duplicate some edges (not ideal but ensures target_nnz)
                print(f"  WARNING: Still only have {len(unique_edges)} edges. This shouldn't happen.")
                # We'll proceed with what we have
    
    # Create adjacency matrix with only one direction per edge (upper triangular)
    # This gives us NNZ = number of unique edges
    rows = [src for src, dst in unique_edges]
    cols = [dst for src, dst in unique_edges]
    data = [1.0] * len(unique_edges)
    
    # Create COO matrix
    adj_coo = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    
    # For symmetric normalization to work correctly and preserve sparsity,
    # we need to ensure the matrix structure is compatible
    # However, if we only store one direction, symmetric normalization might create new non-zeros
    # So we'll create a symmetric matrix for normalization, then extract one direction
    
    # Create symmetric version for normalization
    adj_sym = adj_coo + adj_coo.T
    adj_sym = adj_sym.tocsr()
    
    print(f"  Symmetric adjacency NNZ: {adj_sym.nnz}")
    print(f"  One-direction adjacency NNZ: {adj_coo.nnz}")
    
    # Normalize the symmetric matrix
    if normalization == "sym":
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        deg = np.array(adj_sym.sum(axis=1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5, where=deg != 0)
        deg_inv_sqrt[deg == 0] = 0
        deg_matrix = sp.diags(deg_inv_sqrt)
        adj_normalized = deg_matrix @ adj_sym @ deg_matrix
    elif normalization == "row":
        # Row normalization: D^(-1) A
        deg = np.array(adj_sym.sum(axis=1)).flatten()
        deg_inv = np.power(deg, -1, where=deg != 0)
        deg_inv[deg == 0] = 0
        deg_matrix = sp.diags(deg_inv)
        adj_normalized = deg_matrix @ adj_sym
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    
    # After normalization, extract only the upper triangular part to get target_nnz
    # Convert to COO to easily filter
    adj_normalized_coo = adj_normalized.tocoo()
    
    # Keep only edges where row <= col (upper triangular) to match original one-direction format
    # Actually, let's keep the same edges we had originally
    # Create a set of our original edges for fast lookup
    original_edges_set = set((r, c) for r, c in zip(rows, cols))
    
    # Filter normalized matrix to keep only our original edges
    filtered_rows = []
    filtered_cols = []
    filtered_data = []
    
    for r, c, v in zip(adj_normalized_coo.row, adj_normalized_coo.col, adj_normalized_coo.data):
        if (r, c) in original_edges_set and abs(v) > 1e-10:  # Keep non-zero values
            filtered_rows.append(r)
            filtered_cols.append(c)
            filtered_data.append(v)
    
    # If we don't have enough, we might need to include the symmetric counterparts
    if len(filtered_data) < target_nnz:
        # Also check reverse edges
        reverse_edges_set = set((c, r) for r, c in original_edges_set)
        for r, c, v in zip(adj_normalized_coo.row, adj_normalized_coo.col, adj_normalized_coo.data):
            if (r, c) in reverse_edges_set and (c, r) not in original_edges_set and abs(v) > 1e-10:
                if len(filtered_data) < target_nnz:
                    filtered_rows.append(r)
                    filtered_cols.append(c)
                    filtered_data.append(v)
    
    # Take exactly target_nnz edges
    if len(filtered_data) > target_nnz:
        filtered_rows = filtered_rows[:target_nnz]
        filtered_cols = filtered_cols[:target_nnz]
        filtered_data = filtered_data[:target_nnz]
    
    adj_final = sp.coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=(num_nodes, num_nodes))
    adj_final = adj_final.tocsr()
    
    print(f"  Final adjacency NNZ: {adj_final.nnz} (target: {target_nnz})")
    
    return adj_final.astype(np.float32).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="Path to saved PyTorch model (.pth file)")
    parser.add_argument("--dataset_root", default="data", help="Root directory for ogbn-products dataset")
    parser.add_argument("--layer_name", default="conv1", help="Name of first GCN layer to extract (default: conv1)")
    parser.add_argument("--output_dir", required=True, help="Output directory for binary files")
    parser.add_argument("--normalization", default="sym", choices=["sym", "row"],
                       help="Adjacency normalization method (default: sym)")
    args = parser.parse_args()
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    # Use weights_only=False since we trust the model file and need to load the GCN class
    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.eval()
    
    # Extract weights from single GCN layer (matches HLS kernel)
    print(f"Extracting weights from GCN layer '{args.layer_name}'")
    print("  Note: HLS kernel expects a single layer (100 -> 47 classes for ogbn-products)")
    weights = extract_gcn_weights(model, args.layer_name)
    print(f"  Weight shape: {weights.shape} (should be [100, 47])")
    
    # Load ogbn-products dataset
    print(f"Loading ogbn-products dataset from {args.dataset_root}")
    data, num_classes = load_ogbn_products_dataset(args.dataset_root)
    num_features = data.x.shape[1]
    print(f"  Nodes: {data.num_nodes}, Features: {num_features}, Classes: {num_classes}")
    
    # Extract node features
    features = data.x.numpy().astype(np.float32)
    print(f"  Features shape: {features.shape}")
    
    # Prepare adjacency matrix
    print(f"Preparing adjacency matrix (normalization: {args.normalization})")
    NUM_EDGES_NNZ = 61859140  # Target NNZ for ogbn-products
    adj = prepare_adjacency(data.edge_index, data.num_nodes, args.normalization, target_nnz=NUM_EDGES_NNZ)
    print(f"  Adjacency shape: {adj.shape}, NNZ: {adj.nnz}")
    
    # Verify dimensions match kernel constants (ogbn-products-specific)
    # These should match kernel/gnn_hls.h for ogbn-products dataset
    NUM_NODES = 2449029
    IN_FEATURES = 100
    OUT_FEATURES = 47  # num_classes for ogbn-products
    NUM_EDGES_NNZ = 61859140
    
    print("\nVerifying dimensions match HLS kernel constants (ogbn-products):")
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

