#!/usr/bin/env python3
"""Convert numpy/scipy tensors into the binary layout consumed by the host."""

import argparse
import pathlib

import numpy as np
import scipy.sparse as sp

# ogbn-products dataset dimensions
NUM_NODES = 2449029
IN_FEATURES = 100
OUT_FEATURES = 47  # num_classes for ogbn-products


def write_binary(path: pathlib.Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    print(f"[prepare_graph_data] wrote {array.size} values to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help=".npy with shape (NUM_NODES, IN_FEATURES)")
    parser.add_argument("--weights", required=True, help=".npy with shape (IN_FEATURES, OUT_FEATURES)")
    parser.add_argument("--adjacency", required=True, help=".npz CSR saved via scipy.sparse.save_npz")
    parser.add_argument("--out_dir", required=True, help="Directory to place *.bin outputs")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(args.features).astype(np.float32)
    if features.shape != (NUM_NODES, IN_FEATURES):
        raise ValueError(f"features shape {features.shape} != ({NUM_NODES}, {IN_FEATURES})")

    weights = np.load(args.weights).astype(np.float32)
    if weights.shape != (IN_FEATURES, OUT_FEATURES):
        raise ValueError(f"weights shape {weights.shape} != ({IN_FEATURES}, {OUT_FEATURES})")

    adj = sp.load_npz(args.adjacency).tocsr().astype(np.float32)
    if adj.shape != (NUM_NODES, NUM_NODES):
        raise ValueError(f"adjacency shape {adj.shape} != ({NUM_NODES}, {NUM_NODES})")

    write_binary(out_dir / "features.bin", features.reshape(-1))
    write_binary(out_dir / "weights.bin", weights.reshape(-1))
    write_binary(out_dir / "adj_values.bin", adj.data)
    write_binary(out_dir / "adj_col_indices.bin", adj.indices.astype(np.int32))
    write_binary(out_dir / "adj_row_ptr.bin", adj.indptr.astype(np.int32))


if __name__ == "__main__":
    main()
