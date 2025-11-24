# Graph Tensor Layout

The host application consumes five binary blobs, all stored in little-endian
format and written as raw contiguous arrays:

| File | Type | Element Count | Notes |
| --- | --- | --- | --- |
| `features.bin` | float32 | `NUM_NODES * IN_FEATURES` | Row-major node features |
| `weights.bin` | float32 | `IN_FEATURES * OUT_FEATURES` | First GCN layer weights |
| `adj_values.bin` | float32 | `NUM_EDGES_NNZ` | CSR data array |
| `adj_col_indices.bin` | int32 | `NUM_EDGES_NNZ` | CSR column indices |
| `adj_row_ptr.bin` | int32 | `NUM_NODES + 1` | CSR row pointer |

Use `scripts/prepare_graph_data.py` to convert numpy/scipy tensors into this
layout or pass `--use_synthetic` to the host executable for a deterministic
self-loop graph useful for bring-up.
