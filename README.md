# GNN Inference on Alveo U50

End-to-end reference design for running a single-layer GCN inference kernel on
AMD/Xilinx Alveo U50 cards. The repository contains:

- High-Level Synthesis (HLS) kernel `gnn` plus C-sim/co-sim scaffolding
- Vitis kernel build scripts to produce `.xo`/`.xclbin`
- Host application using the modern XRT C++ API
- Utilities for preparing graph tensors for inference and synthetic data for
  bring-up

## Repository Layout

```
├── kernel/            # HLS sources, kernel Makefile, link config
├── tb/                # C-simulation / co-simulation testbench
├── host/              # Host application sources + Makefile
├── scripts/           # Data preparation helpers
├── data/              # Expected location for packed graph tensors
├── build/             # Generated artifacts (ignored)
├── Makefile           # Top-level build orchestration
└── hls_csim.cfg       # Shared HLS configuration
```

## Prerequisites

- AMD Vitis 2023.2+ with XRT installed on the host machine
- `XILINX_XRT` and (optionally) `XILINX_VITIS` environment variables set
- Target platform: `xilinx_u50_gen3x16_xdma_5_202210_1` (override via
  `PLATFORM` make variable if needed)
- Python 3.8+ with `numpy` and `scipy` for packing real graph tensors

## Quick Start

```bash
# 1. Run C-simulation (software-only)
make csim

# 2. Run co-simulation (compiles RTL + sim)
make cosim

# 3. Build kernel and host for hardware target
make all TARGET=hw PLATFORM=<path/to>.xpfm

# 4. Run on hardware (assumes xclbin + tensors ready)
make run TARGET=hw PLATFORM=<path/to>.xpfm \
    XCLBIN=kernel/build/gcn_layer_hls.hw.xclbin
```

### Targets

- `make csim` — Software validation against the CPU golden model
- `make cosim` — RTL co-simulation (runs `csynth` automatically)
- `make kernel TARGET=hw_emu` — Produce `.xclbin` for hardware emulation
- `make kernel TARGET=hw` — Produce `.xclbin` for hardware runs
- `make host` — Compile the host application (requires `XILINX_XRT`)
- `make run TARGET=hw_emu` — Launch hardware emulation (sets
  `XCL_EMULATION_MODE=hw_emu` and uses the `.hw_emu.xclbin`)
- `make run TARGET=hw` — Launch on real hardware

## Host Application

```
./host/build/host.exe --xclbin_file <path> --data_dir data/cora \
    [--device_id 0] [--use_synthetic] [--enable_golden_check]
```

- `--use_synthetic` generates a deterministic self-loop dataset matching the
  kernel dimensions. Useful for bring-up without real data.
- `--enable_golden_check` runs the CPU reference model and compares results.

## Graph Tensor Preparation

**📖 For detailed step-by-step instructions, see [`scripts/QUICKSTART.md`](scripts/QUICKSTART.md)**

### Quick Methods

**From PyTorch Geometric (Cora dataset):**
```bash
python scripts/extract_pytorch_model.py \
    --model_path models/gcn_model.pth \
    --output_dir data/cora
```

**From NumPy/Scipy (manual):**
```bash
python scripts/prepare_graph_data.py \
    --features features.npy \
    --weights weights.npy \
    --adjacency adjacency.npz \
    --out_dir data/cora
```

**For testing (synthetic data):**
```bash
./host/build/host.exe --use_synthetic --enable_golden_check
```

The host expects 5 binary files in the data directory (see `data/README.md` for format details).

## Hardware Emulation vs Hardware

| Mode     | Build Command                             | Run Command                                   |
|----------|-------------------------------------------|-----------------------------------------------|
| `hw_emu` | `make kernel TARGET=hw_emu`                | `make TARGET=hw_emu run`                       |
| `hw`     | `make kernel TARGET=hw`                    | `make TARGET=hw run`                           |

For `hw_emu`, `make run` automatically generates `emconfig.json` and exports
`XCL_EMULATION_MODE=hw_emu`.

## Troubleshooting

- `XRT build failed`: ensure `XILINX_XRT` points to your XRT install (usually
  `/opt/xilinx/xrt`).
- `Cannot find features.bin`: run `scripts/prepare_graph_data.py` or pass
  `--use_synthetic` to the host for smoke testing.
- `Mismatch vs golden`: verify that the preprocessing used the same normalized
  adjacency and layer weights as the deployed kernel.

## License

See [LICENSE](LICENSE).
