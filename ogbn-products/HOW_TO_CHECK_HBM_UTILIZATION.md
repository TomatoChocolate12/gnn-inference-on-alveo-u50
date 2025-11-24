# How to Check HBM Utilization

## Overview
Your design uses HBM (High Bandwidth Memory) channels as configured in `kernel/link.cfg`:
- `HBM[0]`: h_in and h_out
- `HBM[1]`: w (weights)
- `HBM[2]`: adj_values
- `HBM[3]`: adj_col_indices and adj_row_ptr

## Methods to Check HBM Utilization

### 1. **After Link Step - System Estimate Reports**

After running the link step, check the system estimate report:

```bash
# View the text report
cat kernel/_x/reports/link/system_estimate_gcn_layer_hls.hw_emu.xtxt

# Or open the HTML guidance report
firefox kernel/_x/reports/link/v++_link_gcn_layer_hls.hw_emu_guidance.html
```

The HTML report contains detailed memory connectivity and utilization information.

### 2. **Using Vitis Analyzer (Recommended)**

Vitis Analyzer provides the most comprehensive view of HBM utilization:

```bash
# Open the compile summary (after kernel compilation)
vitis --analyze kernel/build/gcn_layer_hls.hw_emu.xo.compile_summary

# Or open the link summary (after linking)
vitis --analyze kernel/build/gcn_layer_hls.hw_emu.xclbin.link_summary
```

In Vitis Analyzer:
1. Navigate to **System Diagram** → **Memory Connectivity**
2. Look for **HBM** section
3. Check **Memory Utilization** tab
4. View **Bandwidth Analysis** for HBM channels

### 3. **After Full Hardware Build**

After building for hardware (not emulation), check Vivado reports:

```bash
# Navigate to the link directory
cd kernel/_x/link/int

# Open Vivado project (if available)
vivado -mode batch -source <project_file>

# Or check reports directly
find kernel/_x/link -name "*utilization*" -o -name "*memory*" -o -name "*hbm*"
```

### 4. **Runtime Profiling (Hardware Only)**

For runtime HBM bandwidth and utilization:

#### Using XRT Profiling:
```bash
# Enable profiling
export XRT_PROFILE=true
export XRT_PROFILE_DIR=./profile

# Run your application
./host/build/host.exe --xclbin_file kernel/build/gcn_layer_hls.hw.xclbin

# View profile data
xbutil examine -d 0 -r memory
```

#### Using Vitis Analyzer for Runtime:
```bash
# Generate profile summary
xbutil examine -d 0 -r memory -f json -o profile.json

# Open in Vitis Analyzer
vitis --analyze profile.json
```

### 5. **Check HBM Channel Mapping**

Verify which HBM channels are used:

```bash
# Check the link configuration
cat kernel/link.cfg

# Check connectivity in logs
grep -i "HBM\|hbm" kernel/_x/logs/link/v++.log
```

### 6. **Manual Calculation**

Based on your data sizes:

```python
# HBM Channel 0 (h_in + h_out)
h_in_size = NUM_NODES * IN_FEATURES * 2  # bytes (ap_fixed<16,6> = 2 bytes)
h_out_size = NUM_NODES * OUT_FEATURES * 2
hbm0_total = h_in_size + h_out_size

# HBM Channel 1 (w)
w_size = IN_FEATURES * OUT_FEATURES * 2

# HBM Channel 2 (adj_values)
adj_val_size = NUM_EDGES_NNZ * 2

# HBM Channel 3 (adj_col_indices + adj_row_ptr)
adj_col_size = NUM_EDGES_NNZ * 4  # int32 = 4 bytes
adj_row_size = (NUM_NODES + 1) * 4
hbm3_total = adj_col_size + adj_row_size

# Each HBM channel is ~256MB (32 channels × 8MB each)
# Check utilization percentage
hbm0_util = (hbm0_total / (256 * 1024 * 1024)) * 100
hbm1_util = (w_size / (256 * 1024 * 1024)) * 100
hbm2_util = (adj_val_size / (256 * 1024 * 1024)) * 100
hbm3_util = (hbm3_total / (256 * 1024 * 1024)) * 100
```

### 7. **Quick Check Script**

Create a simple script to check HBM usage:

```bash
#!/bin/bash
# check_hbm_usage.sh

echo "=== HBM Channel Usage ==="
echo ""
echo "Channel 0 (h_in + h_out):"
echo "  h_in:  $(echo "scale=2; 2449029 * 100 * 2 / 1024 / 1024" | bc) MB"
echo "  h_out: $(echo "scale=2; 2449029 * 47 * 2 / 1024 / 1024" | bc) MB"
echo ""
echo "Channel 1 (w):"
echo "  w:     $(echo "scale=2; 100 * 47 * 2 / 1024" | bc) KB"
echo ""
echo "Channel 2 (adj_values):"
echo "  adj_values: $(echo "scale=2; 61859140 * 2 / 1024 / 1024" | bc) MB"
echo ""
echo "Channel 3 (adj_col_indices + adj_row_ptr):"
echo "  adj_col: $(echo "scale=2; 61859140 * 4 / 1024 / 1024" | bc) MB"
echo "  adj_row: $(echo "scale=2; 2449030 * 4 / 1024 / 1024" | bc) MB"
```

## Expected HBM Utilization

For ogbn-products dataset:
- **HBM[0]**: ~467 MB (h_in) + ~220 MB (h_out) = ~687 MB / 256 MB = **268%** (needs multiple channels or DDR)
- **HBM[1]**: ~9 KB (very small)
- **HBM[2]**: ~118 MB
- **HBM[3]**: ~236 MB (adj_col) + ~9 MB (adj_row) = ~245 MB

**Note**: HBM[0] exceeds single channel capacity. You may need to:
1. Split across multiple HBM channels
2. Use DDR instead for large buffers
3. Process data in tiles/chunks

## Troubleshooting

If HBM utilization is high:
1. Check if data fits in allocated channels
2. Consider using DDR for very large buffers
3. Optimize data layout and access patterns
4. Use multiple HBM channels for large datasets

## References

- Xilinx HBM Documentation: [UG1393](https://www.xilinx.com/support/documentation/ip_documentation/hbm/v1_0/pg276-axi-hbm.pdf)
- Vitis Analyzer User Guide
- XRT Profiling Guide

