#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../kernel/gnn_hls.h" // Ensure this is in your include path

// --- CPU Reference Implementation ---
// Matches the logic in host/include/graph_loader.hpp but self-contained for HLS TB
void cpu_reference(
    const std::vector<hls_dtype>& h_in,
    const std::vector<hls_dtype>& w,
    const std::vector<hls_dtype>& adj_values,
    const std::vector<int>& adj_col_indices,
    const std::vector<int>& adj_row_ptr,
    std::vector<hls_dtype>& h_out
) {
    std::vector<hls_dtype> aggregated(NUM_NODES * IN_FEATURES, (hls_dtype)0);

    // 1. Aggregation (SpMM)
    for (int node = 0; node < NUM_NODES; ++node) {
        int start = adj_row_ptr[node];
        int end = adj_row_ptr[node + 1];
        for (int idx = start; idx < end; ++idx) {
            int col = adj_col_indices[idx];
            hls_dtype val = adj_values[idx];
            for (int fin = 0; fin < IN_FEATURES; ++fin) {
                aggregated[node * IN_FEATURES + fin] += val * h_in[col * IN_FEATURES + fin];
            }
        }
    }

    // 2. GEMM + ReLU
    for (int node = 0; node < NUM_NODES; ++node) {
        for (int fout = 0; fout < OUT_FEATURES; ++fout) {
            hls_dtype sum = 0;
            for (int fin = 0; fin < IN_FEATURES; ++fin) {
                sum += aggregated[node * IN_FEATURES + fin] * w[fin * OUT_FEATURES + fout];
            }
            // FIX 1: Explicit casting to resolve ambiguity
            h_out[node * OUT_FEATURES + fout] = (sum > (hls_dtype)0) ? sum : (hls_dtype)0;
        }
    }
}

int main() {
    std::cout << "=== GNN HLS Testbench Started ===" << std::endl;
    std::cout << "Configuration: NUM_NODES=" << NUM_NODES 
              << " IN_FEATURES=" << IN_FEATURES 
              << " OUT_FEATURES=" << OUT_FEATURES << std::endl;

    // --- 1. Initialize Vectors ---
    std::vector<hls_dtype> h_in(NUM_NODES * IN_FEATURES);
    std::vector<hls_dtype> w(IN_FEATURES * OUT_FEATURES);
    std::vector<hls_dtype> adj_values(NUM_EDGES_NNZ);
    std::vector<int> adj_col_indices(NUM_EDGES_NNZ);
    std::vector<int> adj_row_ptr(NUM_NODES + 1);
    std::vector<hls_dtype> h_out_hls(NUM_NODES * OUT_FEATURES);
    std::vector<hls_dtype> h_out_gold(NUM_NODES * OUT_FEATURES);

    // --- 2. Generate Synthetic Data ---
    // We use simple patterns to make debugging easier
    std::cout << "Generating synthetic data..." << std::endl;

    for (int i = 0; i < NUM_NODES * IN_FEATURES; ++i) 
        h_in[i] = (hls_dtype)((i % 10) * 0.1);

    for (int i = 0; i < IN_FEATURES * OUT_FEATURES; ++i)
        w[i] = (hls_dtype)((i % 7) * 0.01);

    // Create a simple identity-like adjacency matrix + some neighbors
    // Each node connects to itself and (node+1)%NUM_NODES
    int edge_cnt = 0;
    adj_row_ptr[0] = 0;
    for (int i = 0; i < NUM_NODES; ++i) {
        // Self-loop
        if (edge_cnt < NUM_EDGES_NNZ) {
            adj_col_indices[edge_cnt] = i;
            adj_values[edge_cnt] = (hls_dtype)1.0;
            edge_cnt++;
        }
        // Neighbor
        if (edge_cnt < NUM_EDGES_NNZ) {
            adj_col_indices[edge_cnt] = (i + 1) % NUM_NODES;
            adj_values[edge_cnt] = (hls_dtype)0.5;
            edge_cnt++;
        }
        adj_row_ptr[i + 1] = edge_cnt;
    }
    
    // Fill remaining edges with 0 if any (shouldn't happen with this logic but safe-guard)
    while (edge_cnt < NUM_EDGES_NNZ) {
        adj_col_indices[edge_cnt] = 0;
        adj_values[edge_cnt] = 0;
        edge_cnt++;
    }

    // --- 3. Run Golden Reference (CPU) ---
    std::cout << "Running CPU Reference..." << std::endl;
    cpu_reference(h_in, w, adj_values, adj_col_indices, adj_row_ptr, h_out_gold);

    // --- 4. Run HLS Kernel ---
    std::cout << "Running HLS Kernel..." << std::endl;
    gnn(
        h_in.data(), 
        w.data(), 
        adj_values.data(), 
        adj_col_indices.data(), 
        adj_row_ptr.data(), 
        h_out_hls.data()
    );

    // --- 5. Verify Results ---
    std::cout << "Verifying Results..." << std::endl;
    int errors = 0;
    // Use a slightly larger tolerance for accumulation errors, though ap_fixed should match exactly
    float epsilon = 0.05; 

    for (int i = 0; i < NUM_NODES * OUT_FEATURES; ++i) {
        // FIX 2: Convert ap_fixed to float explicitly before subtraction/abs to avoid ambiguity
        float hls_val = (float)h_out_hls[i];
        float gold_val = (float)h_out_gold[i];
        
        if (std::abs(hls_val - gold_val) > epsilon) {
            if (errors < 10) {
                std::cout << "Mismatch at index " << i 
                          << ": HLS=" << hls_val 
                          << " Gold=" << gold_val << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "Test Passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed with " << errors << " errors." << std::endl;
        return 1;
    }
}