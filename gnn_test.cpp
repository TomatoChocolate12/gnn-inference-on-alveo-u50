/**
 * @file gcn_hls_tb.cpp
 * @brief HLS C-Simulation Testbench for the GCN kernel.
 * (Placed in tb/ directory)
 *
 * This file is used ONLY for C-Simulation and C/RTL Co-simulation
 * inside the Vitis HLS tool. It follows the 'vadd_tb.cpp' pattern.
 *
 * 1. Includes the kernel *header* file 'gcn_hls.h'.
 * 2. Allocates memory for inputs and outputs.
 * 3. Populates inputs with placeholder data.
 * 4. Computes a "golden" CPU-based result for comparison.
 * 5. Calls the HLS kernel function (Design Under Test).
 * 6. Compares the HLS kernel's output to the golden result.
 * 7. Returns 0 on success (match) or 1 on failure (mismatch).
 */

#include <iostream>
#include <vector>
#include <cmath> // For std::abs
#include <cstdlib> // for rand()

// Include the HLS kernel *header* file, NOT the .cpp file
// This line is the fix for the duplicate symbol error.
#include "../kernel/gnn_hls.h" 

// --- Golden CPU computation (Reference) ---
// This logic MUST match the HLS kernel's logic.
void compute_golden_result_on_cpu(
    const std::vector<hls_dtype>& h_in,
    const std::vector<hls_dtype>& w,
    const std::vector<hls_dtype>& adj_values,
    const std::vector<int>& adj_col_indices,
    const std::vector<int>& adj_row_ptr,
    std::vector<hls_dtype>& h_out_golden
) {
    std::cout << "Info: TB computing golden result..." << std::endl;
    std::vector<hls_dtype> aggregated_features(NUM_NODES * IN_FEATURES, 0.0f);

    // 1. Aggregation (SpMM)
    for (int i = 0; i < NUM_NODES; ++i) {
        int start_idx = adj_row_ptr[i];
        int end_idx = adj_row_ptr[i + 1];
        for (int k = start_idx; k < end_idx; ++k) {
            int j = adj_col_indices[k];
            hls_dtype norm_val = adj_values[k];
            for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
                aggregated_features[i * IN_FEATURES + f_in] += norm_val * h_in[j * IN_FEATURES + f_in];
            }
        }
    }

    // 2. Transformation (GEMM) + ReLU
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int f_out = 0; f_out < OUT_FEATURES; ++f_out) {
            hls_dtype sum = 0.0;
            for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
                sum += aggregated_features[i * IN_FEATURES + f_in] * w[f_in * OUT_FEATURES + f_out];
            }
            // ReLU
            h_out_golden[i * OUT_FEATURES + f_out] = (sum > 0.0) ? sum : 0.0;
        }
    }
    std::cout << "Info: TB golden result computed." << std::endl;
}

// --- HLS Testbench Main Function ---
int main() {
    std::cout << "Info: Starting HLS C-Simulation Testbench..." << std::endl;

    // --- 1. Allocate and Populate Host-side Memory ---
    // Use std::vector for easy memory management, just like vadd_tb.cpp
    std::vector<hls_dtype> h_in_vec(NUM_NODES * IN_FEATURES);
    std::vector<hls_dtype> w_vec(IN_FEATURES * OUT_FEATURES);
    std::vector<hls_dtype> adj_values_vec(NUM_EDGES_NNZ);
    std::vector<int> adj_col_indices_vec(NUM_EDGES_NNZ);
    std::vector<int> adj_row_ptr_vec(NUM_NODES + 1);
    
    // Output from HLS kernel
    std::vector<hls_dtype> h_out_hls_vec(NUM_NODES * OUT_FEATURES);
    // Output from golden reference
    std::vector<hls_dtype> h_out_golden_vec(NUM_NODES * OUT_FEATURES);

    // --- 2. Populate with Dummy Data ---
    std::cout << "Info: TB loading dummy data..." << std::endl;
    // Use simple, predictable data for easy debugging
    srand(42); // Seed for reproducibility
    for(size_t i = 0; i < h_in_vec.size(); ++i) h_in_vec[i] = (rand() % 10) * 0.1f + 0.1f;
    for(size_t i = 0; i < w_vec.size(); ++i) w_vec[i] = (rand() % 5) * 0.01f;
    
    // Create a simple adjacency matrix (e.g., self-loops)
    for(int i = 0; i < NUM_NODES; ++i) {
        adj_row_ptr_vec[i] = i;
        adj_col_indices_vec[i] = i; // Self-loop
        adj_values_vec[i] = 1.0;    // Normalized self-loop
    }
    adj_row_ptr_vec[NUM_NODES] = NUM_NODES;
    // Fill remaining edges (if any) to avoid uninitialized data
    for(int i = NUM_NODES; i < NUM_EDGES_NNZ; ++i) {
         adj_col_indices_vec[i] = 0;
         adj_values_vec[i] = 0.0;
    }
    std::cout << "Info: TB dummy data loaded." << std::endl;

    // --- 3. Run Golden CPU Model ---
    compute_golden_result_on_cpu(
        h_in_vec, w_vec, adj_values_vec, adj_col_indices_vec, adj_row_ptr_vec,
        h_out_golden_vec
    );

    // --- 4. Run HLS Kernel Function (The "DUT" - Design Under Test) ---
    // We pass raw pointers (using .data()) to the HLS function,
    // matching the 'vadd' testbench pattern.
    std::cout << "Info: Calling HLS kernel function 'gcn_layer_hls'..." << std::endl;
    gcn_layer_hls(
        h_in_vec.data(),
        w_vec.data(),
        adj_values_vec.data(),
        adj_col_indices_vec.data(),
        adj_row_ptr_vec.data(),
        h_out_hls_vec.data()
    );
    std::cout << "Info: HLS kernel function finished." << std::endl;

    // --- 5. Verify HLS Result vs. Golden Result ---
    int fail_count = 0;
    float max_error = 1e-5; // Max acceptable floating point error
    for (size_t i = 0; i < h_out_hls_vec.size(); ++i) {
        if (std::abs(h_out_hls_vec[i] - h_out_golden_vec[i]) > max_error) {
            fail_count++;
            if(fail_count < 10) { // Print first few errors
                std::cout << "Error: Mismatch at index " << i << std::endl;
                std::cout << "  HLS Result:   " << h_out_hls_vec[i] << std::endl;
                std::cout << "  Golden Result: " << h_out_golden_vec[i] << std::endl;
            }
        }
    }

    // --- 6. Return Result ---
    if (fail_count == 0) {
        std::cout << "TEST PASSED" << std::endl;
        return 0; // Success
    } else {
        std::cout << "TEST FAILED with " << fail_count << " mismatches." << std::endl;
        return 1; // Failure
    }
}


