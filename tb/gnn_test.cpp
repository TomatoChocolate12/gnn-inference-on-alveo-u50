/**
 * @file gcn_hls_tb.cpp
 * @brief HLS C-Simulation Testbench for the GCN kernel.
 * (Placed in tb/ directory)
 *
 * This file is used ONLY for C-Simulation and C/RTL Co-simulation
 * inside the Vitis HLS tool. It follows the 'vadd_tb.cpp' pattern.
 *
 * 1. Includes the kernel *header* file 'gnn_hls.h'.
 * 2. Loads data from data/cora/*.bin files (or generates synthetic if not found).
 * 3. Computes a "golden" CPU-based result for comparison.
 * 4. Calls the HLS kernel function (Design Under Test).
 * 5. Compares the HLS kernel's output to the golden result.
 * 6. Returns 0 on success (match) or 1 on failure (mismatch).
 */

#include <iostream>
#include <vector>
#include <cmath> // For std::abs
#include <cstdlib> // for rand()
#include <fstream> // For file I/O
#include <string>

#include "../kernel/gnn_hls.h"

// Helper function to load binary file
bool load_binary_file(const std::string& path, void* data, size_t size_bytes) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return false;
    }
    ifs.read(reinterpret_cast<char*>(data), size_bytes);
    return ifs.good();
}

// Load data from data/cora directory, fallback to synthetic if not found
bool load_cora_data(
    std::vector<hls_dtype>& h_in_vec,
    std::vector<hls_dtype>& w_vec,
    std::vector<hls_dtype>& adj_values_vec,
    std::vector<int>& adj_col_indices_vec,
    std::vector<int>& adj_row_ptr_vec
) {
    // Try multiple possible paths (HLS may run from different directories)
    std::vector<std::string> possible_paths = {
        "data/cora",           // From project root
        "../data/cora",        // From build/hls
        "../../data/cora"      // From build/hls/csim/build
    };
    
    for (const auto& data_dir : possible_paths) {
        std::cout << "Info: TB attempting to load data from " << data_dir << "..." << std::endl;
        
        bool all_loaded = true;
        all_loaded &= load_binary_file(data_dir + "/features.bin", h_in_vec.data(), 
                                        h_in_vec.size() * sizeof(hls_dtype));
        all_loaded &= load_binary_file(data_dir + "/weights.bin", w_vec.data(), 
                                        w_vec.size() * sizeof(hls_dtype));
        all_loaded &= load_binary_file(data_dir + "/adj_values.bin", adj_values_vec.data(), 
                                        adj_values_vec.size() * sizeof(hls_dtype));
        all_loaded &= load_binary_file(data_dir + "/adj_col_indices.bin", adj_col_indices_vec.data(), 
                                        adj_col_indices_vec.size() * sizeof(int));
        all_loaded &= load_binary_file(data_dir + "/adj_row_ptr.bin", adj_row_ptr_vec.data(), 
                                        adj_row_ptr_vec.size() * sizeof(int));
        
        if (all_loaded) {
            std::cout << "Info: TB successfully loaded data from " << data_dir << std::endl;
            return true;
        }
    }
    
    std::cout << "Info: TB failed to load data from any path, falling back to synthetic data..." << std::endl;
    return false;
}

// Generate synthetic data (fallback)
void generate_synthetic_data(
    std::vector<hls_dtype>& h_in_vec,
    std::vector<hls_dtype>& w_vec,
    std::vector<hls_dtype>& adj_values_vec,
    std::vector<int>& adj_col_indices_vec,
    std::vector<int>& adj_row_ptr_vec
) {
    std::cout << "Info: TB generating synthetic data..." << std::endl;
    srand(42); // Seed for reproducibility
    for(size_t i = 0; i < h_in_vec.size(); ++i) h_in_vec[i] = (rand() % 10) * 0.1f + 0.1f;
    for(size_t i = 0; i < w_vec.size(); ++i) w_vec[i] = (rand() % 5) * 0.01f;
    
    for(int i = 0; i < NUM_NODES; ++i) {
        adj_row_ptr_vec[i] = i;
        adj_col_indices_vec[i] = i; // Self-loop
        adj_values_vec[i] = 1.0;    // Normalized self-loop
    }
    adj_row_ptr_vec[NUM_NODES] = NUM_NODES;
    for(int i = NUM_NODES; i < NUM_EDGES_NNZ; ++i) {
         adj_col_indices_vec[i] = 0;
         adj_values_vec[i] = 0.0;
    }
    std::cout << "Info: TB synthetic data generated." << std::endl;
}

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

int main() {
    std::cout << "Info: Starting HLS C-Simulation Testbench..." << std::endl;

    // --- 1. Allocate and Populate Host-side Memory ---
    std::vector<hls_dtype> h_in_vec(NUM_NODES * IN_FEATURES);
    std::vector<hls_dtype> w_vec(IN_FEATURES * OUT_FEATURES);
    std::vector<hls_dtype> adj_values_vec(NUM_EDGES_NNZ);
    std::vector<int> adj_col_indices_vec(NUM_EDGES_NNZ);
    std::vector<int> adj_row_ptr_vec(NUM_NODES + 1);
    
    std::vector<hls_dtype> h_out_hls_vec(NUM_NODES * OUT_FEATURES);
    std::vector<hls_dtype> h_out_golden_vec(NUM_NODES * OUT_FEATURES);

    // Load data from data/cora, fallback to synthetic if not found
    if (!load_cora_data(h_in_vec, w_vec, adj_values_vec, adj_col_indices_vec, adj_row_ptr_vec)) {
        generate_synthetic_data(h_in_vec, w_vec, adj_values_vec, adj_col_indices_vec, adj_row_ptr_vec);
    }

    // Run Golden CPU Model ---
    compute_golden_result_on_cpu(
        h_in_vec, w_vec, adj_values_vec, adj_col_indices_vec, adj_row_ptr_vec,
        h_out_golden_vec
    );

    //Run HLS Kernel Function (The "DUT" - Design Under Test) ---
    std::cout << "Info: Calling HLS kernel function 'gnn'..." << std::endl;
    gnn(
        h_in_vec.data(),
        w_vec.data(),
        adj_values_vec.data(),
        adj_col_indices_vec.data(),
        adj_row_ptr_vec.data(),
        h_out_hls_vec.data()
    );
    std::cout << "Info: HLS kernel function finished." << std::endl;

    // Verify HLS Result vs. Golden Result ---
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

    //Return Result ---
    if (fail_count == 0) {
        std::cout << "TEST PASSED" << std::endl;
        return 0; // Success
    } else {
        std::cout << "TEST FAILED with " << fail_count << " mismatches." << std::endl;
        return 1; // Failure
    }
}


