#ifndef GCN_HLS_H
#define GCN_HLS_H

#include <hls_stream.h>
#include <ap_fixed.h> 

// --- Define Kernel Constants ---
// These MUST match the host code and testbench
const int NUM_NODES = 2708;
const int NUM_EDGES_NNZ = 10556; // Non-zero entries in adj matrix
const int IN_FEATURES = 1433;
const int OUT_FEATURES = 16;
typedef ap_fixed<16, 6, AP_RND, AP_SAT> hls_dtype;
// --- Top-Level Kernel Function DECLARATION ---
extern "C" {
void gnn(
    // Global Memory Interfaces (Pointers)
    const hls_dtype* h_in,           // Input features
    const hls_dtype* w,              // Weights
    const hls_dtype* adj_values,     // CSR adjacency values
    const int* adj_col_indices,      // CSR adjacency col indices
    const int* adj_row_ptr,          // CSR adjacency row ptr
    hls_dtype* h_out                 // Output features
);
} // extern "C"

#endif // GCN_HLS_H

