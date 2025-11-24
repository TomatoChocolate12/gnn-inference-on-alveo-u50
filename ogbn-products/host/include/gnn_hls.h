#ifndef GCN_HLS_H
#define GCN_HLS_H

#include <ap_fixed.h>

// --- Define Kernel Constants ---
// These MUST match the kernel/gnn_hls.h for ogbn-products dataset
#define NUM_NODES 2449029
#define NUM_EDGES_NNZ 61859140
#define IN_FEATURES 100
#define OUT_FEATURES 47  // num_classes for ogbn-products

// Data Type Definition (must match kernel)
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

