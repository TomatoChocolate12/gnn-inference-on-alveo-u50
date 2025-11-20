#ifndef GCN_HLS_H
#define GCN_HLS_H

#include <hls_stream.h>
#include <ap_fixed.h> 

// If TEST_MODE is defined (passed via Makefile), use Tiny dimensions
#ifdef TEST_MODE
    const int NUM_NODES = 16;
    const int NUM_EDGES_NNZ = 32; 
    const int IN_FEATURES = 32;
    const int OUT_FEATURES = 16;
#else
    // Standard Cora Dataset
    const int NUM_NODES = 2708;
    const int NUM_EDGES_NNZ = 10556;
    const int IN_FEATURES = 1433;
    const int OUT_FEATURES = 16;
#endif

typedef ap_fixed<16, 6, AP_RND, AP_SAT> hls_dtype;

extern "C" {
void gnn(
    const hls_dtype* h_in,
    const hls_dtype* w,
    const hls_dtype* adj_values,
    const int* adj_col_indices,
    const int* adj_row_ptr,
    hls_dtype* h_out
);
}

#endif