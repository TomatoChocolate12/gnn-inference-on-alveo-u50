#ifndef GCN_HLS_H
#define GCN_HLS_H

#include <hls_stream.h>
#include <ap_fixed.h> 

// =========================================================
// CONFIGURATION
// =========================================================
#if defined(__XRT_EMULATION__) || defined(COSIM_TEST) || defined(TEST_MODE)
    #define ENABLE_TEST_MODE
#endif

// Architecture Constants
#define NUM_PE 8             // 8 Parallel Engines to saturate HBM and DSPs
#define VECTOR_SIZE 50       // Process 50 features per cycle (High DSP usage)
#define URAM_BUFFER_SIZE 4096 // Store 4096 nodes results before writing (High URAM usage)

// Dataset Constants
#ifdef ENABLE_TEST_MODE
    #define NUM_NODES 16
    #define NUM_EDGES_NNZ 32
    #define IN_FEATURES 32
    #define OUT_FEATURES 16
    
    // Depths
    #define DEPTH_H_IN (16*32)
    #define DEPTH_W (32*16)
    #define DEPTH_ADJ_VAL 32
    #define DEPTH_ADJ_COL 32
    #define DEPTH_ADJ_ROW (16+1)
    #define DEPTH_H_OUT (16*16)
#else
    // OGBN-PRODUCTS
    #define NUM_NODES 2449029
    #define NUM_EDGES_NNZ 61859140
    #define IN_FEATURES 100
    #define OUT_FEATURES 47 // Products has 47 classes

    // Depths
    #define DEPTH_H_IN (NUM_NODES * IN_FEATURES)
    #define DEPTH_W (IN_FEATURES * OUT_FEATURES)
    #define DEPTH_ADJ_VAL NUM_EDGES_NNZ
    #define DEPTH_ADJ_COL NUM_EDGES_NNZ
    #define DEPTH_ADJ_ROW (NUM_NODES + 1)
    #define DEPTH_H_OUT (NUM_NODES * OUT_FEATURES)
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

#endif // GCN_HLS_H