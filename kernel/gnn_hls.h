#ifndef GCN_HLS_H
#define GCN_HLS_H

#include <hls_stream.h>
#include <ap_fixed.h> 

// =========================================================
// CONFIGURATION SWITCH
// =========================================================
#if defined(__XRT_EMULATION__) || defined(COSIM_TEST) || defined(TEST_MODE)
    #define ENABLE_TEST_MODE // Active for sw_emu, hw_emu, and hls co-sim
#endif

// =========================================================
// GRAPH DIMENSIONS (Using #define for Pragma Compatibility)
// =========================================================
#ifdef ENABLE_TEST_MODE
    // --- TINY DATASET (Fast Simulation) ---
    #define NUM_NODES 100
    #define NUM_EDGES_NNZ 200
    #define IN_FEATURES 1433
    #define OUT_FEATURES 16
    
    // Derived Depths for Pragmas
    #define DEPTH_H_IN (100 * 1433)
    #define DEPTH_W (1433 * 16)
    #define DEPTH_ADJ_VAL 200
    #define DEPTH_ADJ_COL 200
    #define DEPTH_ADJ_ROW (100 + 1)
    #define DEPTH_H_OUT (100 * 16)

#else
    // --- FULL CORA DATASET (Hardware Implementation) ---
    #define NUM_NODES 2708
    #define NUM_EDGES_NNZ 10556
    #define IN_FEATURES 1433
    #define OUT_FEATURES 16

    // Derived Depths for Pragmas
    // (Calculated: 2708 * 1433 = 3880564)
    #define DEPTH_H_IN (2708 * 1433)
    #define DEPTH_W (1433 * 16)
    #define DEPTH_ADJ_VAL 10556
    #define DEPTH_ADJ_COL 10556
    #define DEPTH_ADJ_ROW (2708 + 1)
    #define DEPTH_H_OUT (2708 * 16)
#endif

// Data Type Definition
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