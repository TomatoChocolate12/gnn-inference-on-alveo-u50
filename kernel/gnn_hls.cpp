/**
 * @file gcn_hls.cpp
 * @brief HLS kernel for a GCN layer, based on the 'vadd' dataflow example.
 * (Placed in kernel/ directory)
 *
 * Implements H_out = ReLU( (A_norm * H_in) * W )
 *
 * This kernel uses a simple load-compute-store dataflow pattern.
 * - load_inputs: Reads all graph data from global memory into streams.
 * - compute_gcn: Reads streams, performs SpMM and GEMM, writes to output stream.
 * - store_result: Reads output stream and writes to global memory.
 */

// Include the new header file
#include "gnn_hls.h" 

// --- Constants are now defined in gcn_hls.h ---


//------------------------------------------------------------------
// Stage 1: load_inputs
//
// Reads all necessary data from global memory and streams it.
//------------------------------------------------------------------
static void load_inputs(
    // Inputs from Global Memory
    const hls_dtype* h_in,
    const hls_dtype* w,
    const hls_dtype* adj_values,
    const int* adj_col_indices,
    const int* adj_row_ptr,
    // Output Streams
    hls::stream<hls_dtype>& h_in_stream,
    hls::stream<hls_dtype>& w_stream,
    hls::stream<hls_dtype>& adj_val_stream,
    hls::stream<int>& adj_col_stream,
    hls::stream<int>& adj_row_stream
) {
    // Stream H_in features
    mem_rd_h_in:
    for (int i = 0; i < NUM_NODES * IN_FEATURES; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=3879364 max=3879364
        #pragma HLS PIPELINE II=1
        h_in_stream << h_in[i];
    }

    // Stream W weights
    mem_rd_w:
    for (int i = 0; i < IN_FEATURES * OUT_FEATURES; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=22928 max=22928
        #pragma HLS PIPELINE II=1
        w_stream << w[i];
    }

    // Stream Adjacency Matrix (CSR)
    mem_rd_adj_val:
    for (int i = 0; i < NUM_EDGES_NNZ; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=10556 max=10556
        #pragma HLS PIPELINE II=1
        adj_val_stream << adj_values[i];
        adj_col_stream << adj_col_indices[i];
    }
    mem_rd_adj_row:
    for (int i = 0; i < NUM_NODES + 1; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=2709 max=2709
        #pragma HLS PIPELINE II=1
        adj_row_stream << adj_row_ptr[i];
    }
}


//------------------------------------------------------------------
// Stage 2: compute_gcn
//
// Performs the SpMM (A*H) and GEMM ((A*H)*W) operations.
// Reads from streams, writes to output stream.
//------------------------------------------------------------------
static void compute_gcn(
    // Input Streams
    hls::stream<hls_dtype>& h_in_stream,
    hls::stream<hls_dtype>& w_stream,
    hls::stream<hls_dtype>& adj_val_stream,
    hls::stream<int>& adj_col_stream,
    hls::stream<int>& adj_row_stream,
    // Output Stream
    hls::stream<hls_dtype>& h_out_stream
) {
    // --- Buffer inputs on-chip ---
    // These buffers are essential for HLS to build an efficient datapath.
    // **NOTE: 'static' is critical here to infer BRAM/URAM instead of stack.**
    static hls_dtype h_in_buf[NUM_NODES][IN_FEATURES];
    static hls_dtype w_buf[IN_FEATURES][OUT_FEATURES];
    static hls_dtype adj_val_buf[NUM_EDGES_NNZ];
    static int adj_col_buf[NUM_EDGES_NNZ];
    static int adj_row_buf[NUM_NODES + 1];

    // Partition arrays for parallel access
    #pragma HLS ARRAY_PARTITION variable=h_in_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=2

    // --- Read from Streams into Buffers ---
    read_h_in_buf:
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < IN_FEATURES; ++j) {
            #pragma HLS LOOP_TRIPCOUNT min=3879364 max=3879364
            #pragma HLS PIPELINE II=1
            h_in_buf[i][j] = h_in_stream.read();
        }
    }

    read_w_buf:
    for (int i = 0; i < IN_FEATURES; ++i) {
        for (int j = 0; j < OUT_FEATURES; ++j) {
            #pragma HLS LOOP_TRIPCOUNT min=22928 max=22928
            #pragma HLS PIPELINE II=1
            w_buf[i][j] = w_stream.read();
        }
    }

    read_adj_buf:
    for (int i = 0; i < NUM_EDGES_NNZ; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=10556 max=10556
        #pragma HLS PIPELINE II=1
        adj_val_buf[i] = adj_val_stream.read();
        adj_col_buf[i] = adj_col_stream.read();
    }
    for (int i = 0; i < NUM_NODES + 1; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=2709 max=2709
        #pragma HLS PIPELINE II=1
        adj_row_buf[i] = adj_row_stream.read();
    }

    // --- Perform GCN Computation ---

    // 1. SpMM (Aggregation: H_agg = A * H)
    // We create an intermediate buffer for the aggregated features.
    static hls_dtype aggregated_features[NUM_NODES][IN_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=aggregated_features complete dim=2

    // Initialize aggregate buffer to zeros
    init_agg_buf:
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < IN_FEATURES; ++j) {
            #pragma HLS PIPELINE II=1
            aggregated_features[i][j] = 0.0;
        }
    }

    // SpMM loop
    spmm_outer_loop:
    for (int i = 0; i < NUM_NODES; ++i) { // For each node
        int start_idx = adj_row_buf[i];
        int end_idx = adj_row_buf[i + 1];
        
        spmm_inner_loop:
        for (int k = start_idx; k < end_idx; ++k) { // For each neighbor
            #pragma HLS PIPELINE
            int j = adj_col_buf[k]; // Neighbor index
            hls_dtype norm_val = adj_val_buf[k];

            spmm_feature_loop:
            for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
                #pragma HLS UNROLL
                // Read neighbor's feature and multiply by norm_val
                hls_dtype feature = h_in_buf[j][f_in];
                aggregated_features[i][f_in] += norm_val * feature;
            }
        }
    }

    // 2. GEMM (Transformation: H_out = H_agg * W) + ReLU
    gemm_outer_loop:
    for (int i = 0; i < NUM_NODES; ++i) { // For each node
        gemm_inner_loop:
        for (int f_out = 0; f_out < OUT_FEATURES; ++f_out) { // For each output feature
            #pragma HLS PIPELINE
            hls_dtype sum = 0.0;
            gemm_feature_loop:
            for (int f_in = 0; f_in < IN_FEATURES; ++f_in) { // Dot product
                sum += aggregated_features[i][f_in] * w_buf[f_in][f_out];
            }
            
            // Apply ReLU and write to output stream
            if (sum > 0.0) {
                h_out_stream << sum;
            } else {
                h_out_stream << 0.0;
            }
        }
    }
}


//------------------------------------------------------------------
// Stage 3: store_result
//
// Writes the final results from the output stream to global memory.
//------------------------------------------------------------------
static void store_result(
    hls::stream<hls_dtype>& h_out_stream,
    hls_dtype* h_out
) {
    mem_wr_h_out:
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < OUT_FEATURES; ++j) {
            #pragma HLS LOOP_TRIPCOUNT min=43328 max=43328
            #pragma HLS PIPELINE II=1
            h_out[i * OUT_FEATURES + j] = h_out_stream.read();
        }
    }
}


//------------------------------------------------------------------
// Top-Level Kernel Function
//------------------------------------------------------------------
extern "C" {
void gcn_layer_hls(
    // Global Memory Interfaces (Pointers)
    const hls_dtype* h_in,           // Input features
    const hls_dtype* w,              // Weights
    const hls_dtype* adj_values,     // CSR adjacency values
    const int* adj_col_indices,      // CSR adjacency col indices
    const int* adj_row_ptr,          // CSR adjacency row ptr
    hls_dtype* h_out                 // Output features
) {
    // --- AXI Interface Pragmas ---
    // Connects pointers to m_axi ports for DMA
    #pragma HLS INTERFACE m_axi port = h_in offset = slave bundle = gmem0 depth=3879364
    #pragma HLS INTERFACE m_axi port = w offset = slave bundle = gmem1 depth=22928
    #pragma HLS INTERFACE m_axi port = adj_values offset = slave bundle = gmem2 depth=10556
    #pragma HLS INTERFACE m_axi port = adj_col_indices offset = slave bundle = gmem3 depth=10556
    #pragma HLS INTERFACE m_axi port = adj_row_ptr offset = slave bundle = gmem3 depth=2709
    #pragma HLS INTERFACE m_axi port = h_out offset = slave bundle = gmem0 depth=43328
    
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    // --- Internal Streams ---
    // These streams act as FIFOs connecting the dataflow stages.
    static hls::stream<hls_dtype> h_in_stream("h_in_stream");
    static hls::stream<hls_dtype> w_stream("w_stream");
    static hls::stream<hls_dtype> adj_val_stream("adj_val_stream");
    static hls::stream<int> adj_col_stream("adj_col_stream");
    static hls::stream<int> adj_row_stream("adj_row_stream");
    static hls::stream<hls_dtype> h_out_stream("h_out_stream");

    // --- Dataflow Pragma ---
    // This tells HLS to build the 3 stages to run in parallel.
    #pragma HLS dataflow

    // These three functions will run in a pipelined, parallel fashion.
    load_inputs(
        h_in, w, adj_values, adj_col_indices, adj_row_ptr,
        h_in_stream, w_stream, adj_val_stream, adj_col_stream, adj_row_stream
    );
    
    compute_gcn(
        h_in_stream, w_stream, adj_val_stream, adj_col_stream, adj_row_stream,
        h_out_stream
    );
    
    store_result(
        h_out_stream, h_out
    );
}
} // extern "C"


