#include "gnn_hls.h"

// Stage 1: load_inputs
static void load_inputs(
    const hls_dtype* h_in,
    const hls_dtype* w,
    const hls_dtype* adj_values,
    const int* adj_col_indices,
    const int* adj_row_ptr,
    hls::stream<hls_dtype>& h_in_stream,
    hls::stream<hls_dtype>& w_stream,
    hls::stream<hls_dtype>& adj_val_stream,
    hls::stream<int>& adj_col_stream,
    hls::stream<int>& adj_row_stream
) {
    // Reads are sequential, no complex logic needed here
    load_h_in: for (int i = 0; i < NUM_NODES * IN_FEATURES; ++i) {
        #pragma HLS PIPELINE II=1
        h_in_stream << h_in[i];
    }
    load_w: for (int i = 0; i < IN_FEATURES * OUT_FEATURES; ++i) {
        #pragma HLS PIPELINE II=1
        w_stream << w[i];
    }
    load_adj: for (int i = 0; i < NUM_EDGES_NNZ; ++i) {
        #pragma HLS PIPELINE II=1
        adj_val_stream << adj_values[i];
        adj_col_stream << adj_col_indices[i];
    }
    load_row: for (int i = 0; i < NUM_NODES + 1; ++i) {
        #pragma HLS PIPELINE II=1
        adj_row_stream << adj_row_ptr[i];
    }
}

// Stage 2: compute_gcn
static void compute_gcn(
    hls::stream<hls_dtype>& h_in_stream,
    hls::stream<hls_dtype>& w_stream,
    hls::stream<hls_dtype>& adj_val_stream,
    hls::stream<int>& adj_col_stream,
    hls::stream<int>& adj_row_stream,
    hls::stream<hls_dtype>& h_out_stream
) {
    // Partition factor for parallel processing
    const int UNROLL_FACTOR = 512;
    
    // --- Buffers ---
    // Use URAM for large buffers. 
    // We partition dimension 2 (features) to allow parallel access.
    static hls_dtype h_in_buf[NUM_NODES][IN_FEATURES];
    #pragma HLS BIND_STORAGE variable=h_in_buf type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=h_in_buf cyclic factor=UNROLL_FACTOR dim=2

    // Weights buffer: Completely partition dim 2 (Out Features) 
    // This allows us to compute all 16 output features in one clock cycle per input.
    static hls_dtype w_buf[IN_FEATURES][OUT_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=2

    static hls_dtype adj_val_buf[NUM_EDGES_NNZ];
    static int adj_col_buf[NUM_EDGES_NNZ];
    static int adj_row_buf[NUM_NODES + 1];

    // --- READ PHASE ---
    read_h_in: for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < IN_FEATURES; ++j) {
            #pragma HLS PIPELINE II=1
            h_in_buf[i][j] = h_in_stream.read();
        }
    }
    
    read_w: for (int i = 0; i < IN_FEATURES; ++i) {
        for (int j = 0; j < OUT_FEATURES; ++j) {
            #pragma HLS PIPELINE II=1
            w_buf[i][j] = w_stream.read();
        }
    }
    
    read_adj: for (int i = 0; i < NUM_EDGES_NNZ; ++i) {
        #pragma HLS PIPELINE II=1
        adj_val_buf[i] = adj_val_stream.read();
        adj_col_buf[i] = adj_col_stream.read();
    }
    read_row: for (int i = 0; i < NUM_NODES + 1; ++i) {
        #pragma HLS PIPELINE II=1
        adj_row_buf[i] = adj_row_stream.read();
    }

    // --- SPMM (Aggregation) ---
    static hls_dtype aggregated_features[NUM_NODES][IN_FEATURES];
    #pragma HLS BIND_STORAGE variable=aggregated_features type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=aggregated_features cyclic factor=UNROLL_FACTOR dim=2
    
    // Initialize to 0
    init_agg: for (int i = 0; i < NUM_NODES; ++i) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < IN_FEATURES; ++j) {
            aggregated_features[i][j] = 0;
        }
    }

    spmm_nodes: for (int i = 0; i < NUM_NODES; ++i) {
        int start_idx = adj_row_buf[i];
        int end_idx = adj_row_buf[i + 1];

        spmm_neighbors: for(int k = start_idx; k < end_idx; ++k){
            #pragma HLS PIPELINE II=1
            int neighbor_idx = adj_col_buf[k];
            hls_dtype norm_val = adj_val_buf[k];

            // Parallel update of features
            spmm_features: for(int f = 0; f < IN_FEATURES; ++f) {
                #pragma HLS UNROLL factor=UNROLL_FACTOR
                hls_dtype feat = h_in_buf[neighbor_idx][f];
                aggregated_features[i][f] += norm_val * feat;
            }
        }
    }

    // --- GEMM (Dense Layer) + ReLU ---
    // Optimized Loop Order: Input Stationary
    gemm_nodes: for (int i = 0; i < NUM_NODES; ++i) {
        #pragma HLS PIPELINE II=1

        // Temporary accumulator registers for one node's output
        hls_dtype output_acc[OUT_FEATURES];
        #pragma HLS ARRAY_PARTITION variable=output_acc complete

        // Reset Accumulators
        init_acc: for(int fo = 0; fo < OUT_FEATURES; ++fo) output_acc[fo] = 0;

        // Loop over Input Features
        gemm_mult: for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
            hls_dtype in_val = aggregated_features[i][f_in];
            
            // Broadcast input feature to all 16 output weights in parallel
            gemm_broadcast: for (int f_out = 0; f_out < OUT_FEATURES; ++f_out) {
                #pragma HLS UNROLL
                output_acc[f_out] += in_val * w_buf[f_in][f_out];
            }
        }

        // ReLU and Stream Out
        gemm_write: for (int f_out = 0; f_out < OUT_FEATURES; ++f_out) {
            hls_dtype val = output_acc[f_out];
            h_out_stream << (val > 0.0 ? val : (hls_dtype)0.0);
        }
    }
}

// Stage 3: store_result
static void store_result(
    hls::stream<hls_dtype>& h_out_stream,
    hls_dtype* h_out
) {
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < OUT_FEATURES; ++j) {
            #pragma HLS PIPELINE II=1
            h_out[i * OUT_FEATURES + j] = h_out_stream.read();
        }
    }
}

extern "C" {
void gnn(
    const hls_dtype* h_in,
    const hls_dtype* w,
    const hls_dtype* adj_values,
    const int* adj_col_indices,
    const int* adj_row_ptr,
    hls_dtype* h_out
) {
    #pragma HLS INTERFACE m_axi port = h_in offset = slave bundle = gmem0  depth=DEPTH_H_IN
    #pragma HLS INTERFACE m_axi port = w offset = slave bundle = gmem1 depth=DEPTH_W
    #pragma HLS INTERFACE m_axi port = adj_values offset = slave bundle = gmem2 depth=DEPTH_ADJ_VAL
    #pragma HLS INTERFACE m_axi port = adj_col_indices offset = slave bundle = gmem3 depth=DEPTH_ADJ_COL
    #pragma HLS INTERFACE m_axi port = adj_row_ptr offset = slave bundle = gmem3 depth=DEPTH_ADJ_ROW
    #pragma HLS INTERFACE m_axi port = h_out offset = slave bundle = gmem0 depth=DEPTH_H_OUT

    #pragma HLS INTERFACE s_axilite port = h_in bundle = control
    #pragma HLS INTERFACE s_axilite port = w bundle = control
    #pragma HLS INTERFACE s_axilite port = adj_values bundle = control
    #pragma HLS INTERFACE s_axilite port = adj_col_indices bundle = control
    #pragma HLS INTERFACE s_axilite port = adj_row_ptr bundle = control
    #pragma HLS INTERFACE s_axilite port = h_out bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    // FIX: Explicit Stream Depths to prevent deadlocks
    static hls::stream<hls_dtype> h_in_stream("h_in_stream");
    static hls::stream<hls_dtype> w_stream("w_stream");
    static hls::stream<hls_dtype> adj_val_stream("adj_val_stream");
    static hls::stream<int> adj_col_stream("adj_col_stream");
    static hls::stream<int> adj_row_stream("adj_row_stream");
    static hls::stream<hls_dtype> h_out_stream("h_out_stream");

    #pragma HLS STREAM variable=h_in_stream depth=64
    #pragma HLS STREAM variable=w_stream depth=64
    #pragma HLS STREAM variable=adj_val_stream depth=64
    #pragma HLS STREAM variable=adj_col_stream depth=64
    #pragma HLS STREAM variable=adj_row_stream depth=64
    #pragma HLS STREAM variable=h_out_stream depth=64

    #pragma HLS dataflow

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
}