/**
 * @file gnn_hls.cpp
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

#include "gnn_hls.h"

//------------------------------------------------------------------
// Stage 1: load_inputs
//------------------------------------------------------------------
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
    for (int i = 0; i < NUM_NODES * IN_FEATURES; ++i) {
        #pragma HLS PIPELINE II=1
        h_in_stream << h_in[i];
    }
    for (int i = 0; i < IN_FEATURES * OUT_FEATURES; ++i) {
        #pragma HLS PIPELINE II=1
        w_stream << w[i];
    }
    for (int i = 0; i < NUM_EDGES_NNZ; ++i) {
        #pragma HLS PIPELINE II=1
        adj_val_stream << adj_values[i];
        adj_col_stream << adj_col_indices[i];
    }
    for (int i = 0; i < NUM_NODES + 1; ++i) {
        #pragma HLS PIPELINE II=1
        adj_row_stream << adj_row_ptr[i];
    }
}

//------------------------------------------------------------------
// Stage 2: compute_gcn
//------------------------------------------------------------------
static void compute_gcn(
    hls::stream<hls_dtype>& h_in_stream,
    hls::stream<hls_dtype>& w_stream,
    hls::stream<hls_dtype>& adj_val_stream,
    hls::stream<int>& adj_col_stream,
    hls::stream<int>& adj_row_stream,
    hls::stream<hls_dtype>& h_out_stream
) {
    static hls_dtype h_in_buf[NUM_NODES][IN_FEATURES];
    static hls_dtype w_buf[IN_FEATURES][OUT_FEATURES];
    static hls_dtype adj_val_buf[NUM_EDGES_NNZ];
    static int adj_col_buf[NUM_EDGES_NNZ];
    static int adj_row_buf[NUM_NODES + 1];

    #pragma HLS ARRAY_PARTITION variable=h_in_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=2

    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < IN_FEATURES; ++j) {
            #pragma HLS PIPELINE II=1
            h_in_buf[i][j] = h_in_stream.read();
        }
    }
    for (int i = 0; i < IN_FEATURES; ++i) {
        for (int j = 0; j < OUT_FEATURES; ++j) {
            #pragma HLS PIPELINE II=1
            w_buf[i][j] = w_stream.read();
        }
    }
    for (int i = 0; i < NUM_EDGES_NNZ; ++i) {
        #pragma HLS PIPELINE II=1
        adj_val_buf[i] = adj_val_stream.read();
        adj_col_buf[i] = adj_col_stream.read();
    }
    for (int i = 0; i < NUM_NODES + 1; ++i) {
        #pragma HLS PIPELINE II=1
        adj_row_buf[i] = adj_row_stream.read();
    }

    static hls_dtype aggregated_features[NUM_NODES][IN_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=aggregated_features complete dim=2

    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < IN_FEATURES; ++j) {
            #pragma HLS PIPELINE II=1
            aggregated_features[i][j] = 0.0;
        }
    }

    for (int i = 0; i < NUM_NODES; ++i) {
        int start_idx = adj_row_buf[i];
        int end_idx = adj_row_buf[i + 1];
        for (int k = start_idx; k < end_idx; ++k) {
            #pragma HLS PIPELINE
            int j = adj_col_buf[k];
            hls_dtype norm_val = adj_val_buf[k];
            for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
                #pragma HLS UNROLL
                hls_dtype feature = h_in_buf[j][f_in];
                aggregated_features[i][f_in] += norm_val * feature;
            }
        }
    }

    for (int i = 0; i < NUM_NODES; ++i) {
        for (int f_out = 0; f_out < OUT_FEATURES; ++f_out) {
            #pragma HLS PIPELINE
            hls_dtype sum = 0.0;
            for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
                sum += aggregated_features[i][f_in] * w_buf[f_in][f_out];
            }
            h_out_stream << (sum > 0.0 ? sum : 0.0);
        }
    }
}

//------------------------------------------------------------------
// Stage 3: store_result
//------------------------------------------------------------------
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

//------------------------------------------------------------------
// Top-Level Kernel Function (for HLS synthesis)
//------------------------------------------------------------------
extern "C" {
void gnn(
    const hls_dtype* h_in,
    const hls_dtype* w,
    const hls_dtype* adj_values,
    const int* adj_col_indices,
    const int* adj_row_ptr,
    hls_dtype* h_out
) {
    #pragma HLS INTERFACE m_axi port = h_in offset = slave bundle = gmem0 depth=3879364
    #pragma HLS INTERFACE m_axi port = w offset = slave bundle = gmem1 depth=22928
    #pragma HLS INTERFACE m_axi port = adj_values offset = slave bundle = gmem2 depth=10556
    #pragma HLS INTERFACE m_axi port = adj_col_indices offset = slave bundle = gmem3 depth=10556
    #pragma HLS INTERFACE m_axi port = adj_row_ptr offset = slave bundle = gmem3 depth=2709
    #pragma HLS INTERFACE m_axi port = h_out offset = slave bundle = gmem0 depth=43328

    #pragma HLS INTERFACE s_axilite port = h_in bundle = control
    #pragma HLS INTERFACE s_axilite port = w bundle = control
    #pragma HLS INTERFACE s_axilite port = adj_values bundle = control
    #pragma HLS INTERFACE s_axilite port = adj_col_indices bundle = control
    #pragma HLS INTERFACE s_axilite port = adj_row_ptr bundle = control
    #pragma HLS INTERFACE s_axilite port = h_out bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    static hls::stream<hls_dtype> h_in_stream("h_in_stream");
    static hls::stream<hls_dtype> w_stream("w_stream");
    static hls::stream<hls_dtype> adj_val_stream("adj_val_stream");
    static hls::stream<int> adj_col_stream("adj_col_stream");
    static hls::stream<int> adj_row_stream("adj_row_stream");
    static hls::stream<hls_dtype> h_out_stream("h_out_stream");

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
} // extern "C"