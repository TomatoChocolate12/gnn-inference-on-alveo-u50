/**
 * @file gcn_hls.cpp
 * @brief Conceptual HLS implementation of a single GCN layer.
 *
 * This code is refactored to use the efficient "load/compute/store"
 * dataflow pattern, inspired by the Xilinx vadd example.
 *
 * The GCN layer H_out = ReLU(A_norm * H_in * W) is decomposed into
 * three dataflow processes connected by HLS streams:
 *
 * 1. compute_aggregate (SpMM): Reads H_in and A_norm, computes H_agg.
 * 2. compute_transform (GEMM): Reads W, streams H_agg, computes H_out.
 * 3. store_result: Streams H_out and writes to global memory.
 *
 * This allows the stages to operate in parallel.
 */

#include <hls_stream.h>

// --- HLS-friendly constants ---
// These would be template parameters or #defines in a real design
const int NUM_NODES = 2708;     // e.g., Cora dataset
const int NUM_EDGES_NNZ = 10556; // Non-zero entries in adj_matrix
const int IN_FEATURES = 1433;    // Input features
const int OUT_FEATURES = 16;     // Output features (hidden dim)

// --- Define a basic data type (e.g., float or fixed-point) ---
typedef float hls_dtype;

// --- ReLU Activation Function ---
hls_dtype relu(hls_dtype a) {
    #pragma HLS INLINE
    return (a > 0.0) ? a : 0.0;
}

/**
 * @brief Task 1: Aggregation (SpMM: H_agg = A_norm * H_in)
 * Reads sparse adjacency matrix and input features from global memory.
 * Computes the aggregated features and streams them to the next stage.
 */
static void compute_aggregate(
    // Inputs from global memory
    hls_dtype h_in[NUM_NODES][IN_FEATURES],
    const hls_dtype adj_values[NUM_EDGES_NNZ],
    const int adj_col_indices[NUM_EDGES_NNZ],
    const int adj_row_ptr[NUM_NODES + 1],
    // Output stream
    hls::stream<hls_dtype>& h_agg_stream) {

    // This buffer holds the intermediate aggregated features.
    // In a real design, this might be streamed out row by row
    // instead of being fully buffered, depending on resources.
    hls_dtype aggregated_features[NUM_NODES][IN_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=aggregated_features complete dim=2

// Loop over each node 'i' in the graph
NODE_LOOP:
    for (int i = 0; i < NUM_NODES; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min = NUM_NODES max = NUM_NODES
        
        // --- Initialize aggregated feature vector for node 'i' to zeros ---
    INIT_AGG_FEAT:
        for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
            #pragma HLS PIPELINE
            aggregated_features[i][f_in] = 0.0;
        }

        // --- Get the start and end index for neighbors of node 'i' ---
        int start_idx = adj_row_ptr[i];
        int end_idx = adj_row_ptr[i + 1];

    // Loop over all non-zero entries (neighbors) for node 'i'
    AGGREGATION_LOOP:
        for (int k = start_idx; k < end_idx; ++k) {
            #pragma HLS LOOP_TRIPCOUNT min = 0 max = NUM_EDGES_NNZ
            #pragma HLS PIPELINE
            
            // Get neighbor node index 'j' and the sparse matrix value A_ij
            int j = adj_col_indices[k];
            hls_dtype norm_val = adj_values[k];

            // Perform: agg_feat[i] += norm_val * h_in[j]
        AGG_FEAT_LOOP:
            for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
                #pragma HLS UNROLL
                aggregated_features[i][f_in] += norm_val * h_in[j][f_in];
            }
        }
    }

    // --- Stream out the aggregated features ---
// Stream aggregated features one-by-one
STREAM_AGG_NODE:
    for (int i = 0; i < NUM_NODES; ++i) {
    STREAM_AGG_FEAT:
        for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
            #pragma HLS PIPELINE
            #pragma HLS LOOP_TRIPCOUNT min = NUM_NODES*IN_FEATURES max = NUM_NODES*IN_FEATURES
            h_agg_stream << aggregated_features[i][f_in];
        }
    }
}

/**
 * @brief Task 2: Transformation (GEMM: H_out = ReLU(H_agg * W))
 * Buffers weights from global memory.
 * Reads aggregated features from input stream.
 * Computes the transformation (GEMM) and ReLU.
 * Streams the final output features.
 */
static void compute_transform(
    // Input stream
    hls::stream<hls_dtype>& h_agg_stream,
    // Input from global memory
    hls_dtype w[IN_FEATURES][OUT_FEATURES],
    // Output stream
    hls::stream<hls_dtype>& h_out_stream) {
    
    // --- Buffer weights locally in BRAM for high-speed access ---
    hls_dtype w_local[IN_FEATURES][OUT_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=w_local complete dim=2

LOAD_W_FEAT_IN:
    for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
    LOAD_W_FEAT_OUT:
        for (int f_out = 0; f_out < OUT_FEATURES; ++f_out) {
            #pragma HLS PIPELINE
            #pragma HLS LOOP_TRIPCOUNT min = IN_FEATURES*OUT_FEATURES max = IN_FEATURES*OUT_FEATURES
            w_local[f_in][f_out] = w[f_in][f_out];
        }
    }

    // --- Process aggregated features from the stream ---
// Loop over each node 'i'
NODE_TRANSFORM_LOOP:
    for (int i = 0; i < NUM_NODES; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min = NUM_NODES max = NUM_NODES

        // Buffer for one row of aggregated features
        hls_dtype h_agg_row[IN_FEATURES];
        #pragma HLS ARRAY_PARTITION variable=h_agg_row complete dim=1
    
    // Read one row (node) from the input stream
    READ_H_AGG_ROW:
        for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
            #pragma HLS PIPELINE
            #pragma HLS LOOP_TRIPCOUNT min = IN_FEATURES max = IN_FEATURES
            h_agg_row[f_in] = h_agg_stream.read();
        }

    // Loop over each output feature 'f_out'
    OUT_FEAT_LOOP:
        for (int f_out = 0; f_out < OUT_FEATURES; ++f_out) {
            #pragma HLS PIPELINE
            
            // Core Matrix-Vector Multiply (GEMM)
            hls_dtype sum = 0.0;
        IN_FEAT_LOOP:
            for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
                sum += h_agg_row[f_in] * w_local[f_in][f_out];
            }
            
            // Apply ReLU and write to output stream
            h_out_stream << relu(sum);
        }
    }
}

/**
 * @brief Task 3: Store Result
 * Reads the final computed features from the stream
 * and writes them back to global memory.
 */
static void store_result(
    // Input stream
    hls::stream<hls_dtype>& h_out_stream,
    // Output to global memory
    hls_dtype h_out[NUM_NODES][OUT_FEATURES]) {

// Write results node by node, feature by feature
STORE_NODE_LOOP:
    for (int i = 0; i < NUM_NODES; ++i) {
    STORE_FEAT_LOOP:
        for (int f_out = 0; f_out < OUT_FEATURES; ++f_out) {
            #pragma HLS PIPELINE
            #pragma HLS LOOP_TRIPCOUNT min = NUM_NODES*OUT_FEATURES max = NUM_NODES*OUT_FEATURES
            h_out[i][f_out] = h_out_stream.read();
        }
    }
}

/**
 * @brief Top-level GCN Layer Kernel
 *
 * This function orchestrates the dataflow between the 
 * aggregate, transform, and store stages.
 *
 * @param h_in           Input node features (H_in)
 * @param w              Layer weights (W)
 * @param adj_values     CSR non-zero values
 * @param adj_col_indices  CSR column indices
 * @param adj_row_ptr    CSR row pointers
 * @param h_out          Output node features (H_out)
 */
extern "C" {
void gcn_layer_hls(
    // Inputs
    hls_dtype h_in[NUM_NODES][IN_FEATURES],
    hls_dtype w[IN_FEATURES][OUT_FEATURES],
    const hls_dtype adj_values[NUM_EDGES_NNZ],
    const int adj_col_indices[NUM_EDGES_NNZ],
    const int adj_row_ptr[NUM_NODES + 1],
    // Output
    hls_dtype h_out[NUM_NODES][OUT_FEATURES]) {
    
    // --- HLS Interface Pragmas ---
    // Define memory interfaces for global memory access
    #pragma HLS INTERFACE m_axi port = h_in offset = slave bundle = gmem0 depth = NUM_NODES*IN_FEATURES
    #pragma HLS INTERFACE m_axi port = w offset = slave bundle = gmem1 depth = IN_FEATURES*OUT_FEATURES
    #pragma HLS INTERFACE m_axi port = adj_values offset = slave bundle = gmem2 depth = NUM_EDGES_NNZ
    #pragma HLS INTERFACE m_axi port = adj_col_indices offset = slave bundle = gmem2 depth = NUM_EDGES_NNZ
    #pragma HLS INTERFACE m_axi port = adj_row_ptr offset = slave bundle = gmem2 depth = NUM_NODES+1
    #pragma HLS INTERFACE m_axi port = h_out offset = slave bundle = gmem0 depth = NUM_NODES*OUT_FEATURES

    // Define control interface
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    // --- Internal Streams ---
    // These streams connect the dataflow tasks
    static hls::stream<hls_dtype> h_agg_stream("h_agg_stream");
    static hls::stream<hls_dtype> h_out_stream("h_out_stream");

    // Define stream depths to prevent deadlocks
    #pragma HLS STREAM variable=h_agg_stream depth=IN_FEATURES
    #pragma HLS STREAM variable=h_out_stream depth=OUT_FEATURES

    // --- Dataflow Region ---
    // This pragma instructs HLS to create parallel, pipelined hardware
    // for each of the three functions, similar to the vadd example.
    #pragma HLS dataflow

    // Task 1: H_agg = A_norm * H_in
    compute_aggregate(
        h_in,
        adj_values,
        adj_col_indices,
        adj_row_ptr,
        h_agg_stream);

    // Task 2: H_out = ReLU(H_agg * W)
    compute_transform(
        h_agg_stream,
        w,
        h_out_stream);

    // Task 3: Write H_out to global memory
    store_result(
        h_out_stream,
        h_out);
}
}


