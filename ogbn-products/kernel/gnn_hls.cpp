#include "gnn_hls.h"

// =========================================================================
// COMPUTE ENGINE
// This function acts as one "Core". We instantiate 8 of these.
// =========================================================================
static void compute_engine(
    int pe_id,
    const hls_dtype* h_in,
    const int* adj_row_ptr,
    const int* adj_col_indices,
    const hls_dtype* adj_values,
    const hls_dtype w_local[IN_FEATURES][OUT_FEATURES],
    hls_dtype* h_out
) {
    // -------------------------------------------------------------
    // URAM OUTPUT BUFFER (Maximizing URAM Usage)
    // -------------------------------------------------------------
    // Each PE buffers a large chunk of results before writing to HBM.
    // This improves write efficiency significantly.
    static hls_dtype out_buf[URAM_BUFFER_SIZE][OUT_FEATURES];
    #pragma HLS BIND_STORAGE variable=out_buf type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=out_buf cyclic factor=16 dim=2

    // Loop over nodes assigned to this PE
    // We stride by NUM_PE (e.g., Node 0, 8, 16...)
    int buf_ptr = 0;
    int write_offset = 0; // Tracks where the current URAM buffer starts in global memory

    // Force loop flattening off to keep hierarchy clean for reporting
    // #pragma HLS LOOP_FLATTEN off

    node_loop_pe: for(int i = pe_id; i < NUM_NODES; i += NUM_PE) {
        #pragma HLS LOOP_TRIPCOUNT min=300000 max=300000
        
        // 1. Fetch Neighbor Range
        int start_idx = adj_row_ptr[i];
        int end_idx   = adj_row_ptr[i + 1];

        // 2. Local Accumulator (Register/LUTRAM)
        // Partitioned to allow SIMD math
        hls_dtype agg_features[IN_FEATURES];
        #pragma HLS ARRAY_PARTITION variable=agg_features complete

        init_agg: for(int f=0; f<IN_FEATURES; ++f) {
            #pragma HLS UNROLL
            agg_features[f] = 0;
        }

        // 3. SpMM: Process Neighbors
        neighbors_loop: for (int k = start_idx; k < end_idx; ++k) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=100 avg=25
            
            int neighbor_idx = adj_col_indices[k];
            hls_dtype edge_val = adj_values[k];

            // Manual Burst Read Hint
            const hls_dtype* neighbor_vec_ptr = h_in + (neighbor_idx * IN_FEATURES);
            
            // VECTORIZED MAC LOOP (Maximizing DSP Usage)
            // We unroll by VECTOR_SIZE (50). Since IN_FEATURES=100, this loops twice.
            // This consumes ~50 DSPs per PE * 8 PEs = 400 DSPs active.
            features_loop: for (int f = 0; f < IN_FEATURES; ++f) {
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=VECTOR_SIZE
                hls_dtype feat_val = neighbor_vec_ptr[f];
                agg_features[f] += edge_val * feat_val;
            }
        }

        // 4. GEMM: Dense Layer
        hls_dtype res_acc[OUT_FEATURES];
        #pragma HLS ARRAY_PARTITION variable=res_acc complete

        gemm_outer: for (int fo = 0; fo < OUT_FEATURES; ++fo) {
            #pragma HLS PIPELINE II=1
            hls_dtype sum = 0;
            // Fully unrolled dot product
            gemm_inner: for (int fi = 0; fi < IN_FEATURES; ++fi) {
                #pragma HLS UNROLL
                sum += agg_features[fi] * w_local[fi][fo];
            }
            res_acc[fo] = (sum > 0) ? sum : (hls_dtype)0;
        }

        // 5. Buffer Result in URAM
        store_uram: for(int fo = 0; fo < OUT_FEATURES; ++fo) {
            #pragma HLS UNROLL
            out_buf[buf_ptr][fo] = res_acc[fo];
        }
        buf_ptr++;

        // 6. Flush URAM to Global Memory if Full
        if (buf_ptr == URAM_BUFFER_SIZE) {
            // Burst write logic
            int global_start_node = i - (URAM_BUFFER_SIZE - 1) * NUM_PE;
            
            // We need to calculate exact address, tricky with stride.
            // Simplified: Write individually or bulk if reordered.
            // With stride, we can't do one giant burst for all, 
            // but we can burst per-node logic efficiently here.
            
            flush_loop: for(int b=0; b<URAM_BUFFER_SIZE; ++b) {
                int curr_node = global_start_node + (b * NUM_PE);
                // Burst write one node's output (OUT_FEATURES size)
                for(int fo = 0; fo < OUT_FEATURES; ++fo) {
                    #pragma HLS PIPELINE II=1
                    h_out[curr_node * OUT_FEATURES + fo] = out_buf[b][fo];
                }
            }
            buf_ptr = 0;
        }
    }

    // Flush remaining
    if (buf_ptr > 0) {
        int global_start_node = NUM_NODES - ((NUM_NODES % NUM_PE) > pe_id ? (NUM_NODES/NUM_PE) + 1 : (NUM_NODES/NUM_PE)) * NUM_PE + (NUM_PE * (NUM_NODES/NUM_PE - buf_ptr)); // Simplified approximate flushing
        // Note: Exact flush index logic for strided access is complex. 
        // For HLS, simple loop is safer:
        
        // Re-calculate the specific nodes we buffered
        int processed_count = (NUM_NODES - pe_id + NUM_PE - 1) / NUM_PE;
        int remaining_start_idx = (processed_count - buf_ptr) * NUM_PE + pe_id;
        
        flush_rem: for(int b=0; b<buf_ptr; ++b) {
            int curr_node = remaining_start_idx + (b * NUM_PE);
            for(int fo = 0; fo < OUT_FEATURES; ++fo) {
                #pragma HLS PIPELINE II=1
                h_out[curr_node * OUT_FEATURES + fo] = out_buf[b][fo];
            }
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
    // Interface Definitions
    // We utilize max_read_burst_length to optimize the HBM random access
    #pragma HLS INTERFACE m_axi port = h_in            offset = slave bundle = gmem0 depth = DEPTH_H_IN max_read_burst_length=256 num_read_outstanding=32
    #pragma HLS INTERFACE m_axi port = w               offset = slave bundle = gmem1 depth = DEPTH_W
    #pragma HLS INTERFACE m_axi port = adj_values      offset = slave bundle = gmem2 depth = DEPTH_ADJ_VAL num_read_outstanding=32
    #pragma HLS INTERFACE m_axi port = adj_col_indices offset = slave bundle = gmem3 depth = DEPTH_ADJ_COL num_read_outstanding=32
    #pragma HLS INTERFACE m_axi port = adj_row_ptr     offset = slave bundle = gmem3 depth = DEPTH_ADJ_ROW
    #pragma HLS INTERFACE m_axi port = h_out           offset = slave bundle = gmem0 depth = DEPTH_H_OUT

    #pragma HLS INTERFACE s_axilite port = h_in bundle = control
    #pragma HLS INTERFACE s_axilite port = w bundle = control
    #pragma HLS INTERFACE s_axilite port = adj_values bundle = control
    #pragma HLS INTERFACE s_axilite port = adj_col_indices bundle = control
    #pragma HLS INTERFACE s_axilite port = adj_row_ptr bundle = control
    #pragma HLS INTERFACE s_axilite port = h_out bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    // -------------------------------------------------------------
    // WEIGHT REPLICATION (Maximizing BRAM Usage)
    // -------------------------------------------------------------
    // We create a local copy of weights for EVERY PE.
    // 8 PEs * (100*47*2B) = ~75KB (Small, but uses BRAM ports efficiently)
    static hls_dtype w_local[NUM_PE][IN_FEATURES][OUT_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=w_local complete dim=1 // Separate BRAM for each PE
    #pragma HLS ARRAY_PARTITION variable=w_local complete dim=3 // Allow parallel read of output cols

    // Load Weights once
    load_w_i: for(int i=0; i<IN_FEATURES; ++i) {
        load_w_j: for(int j=0; j<OUT_FEATURES; ++j) {
            #pragma HLS PIPELINE II=1
            hls_dtype val = w[i * OUT_FEATURES + j];
            // Broadcast to all PEs
            for(int p=0; p<NUM_PE; ++p) {
                #pragma HLS UNROLL
                w_local[p][i][j] = val;
            }
        }
    }

    // -------------------------------------------------------------
    // LAUNCH PARALLEL ENGINES
    // -------------------------------------------------------------
    // The 'unroll' here creates 8 physical copies of the compute_engine logic.
    // They run concurrently.
    
    pe_loop: for(int p=0; p<NUM_PE; ++p) {
        #pragma HLS UNROLL
        compute_engine(p, h_in, adj_row_ptr, adj_col_indices, adj_values, w_local[p], h_out);
    }
}
}