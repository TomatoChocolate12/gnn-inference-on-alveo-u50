/**
 * @file gcn_host.cpp
 * @brief Host application for the GCN HLS kernel.
 *
 * This C++ application runs on the host CPU (x86 or ARM).
 * It uses the OpenCL C++ bindings (managed by XRT) to:
 * 1. Load the graph data (from a "database" or files).
 * 2. Find and program the FPGA with the .xclbin file.
 * 3. Allocate buffers on the FPGA's global memory.
 * 4. Transfer graph data to the FPGA.
 * 5. Set the arguments for the 'gcn_layer_hls' kernel.
 * 6. Run the kernel and measure its performance.
 * 7. Read back the results and verify them.
 */

// XRT/OpenCL headers
// Make sure to include the XRT and OpenCL include paths during compilation
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FAILURE_ERRORS 1
#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <fstream
#include <chrono>

// --- Use the same constants as the HLS kernel ---
// This ensures host and kernel memory allocations match.
const int NUM_NODES = 2708;
const int NUM_EDGES_NNZ = 10556;
const int IN_FEATURES = 1433;
const int OUT_FEATURES = 16;
typedef float hls_dtype;

// --- Helper function to load the .xclbin file ---
std::vector<unsigned char> read_xclbin(const std::string& xclbin_path) {
    std::ifstream bin_file(xclbin_path, std::ifstream::binary);
    if (!bin_file.is_open()) {
        throw std::runtime_error("Could not open xclbin file: " + xclbin_path);
    }
    bin_file.seekg(0, bin_file.end);
    auto nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    std::vector<unsigned char> buf(nb);
    bin_file.read(reinterpret_cast<char*>(buf.data()), nb);
    return buf;
}

// --- Placeholder for your data loading logic ---
void load_graph_data_from_database(
    std::vector<hls_dtype>& h_in,
    std::vector<hls_dtype>& w,
    std::vector<hls_dtype>& adj_values,
    std::vector<int>& adj_col_indices,
    std::vector<int>& adj_row_ptr
) {
    std::cout << "Info: Loading graph data from 'database' (using placeholders)..." << std::endl;
    // ---
    // In a real application, you would load your data from files
    // (e.g., .csv, .npy, or a graph database) here and populate
    // the host-side vectors.
    // ---
    // Using dummy data for demonstration:
    h_in.assign(NUM_NODES * IN_FEATURES, 1.0f); // Dummy features
    w.assign(IN_FEATURES * OUT_FEATURES, 0.5f); // Dummy weights
    adj_values.assign(NUM_EDGES_NNZ, 1.0f);      // Dummy adj values
    adj_col_indices.assign(NUM_EDGES_NNZ, 0);    // Dummy adj cols
    adj_row_ptr.assign(NUM_NODES + 1, 0);        // Dummy adj rows
    
    // Create a simple identity matrix in CSR for adj_row_ptr and adj_col_indices
    // This is just to have valid data.
    for(int i = 0; i < NUM_NODES; ++i) {
        adj_row_ptr[i] = i;
        adj_col_indices[i] = i; // Self-loop
        adj_values[i] = 1.0;
    }
    adj_row_ptr[NUM_NODES] = NUM_NODES;
    // Fill remaining dummy values
    for(int i = NUM_NODES; i < NUM_EDGES_NNZ; ++i) {
         adj_col_indices[i] = 0;
         adj_values[i] = 0.0;
    }
    std::cout << "Info: Dummy graph data loaded." << std::endl;
}

// --- Placeholder for CPU-based verification ---
void compute_golden_result_on_cpu(
    const std::vector<hls_dtype>& h_in,
    const std::vector<hls_dtype>& w,
    const std::vector<hls_dtype>& adj_values,
    const std::vector<int>& adj_col_indices,
    const std::vector<int>& adj_row_ptr,
    std::vector<hls_dtype>& h_out_golden
) {
    std::cout << "Info: Computing golden result on CPU for verification..." << std::endl;
    // This logic should exactly match your HLS kernel's logic.
    std::vector<hls_dtype> aggregated_features(NUM_NODES * IN_FEATURES, 0.0f);

    // 1. Aggregation (SpMM)
    for (int i = 0; i < NUM_NODES; ++i) {
        int start_idx = adj_row_ptr[i];
        int end_idx = adj_row_ptr[i + 1];
        for (int k = start_idx; k < end_idx; ++k) {
            int j = adj_col_indices[k];
            hls_dtype norm_val = adj_values[k];
            for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
                aggregated_features[i * IN_FEATURES + f_in] += norm_val * h_in[j * IN_FEATURES + f_in];
            }
        }
    }

    // 2. Transformation (GEMM) + ReLU
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int f_out = 0; f_out < OUT_FEATURES; ++f_out) {
            hls_dtype sum = 0.0;
            for (int f_in = 0; f_in < IN_FEATURES; ++f_in) {
                sum += aggregated_features[i * IN_FEATURES + f_in] * w[f_in * OUT_FEATURES + f_out];
            }
            // ReLU
            h_out_golden[i * OUT_FEATURES + f_out] = (sum > 0.0) ? sum : 0.0;
        }
    }
     std::cout << "Info: CPU golden result computed." << std::endl;
}

// --- Main Host Function ---
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_xclbin>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string xclbin_path = argv[1];

    // --- 1. Load Data from "Database" ---
    // Host-side memory containers
    std::vector<hls_dtype> h_in(NUM_NODES * IN_FEATURES);
    std::vector<hls_dtype> w(IN_FEATURES * OUT_FEATURES);
    std::vector<hls_dtype> adj_values(NUM_EDGES_NNZ);
    std::vector<int> adj_col_indices(NUM_EDGES_NNZ);
    std::vector<int> adj_row_ptr(NUM_NODES + 1);
    std::vector<hls_dtype> h_out(NUM_NODES * OUT_FEATURES);
    
    // Populate these vectors with your graph data
    load_graph_data_from_database(h_in, w, adj_values, adj_col_indices, adj_row_ptr);

    // --- 2. OpenCL/XRT Setup ---
    std::cout << "Info: Setting up OpenCL/XRT environment." << std::endl;
    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0]; // Assuming first platform is Xilinx
    
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    cl::Device device = devices[0]; // Assuming one device
    
    cl::Context context(device, NULL, NULL, NULL, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    
    // Load the .xclbin
    std::vector<unsigned char> xclbin = read_xclbin(xclbin_path);
    cl::Program::Binaries bins{{xclbin.data(), xclbin.size()}};
    cl::Program program(context, {device}, bins, NULL, &err);
    cl::Kernel kernel(program, "gcn_layer_hls", &err);
    
    std::cout << "Info: FPGA programmed with " << xclbin_path << std::endl;

    // --- 3. Allocate Buffers on FPGA Device Memory ---
    std::cout << "Info: Allocating buffers on FPGA." << std::endl;
    cl::Buffer d_h_in(context, CL_MEM_READ_ONLY, sizeof(hls_dtype) * h_in.size(), NULL, &err);
    cl::Buffer d_w(context, CL_MEM_READ_ONLY, sizeof(hls_dtype) * w.size(), NULL, &err);
    cl::Buffer d_adj_values(context, CL_MEM_READ_ONLY, sizeof(hls_dtype) * adj_values.size(), NULL, &err);
    cl::Buffer d_adj_col_indices(context, CL_MEM_READ_ONLY, sizeof(int) * adj_col_indices.size(), NULL, &err);
    cl::Buffer d_adj_row_ptr(context, CL_MEM_READ_ONLY, sizeof(int) * adj_row_ptr.size(), NULL, &err);
    cl::Buffer d_h_out(context, CL_MEM_WRITE_ONLY, sizeof(hls_dtype) * h_out.size(), NULL, &err);

    // --- 4. Transfer Data from Host RAM to FPGA Buffers ---
    std::cout << "Info: Transferring data from Host to FPGA..." << std::endl;
    auto start_transfer_host_to_dev = std::chrono::high_resolution_clock::now();
    q.enqueueWriteBuffer(d_h_in, CL_TRUE, 0, sizeof(hls_dtype) * h_in.size(), h_in.data());
    q.enqueueWriteBuffer(d_w, CL_TRUE, 0, sizeof(hls_dtype) * w.size(), w.data());
    q.enqueueWriteBuffer(d_adj_values, CL_TRUE, 0, sizeof(hls_dtype) * adj_values.size(), adj_values.data());
    q.enqueueWriteBuffer(d_adj_col_indices, CL_TRUE, 0, sizeof(int) * adj_col_indices.size(), adj_col_indices.data());
    q.enqueueWriteBuffer(d_adj_row_ptr, CL_TRUE, 0, sizeof(int) * adj_row_ptr.size(), adj_row_ptr.data());
    q.finish();
    auto end_transfer_host_to_dev = std::chrono::high_resolution_clock::now();
    
    // --- 5. Set Kernel Arguments ---
    kernel.setArg(0, d_h_in);
    kernel.setArg(1, d_w);
    kernel.setArg(2, d_adj_values);
    kernel.setArg(3, d_adj_col_indices);
    kernel.setArg(4, d_adj_row_ptr);
    kernel.setArg(5, d_h_out);

    // --- 6. Execute Kernel and Measure Performance ---
    std::cout << "Info: Executing GCN kernel on FPGA..." << std::endl;
    auto start_kernel = std::chrono::high_resolution_clock::now();
    
    // Launch the kernel
    q.enqueueTask(kernel);
    
    // Wait for kernel to finish
    q.finish(); 
    
    auto end_kernel = std::chrono::high_resolution_clock::now();
    std::cout << "Info: Kernel execution finished." << std::endl;

    // --- 7. Transfer Results from FPGA back to Host ---
    std::cout << "Info: Transferring results from FPGA to Host..." << std::endl;
    auto start_transfer_dev_to_host = std::chrono::high_resolution_clock::now();
    q.enqueueReadBuffer(d_h_out, CL_TRUE, 0, sizeof(hls_dtype) * h_out.size(), h_out.data());
    q.finish();
    auto end_transfer_dev_to_host = std::chrono::high_resolution_clock::now();

    // --- 8. Report Performance ---
    std::chrono::duration<double, std::milli> transfer_in_ms = end_transfer_host_to_dev - start_transfer_host_to_dev;
    std::chrono::duration<double, std::milli> kernel_ms = end_kernel - start_kernel;
    std::chrono::duration<double, std::milli> transfer_out_ms = end_transfer_dev_to_host - start_transfer_dev_to_host;
    
    std::cout << "\n--- Performance Results ---" << std::endl;
    std::cout << "Host -> FPGA Transfer Time: " << transfer_in_ms.count() << " ms" << std::endl;
    std::cout << "FPGA Kernel Execution Time: " << kernel_ms.count() << " ms" << std::endl;
    std::cout << "FPGA -> Host Transfer Time: " << transfer_out_ms.count() << " ms" << std::endl;
    std::cout << "---------------------------\n" << std::endl;

    // --- 9. Verify Results ---
    std::vector<hls_dtype> h_out_golden(NUM_NODES * OUT_FEATURES);
    compute_golden_result_on_cpu(h_in, w, adj_values, adj_col_indices, adj_row_ptr, h_out_golden);
    
    bool match = true;
    for (int i = 0; i < h_out.size(); ++i) {
        // Use an approximate comparison for floating-point numbers
        if (std::abs(h_out[i] - h_out_golden[i]) > 1e-5) {
            std::cout << "Error: Mismatch found at index " << i << std::endl;
            std::cout << "  FPGA Result: " << h_out[i] << std::endl;
            std::cout << "  CPU Result:  " << h_out_golden[i] << std::endl;
            match = false;
            break;
        }
    }
    
    if (match) {
        std::cout << "Success: FPGA result matches CPU golden result!" << std::endl;
    } else {
        std::cout << "Failure: FPGA result does not match CPU golden result." << std::endl;
    }

    return match ? EXIT_SUCCESS : EXIT_FAILURE;
}

