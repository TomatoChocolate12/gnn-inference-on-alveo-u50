#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

// This header defines hls_dtype. 
// Make sure your gnn_hls.h now has: typedef ap_fixed<16, 6, AP_RND, AP_SAT> hls_dtype;
#include "gnn_hls.h"

struct GraphTensors {
  std::vector<hls_dtype> features;      // NUM_NODES * IN_FEATURES
  std::vector<hls_dtype> weights;       // IN_FEATURES * OUT_FEATURES
  std::vector<hls_dtype> adj_values;    // NUM_EDGES_NNZ
  std::vector<int> adj_col_indices;     // NUM_EDGES_NNZ
  std::vector<int> adj_row_ptr;         // NUM_NODES + 1
};

inline std::string join_path(const std::string &dir, const std::string &file) {
  if (dir.empty()) {
    return file;
  }
  if (dir.back() == '/') {
    return dir + file;
  }
  return dir + "/" + file;
}

inline void expect_size(std::size_t actual, std::size_t expected, const std::string &name) {
  if (actual != expected) {
    throw std::runtime_error("Unexpected element count for " + name + ": got " +
                             std::to_string(actual) + ", expected " + std::to_string(expected));
  }
}

inline std::vector<uint8_t> load_binary_blob(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(static_cast<std::size_t>(size));
  if (!ifs.read(reinterpret_cast<char *>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

inline GraphTensors load_graph_from_dir(const std::string &dir) {
  GraphTensors tensors;
  
  // Resize vectors to the target hardware size
  tensors.features.resize(NUM_NODES * IN_FEATURES);
  tensors.weights.resize(IN_FEATURES * OUT_FEATURES);
  tensors.adj_values.resize(NUM_EDGES_NNZ);
  tensors.adj_col_indices.resize(NUM_EDGES_NNZ);
  tensors.adj_row_ptr.resize(NUM_NODES + 1);

  // --- Helper: Load Float Binary and Convert to Fixed Point ---
  auto load_and_convert_float = [](auto &dst_vec, const std::string &filepath) {
    auto blob = load_binary_blob(filepath);
    
    // The binary file contains 32-bit floats
    size_t num_floats = blob.size() / sizeof(float);
    
    // Verify that the number of floats in the file matches the destination vector size
    if (num_floats != dst_vec.size()) {
        throw std::runtime_error("Size mismatch in " + filepath + 
            ": Expected " + std::to_string(dst_vec.size()) + 
            " elements, got " + std::to_string(num_floats));
    }

    // Interpret bytes as floats
    const float* float_data = reinterpret_cast<const float*>(blob.data());
    
    // Convert float -> hls_dtype (ap_fixed)
    for (size_t i = 0; i < num_floats; ++i) {
        dst_vec[i] = (hls_dtype)float_data[i];
    }
  };

  // --- Helper: Load Integer Binary (Direct Copy) ---
  auto load_int = [](auto &dst_vec, const std::string &filepath) {
    auto blob = load_binary_blob(filepath);
    size_t num_ints = blob.size() / sizeof(int);
    
    if (num_ints != dst_vec.size()) {
        throw std::runtime_error("Size mismatch in " + filepath);
    }
    // Integers don't need conversion, memcpy is safe
    std::memcpy(dst_vec.data(), blob.data(), blob.size());
  };

  // 1. Load Float Data (Features, Weights, Values) -> Convert to ap_fixed
  load_and_convert_float(tensors.features, join_path(dir, "features.bin"));
  load_and_convert_float(tensors.weights, join_path(dir, "weights.bin"));
  load_and_convert_float(tensors.adj_values, join_path(dir, "adj_values.bin"));

  // 2. Load Integer Data (Indices) -> Direct Copy
  load_int(tensors.adj_col_indices, join_path(dir, "adj_col_indices.bin"));
  load_int(tensors.adj_row_ptr, join_path(dir, "adj_row_ptr.bin"));

  return tensors;
}

inline GraphTensors make_synthetic_graph() {
  GraphTensors tensors;
  tensors.features.resize(NUM_NODES * IN_FEATURES);
  tensors.weights.resize(IN_FEATURES * OUT_FEATURES);
  tensors.adj_values.resize(NUM_EDGES_NNZ);
  tensors.adj_col_indices.resize(NUM_EDGES_NNZ);
  tensors.adj_row_ptr.resize(NUM_NODES + 1);

  for (int node = 0; node < NUM_NODES; ++node) {
    for (int fin = 0; fin < IN_FEATURES; ++fin) {
      // Perform calculation in float, then cast to fixed point
      tensors.features[node * IN_FEATURES + fin] = static_cast<hls_dtype>((node % 13) * 0.1f + fin * 0.0001f);
    }
    tensors.adj_row_ptr[node] = node;
    tensors.adj_col_indices[node] = node;
    tensors.adj_values[node] = 1.0f;
  }
  tensors.adj_row_ptr[NUM_NODES] = NUM_NODES;
  for (int idx = NUM_NODES; idx < NUM_EDGES_NNZ; ++idx) {
    tensors.adj_col_indices[idx] = 0;
    tensors.adj_values[idx] = 0.0f;
  }
  for (int fin = 0; fin < IN_FEATURES; ++fin) {
    for (int fout = 0; fout < OUT_FEATURES; ++fout) {
      tensors.weights[fin * OUT_FEATURES + fout] = static_cast<hls_dtype>((fin + fout) % 7 * 0.01f);
    }
  }
  return tensors;
}

inline std::vector<hls_dtype> run_cpu_reference(const GraphTensors &tensors) {
  std::vector<hls_dtype> aggregated(NUM_NODES * IN_FEATURES, 0.0f);
  
  // Bit-accurate simulation: Since hls_dtype is ap_fixed, 
  // this CPU code will simulate the exact quantization effects of the hardware.
  for (int node = 0; node < NUM_NODES; ++node) {
    int start = tensors.adj_row_ptr[node];
    int end = tensors.adj_row_ptr[node + 1];
    for (int idx = start; idx < end; ++idx) {
      int col = tensors.adj_col_indices[idx];
      hls_dtype w = tensors.adj_values[idx];
      for (int fin = 0; fin < IN_FEATURES; ++fin) {
        aggregated[node * IN_FEATURES + fin] += w * tensors.features[col * IN_FEATURES + fin];
      }
    }
  }
  std::vector<hls_dtype> out(NUM_NODES * OUT_FEATURES, 0.0f);
  for (int node = 0; node < NUM_NODES; ++node) {
    for (int fout = 0; fout < OUT_FEATURES; ++fout) {
      hls_dtype sum = 0.0f;
      for (int fin = 0; fin < IN_FEATURES; ++fin) {
        sum += aggregated[node * IN_FEATURES + fin] *
               tensors.weights[fin * OUT_FEATURES + fout];
      }
      out[node * OUT_FEATURES + fout] = (sum > (hls_dtype)0) ? sum : (hls_dtype);
    }
  }
  return out;
}

inline void compare_results(const std::vector<hls_dtype> &golden,
                            const std::vector<hls_dtype> &actual,
                            float tol = 1e-2f) { // Increased tolerance slightly for quantization
  if (golden.size() != actual.size()) {
    throw std::runtime_error("Size mismatch when comparing results");
  }
  int mismatches = 0;
  std::ostringstream report;
  for (std::size_t i = 0; i < golden.size(); ++i) {
    // Convert back to float to calculate the absolute difference correctly
    float gold_val = golden[i].to_float();
    float act_val = actual[i].to_float();
    float diff = std::fabs(gold_val - act_val);
    
    if (diff > tol) {
      if (mismatches < 10) {
        report << "idx=" << i << " gold=" << gold_val
               << " got=" << act_val << " diff=" << diff << "\n";
      }
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    report << "Total mismatches: " << mismatches;
    throw std::runtime_error(report.str());
  }
}