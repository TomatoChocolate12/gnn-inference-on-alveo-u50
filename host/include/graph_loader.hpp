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
  tensors.features.resize(NUM_NODES * IN_FEATURES);
  tensors.weights.resize(IN_FEATURES * OUT_FEATURES);
  tensors.adj_values.resize(NUM_EDGES_NNZ);
  tensors.adj_col_indices.resize(NUM_EDGES_NNZ);
  tensors.adj_row_ptr.resize(NUM_NODES + 1);

  auto copy_blob = [](auto &dst, const std::vector<uint8_t> &blob) {
    std::memcpy(dst.data(), blob.data(), blob.size());
  };

  auto features_blob = load_binary_blob(join_path(dir, "features.bin"));
  expect_size(features_blob.size() / sizeof(hls_dtype), tensors.features.size(), "features");
  copy_blob(tensors.features, features_blob);

  auto weights_blob = load_binary_blob(join_path(dir, "weights.bin"));
  expect_size(weights_blob.size() / sizeof(hls_dtype), tensors.weights.size(), "weights");
  copy_blob(tensors.weights, weights_blob);

  auto adj_val_blob = load_binary_blob(join_path(dir, "adj_values.bin"));
  expect_size(adj_val_blob.size() / sizeof(hls_dtype), tensors.adj_values.size(), "adj_values");
  copy_blob(tensors.adj_values, adj_val_blob);

  auto adj_col_blob = load_binary_blob(join_path(dir, "adj_col_indices.bin"));
  expect_size(adj_col_blob.size() / sizeof(int), tensors.adj_col_indices.size(), "adj_col_indices");
  copy_blob(tensors.adj_col_indices, adj_col_blob);

  auto adj_row_blob = load_binary_blob(join_path(dir, "adj_row_ptr.bin"));
  expect_size(adj_row_blob.size() / sizeof(int), tensors.adj_row_ptr.size(), "adj_row_ptr");
  copy_blob(tensors.adj_row_ptr, adj_row_blob);

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
      out[node * OUT_FEATURES + fout] = (sum > 0.0f) ? sum : 0.0f;
    }
  }
  return out;
}

inline void compare_results(const std::vector<hls_dtype> &golden,
                            const std::vector<hls_dtype> &actual,
                            float tol = 1e-3f) {
  if (golden.size() != actual.size()) {
    throw std::runtime_error("Size mismatch when comparing results");
  }
  int mismatches = 0;
  std::ostringstream report;
  for (std::size_t i = 0; i < golden.size(); ++i) {
    float diff = std::fabs(golden[i] - actual[i]);
    if (diff > tol) {
      if (mismatches < 10) {
        report << "idx=" << i << " gold=" << golden[i]
               << " got=" << actual[i] << " diff=" << diff << "\\n";
      }
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    report << "Total mismatches: " << mismatches;
    throw std::runtime_error(report.str());
  }
}
