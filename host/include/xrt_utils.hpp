#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include "graph_loader.hpp"
#include "logger.hpp"

struct XrtHandles {
  xrt::device device;
  xrt::uuid uuid;
  xrt::kernel kernel;
};

inline XrtHandles init_xrt(const std::string &xclbin_path, int device_index) {
  LogInfo("Opening device index " + std::to_string(device_index));
  xrt::device device(device_index);
  LogInfo("Loading xclbin from " + xclbin_path);
  auto uuid = device.load_xclbin(xclbin_path);
  xrt::kernel kernel(device, uuid, "gnn");
  return {std::move(device), std::move(uuid), std::move(kernel)};
}

inline std::vector<xrt::bo> create_input_bos(const XrtHandles &handles,
                                             const GraphTensors &tensors) {
  std::vector<xrt::bo> bos;
  bos.reserve(6);
  auto push = [&](const auto &vec, int arg_idx) {
    auto bo = xrt::bo(handles.device, vec.size() * sizeof(vec[0]),
                      handles.kernel.group_id(arg_idx));
    bo.write(vec.data());
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.emplace_back(std::move(bo));
  };
  push(tensors.features, 0);
  push(tensors.weights, 1);
  push(tensors.adj_values, 2);
  push(tensors.adj_col_indices, 3);
  push(tensors.adj_row_ptr, 4);

  std::size_t out_elems = static_cast<std::size_t>(NUM_NODES) * OUT_FEATURES;
  auto out_bo = xrt::bo(handles.device, out_elems * sizeof(hls_dtype),
                        handles.kernel.group_id(5));
  bos.emplace_back(std::move(out_bo));
  return bos;
}

inline std::vector<hls_dtype> fetch_output(xrt::bo &bo) {
  std::size_t elements = (bo.size() / sizeof(hls_dtype));
  std::vector<hls_dtype> out(elements);
  bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo.read(out.data());
  return out;
}
