#pragma once

#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct Arguments {
  std::string xclbin_path;
  std::string data_dir;
  int device_index = 0;
  bool use_synthetic = false;
  bool enable_golden_check = false;
};

class ArgumentParser {
 public:
  ArgumentParser() = default;

  Arguments parse(int argc, const char *const argv[]) {
    Arguments args;
    args.xclbin_path = "kernel/build/gcn_layer_hls.hw.xclbin";
    args.data_dir = "data/cora";

    for (int i = 1; i < argc; ++i) {
      std::string token = argv[i];
      if (token == "--help" || token == "-h") {
        print_help(argv[0]);
        std::exit(0);
      } else if (token == "--xclbin_file" && i + 1 < argc) {
        args.xclbin_path = argv[++i];
      } else if (token == "--data_dir" && i + 1 < argc) {
        args.data_dir = argv[++i];
      } else if (token == "--device_id" && i + 1 < argc) {
        args.device_index = std::stoi(argv[++i]);
      } else if (token == "--use_synthetic") {
        args.use_synthetic = true;
      } else if (token == "--enable_golden_check") {
        args.enable_golden_check = true;
      } else {
        throw std::runtime_error("Unknown argument: " + token);
      }
    }
    return args;
  }

  static void print_help(const char *exe_name) {
    std::cout << "Usage: " << exe_name << " [options]\n"
              << "  --xclbin_file <path>       Path to xclbin file (default kernel/build/...)\n"
              << "  --data_dir <dir>           Directory with binary graph tensors\n"
              << "  --device_id <id>           Xilinx device index (default 0)\n"
              << "  --use_synthetic           Generate deterministic synthetic data\n"
              << "  --enable_golden_check     Run CPU golden model to verify output\n"
              << "  -h, --help                Show this message\n";
  }
};
