#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "argument_parser.hpp"
#include "graph_loader.hpp"
#include "logger.hpp"
#include "xrt_utils.hpp"

int main(int argc, const char *const argv[]) {
  try {
    ArgumentParser parser;
    auto args = parser.parse(argc, argv);

    GraphTensors tensors = args.use_synthetic ? make_synthetic_graph()
                                              : load_graph_from_dir(args.data_dir);
    LogInfo("Graph tensors prepared");

    std::vector<hls_dtype> golden;
    if (args.enable_golden_check) {
      LogInfo("Running CPU reference");
      auto start = std::chrono::steady_clock::now();
      golden = run_cpu_reference(tensors);
      auto end = std::chrono::steady_clock::now();
      LogInfo("CPU reference time: " +
              std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) +
              " ms");
    }

    auto handles = init_xrt(args.xclbin_path, args.device_index);
    auto bos = create_input_bos(handles, tensors);

    LogInfo("Launching kernel");
    auto start = std::chrono::steady_clock::now();
    auto run = handles.kernel(bos[0], bos[1], bos[2], bos[3], bos[4], bos[5]);
    run.wait();
    auto end = std::chrono::steady_clock::now();
    LogInfo("Kernel execution time: " +
            std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) +
            " ms");

    auto result = fetch_output(bos.back());
    LogInfo("Fetched " + std::to_string(result.size()) + " outputs");

    if (args.enable_golden_check) {
      LogInfo("Comparing against CPU reference");
      compare_results(golden, result);
      LogInfo("Golden comparison passed");
    }

    LogInfo("Host run completed successfully");
    return 0;
  } catch (const std::exception &e) {
    LogError(std::string("Exception: ") + e.what());
    return 1;
  }
}
