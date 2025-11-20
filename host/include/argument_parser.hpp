#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdlib>

struct Arguments {
    std::string xclbin_path;
    int device_index = 0;
    bool use_synthetic = false;       // Triggered by -s or --synthetic
    bool enable_golden_check = true;  // Default true, disable with --no-golden
    std::string data_dir = "";
};

class ArgumentParser {
public:
    Arguments parse(int argc, const char* const argv[]) {
        Arguments args;
        bool xclbin_found = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "--xclbin_file") {
                if (i + 1 < argc) {
                    args.xclbin_path = argv[++i];
                    xclbin_found = true;
                } else {
                    throw std::runtime_error("Missing value for --xclbin_file");
                }
            } else if (arg == "--device_id") {
                if (i + 1 < argc) {
                    args.device_index = std::stoi(argv[++i]);
                } else {
                    throw std::runtime_error("Missing value for --device_id");
                }
            } else if (arg == "-s" || arg == "--synthetic") {
                args.use_synthetic = true;
            } else if (arg == "--golden") {
                args.enable_golden_check = true;
            } else if (arg == "--no-golden") {
                args.enable_golden_check = false;
            } else {
                // If argument doesn't start with '-', assume it's the data directory
                if (arg[0] != '-') {
                    args.data_dir = arg;
                } else {
                    throw std::runtime_error("Unknown argument: " + arg);
                }
            }
        }

        if (!xclbin_found) {
            throw std::runtime_error("Option --xclbin_file is required.");
        }

        // If not using synthetic data and no data_dir provided, we might want to warn or fail.
        // But for now, main.cpp usually falls back or checks this logic.
        
        return args;
    }
};