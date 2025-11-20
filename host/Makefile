CXX ?= g++
# REMOVED -Werror. Added -Wno-... flags to silence Xilinx header noise.
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label -Wno-comment -Wno-deprecated-copy $(EXTRA_FLAGS)
LDFLAGS ?=
INCLUDES := -Iinclude
SRC := $(wildcard src/*.cpp)
BUILD_DIR := build

# FIX: Renamed variable from TARGET to HOST_EXE to avoid conflict with top-level Makefile
HOST_EXE := $(BUILD_DIR)/host.exe

ifndef XILINX_XRT
$(error XILINX_XRT environment variable is not set)
endif

ifndef XILINX_HLS
$(error XILINX_HLS environment variable is not set. Please source Vitis settings)
endif

LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lxrt_coreutil -pthread
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include

.PHONY: all clean

all: $(HOST_EXE)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(HOST_EXE): $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)