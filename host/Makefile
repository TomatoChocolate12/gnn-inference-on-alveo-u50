CXX ?= g++
# REMOVED -Werror. Added -Wno-... flags to silence Xilinx header noise.
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label -Wno-comment -Wno-deprecated-copy
LDFLAGS ?=
INCLUDES := -Iinclude
SRC := $(wildcard src/*.cpp)
BUILD_DIR := build
TARGET := $(BUILD_DIR)/host.exe

ifndef XILINX_XRT
$(error XILINX_XRT environment variable is not set)
endif

# You MUST check if XILINX_HLS is set, just like XRT
ifndef XILINX_HLS
$(error XILINX_HLS environment variable is not set. Please source Vitis settings)
endif

LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lxrt_coreutil -pthread
# You MUST keep this line. Sourcing settings isn't enough; the compiler needs the -I flag.
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)