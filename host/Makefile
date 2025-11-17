CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror
LDFLAGS ?=
INCLUDES := -Iinclude -I../kernel
SRC := $(wildcard src/*.cpp)
BUILD_DIR := build
TARGET := $(BUILD_DIR)/host.exe

ifndef XILINX_XRT
$(error XILINX_XRT environment variable is not set)
endif

LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lxrt_coreutil -pthread
CXXFLAGS += -I$(XILINX_XRT)/include

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
