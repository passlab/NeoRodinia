# Compiler and flags
# Two options:clang and gcc-11
CC := clang
CXX := clang++

NVCC := nvcc

COMMON_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
COMMON_SRC := $(COMMON_DIR)/utils.c

# Default optimization level is -O1
OPT_LEVEL := O1
CFLAGS := -Wall -$(OPT_LEVEL) -I$(COMMON_DIR)
LDFLAGS :=
OMP_FLAG := -fopenmp
OFFLOADING_FLAG := -fopenmp-targets=nvptx64

ifeq ($(findstring gcc,$(CC)),gcc)
	OFFLOADING_FLAG :=
	CXX := g++
endif

ifeq ($(findstring g++,$(CXX)),g++)
	OFFLOADING_FLAG :=
	CC := gcc
endif

ifeq ($(findstring clang,$(CC)),clang)
	OFFLOADING_FLAG := -fopenmp-targets=nvptx64
	CXX := clang++
endif

ifeq ($(findstring clang++,$(CXX)),clang++)
	OFFLOADING_FLAG := -fopenmp-targets=nvptx64
	CC := clang
endif

