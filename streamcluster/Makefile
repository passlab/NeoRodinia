include ../common/environment.mk

EXECUTABLES := sc_serial_CPU_P0 sc_omp_CPU_P1 sc_omp_CPU_P2 sc_omp_Offloading_P1 sc_omp_Offloading_P2 sc_omp_Offloading_P3 sc_omp_Offloading_P4 sc_cuda_Offloading_P1

all: $(EXECUTABLES)

utils.o: $(COMMON_DIR)/utils.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -c

omp_utils.o: $(COMMON_DIR)/omp_utils.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $^ -o $@ $(LDFLAGS) -c

sc_serial_CPU_P0: utils.o sc_serial_CPU_P0.cpp main.cpp
	$(CXX) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
sc_omp_CPU_P1: utils.o sc_omp_CPU_P1.cpp main.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

sc_omp_CPU_P2: utils.o sc_omp_CPU_P2.cpp main.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

sc_omp_Offloading_P1: utils.o sc_omp_Offloading_P1.cpp main.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

sc_omp_Offloading_P2: utils.o sc_omp_Offloading_P2.cpp main.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
sc_omp_Offloading_P3: utils.o sc_omp_Offloading_P3.cpp main.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

sc_omp_Offloading_P4: utils.o omp_utils.o sc_omp_Offloading_P4.cpp main.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
sc_cuda_Offloading_P1: $(COMMON_SRC) sc_cuda_Offloading_P1.cu main.cpp
	$(eval NV_ARCH_CODE=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | head -n 1))
	$(NVCC) -arch=sm_$(NV_ARCH_CODE) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv *.log *.o *.png *.svg
