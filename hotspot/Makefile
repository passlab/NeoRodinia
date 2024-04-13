include ../common/environment.mk

EXECUTABLES := hotspot_serial_CPU_P0 hotspot_omp_CPU_P1 hotspot_omp_CPU_P2 hotspot_omp_CPU_P3 hotspot_omp_Offloading_P1 hotspot_omp_Offloading_P2 hotspot_omp_Offloading_P3 hotspot_omp_Offloading_P4 hotspot_cuda_Offloading_P1

all: $(EXECUTABLES)

utils.o: $(COMMON_SRC)
	$(CC) $(CFLAGS) $^ -fPIE -o utils.o $(LDFLAGS) -c 

hotspot_serial_CPU_P0: utils.o hotspot_serial_CPU_P0.cpp main.cpp hotspot_common.cpp
	$(CXX) $(CFLAGS) $^ -fPIE -o $@_$(CXX)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
hotspot_omp_CPU_P1: utils.o hotspot_omp_CPU_P1.cpp main.cpp hotspot_common.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CXX)_$(OPT_LEVEL)_exec $(LDFLAGS)

hotspot_omp_CPU_P2: utils.o hotspot_omp_CPU_P2.cpp main.cpp hotspot_common.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CXX)_$(OPT_LEVEL)_exec $(LDFLAGS)

hotspot_omp_CPU_P3: utils.o hotspot_omp_CPU_P3.cpp main.cpp hotspot_common.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CXX)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
hotspot_omp_Offloading_P1: utils.o hotspot_omp_Offloading_P1.cpp main.cpp hotspot_common.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CXX)_$(OPT_LEVEL)_exec $(LDFLAGS)

hotspot_omp_Offloading_P2: utils.o hotspot_omp_Offloading_P2.cpp main.cpp hotspot_common.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CXX)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
hotspot_omp_Offloading_P3: utils.o hotspot_omp_Offloading_P3.cpp main.cpp hotspot_common.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CXX)_$(OPT_LEVEL)_exec $(LDFLAGS)

hotspot_omp_Offloading_P4: utils.o hotspot_omp_Offloading_P4.cpp main.cpp hotspot_common.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CXX)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
hotspot_cuda_Offloading_P1: $(COMMON_SRC) hotspot_cuda_Offloading_P1.cu main.cpp hotspot_common.cpp
	$(CXX) $(CFLAGS) -c -o main.o main.cpp -fPIE
	$(NVCC) -Xcompiler "$(CFLAGS)" hotspot_cuda_Offloading_P1.cu -o hotspot_cuda.o -c $(MATH_LIB)
	$(CXX) $(CFLAGS) -c -o hotspot_common.o hotspot_common.cpp -fPIE
	$(NVCC) -Xcompiler "$(CFLAGS)" $(COMMON_DIR)/utils.c hotspot_cuda.o hotspot_common.o main.o -o $@_nvcc_$(OPT_LEVEL)_exec $(MATH_LIB)

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv output.out utils.o *.png
