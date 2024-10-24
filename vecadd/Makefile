include ../common/environment.mk

EXECUTABLES := vecadd_serial_CPU_P0 vecadd_omp_CPU_P1 vecadd_omp_CPU_P2 vecadd_omp_CPU_P3 vecadd_omp_Offloading_P1 vecadd_omp_Offloading_P2 vecadd_omp_Offloading_P3 vecadd_omp_Offloading_P4 vecadd_cuda_Offloading_P1

all: $(EXECUTABLES)

vecadd_serial_CPU_P0: $(COMMON_SRC) vecadd_serial_CPU_P0.c main.c vecadd_common.c
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
vecadd_omp_CPU_P1: $(COMMON_SRC) vecadd_omp_CPU_P1.c main.c vecadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

vecadd_omp_CPU_P2: $(COMMON_SRC) vecadd_omp_CPU_P2.c main.c vecadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

vecadd_omp_CPU_P3: $(COMMON_SRC) vecadd_omp_CPU_P3.c main.c vecadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
vecadd_omp_Offloading_P1: $(COMMON_SRC) vecadd_omp_Offloading_P1.c main.c vecadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

vecadd_omp_Offloading_P2: $(COMMON_SRC) vecadd_omp_Offloading_P2.c main.c vecadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

vecadd_omp_Offloading_P3: $(COMMON_SRC) vecadd_omp_Offloading_P3.c main.c vecadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

vecadd_omp_Offloading_P4: $(COMMON_SRC) vecadd_omp_Offloading_P4.c main.c vecadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
vecadd_cuda_Offloading_P1: $(COMMON_SRC) vecadd_cuda_Offloading_P1.cu main.c vecadd_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec
	
vecadd_cuda_Offloading_P2: $(COMMON_SRC) vecadd_cuda_Offloading_P2.cu main.c vecadd_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec

vecadd_cuda_Offloading_P3: $(COMMON_SRC) vecadd_cuda_Offloading_P3.cu main.c vecadd_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec
# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv *.png *.svg
