include ../common/environment.mk

EXECUTABLES := matadd_serial_CPU_P0 matadd_omp_CPU_P1 matadd_omp_CPU_P2 matadd_omp_CPU_P3 matadd_omp_Offloading_P1 matadd_omp_Offloading_P2 matadd_omp_Offloading_P3 matadd_omp_Offloading_P4 matadd_cuda_Offloading_P1

all: $(EXECUTABLES)

matadd_serial_CPU_P0: $(COMMON_SRC) matadd_serial_CPU_P0.c main.c matadd_common.c
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matadd_omp_CPU_P1: $(COMMON_SRC) matadd_omp_CPU_P1.c main.c matadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matadd_omp_CPU_P2: $(COMMON_SRC) matadd_omp_CPU_P2.c main.c matadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matadd_omp_CPU_P3: $(COMMON_SRC) matadd_omp_CPU_P3.c main.c matadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matadd_omp_Offloading_P1: $(COMMON_SRC) matadd_omp_Offloading_P1.c main.c matadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matadd_omp_Offloading_P2: $(COMMON_SRC) matadd_omp_Offloading_P2.c main.c matadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matadd_omp_Offloading_P3: $(COMMON_SRC) matadd_omp_Offloading_P3.c main.c matadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matadd_omp_Offloading_P4: $(COMMON_SRC) matadd_omp_Offloading_P4.c main.c matadd_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matadd_cuda_Offloading_P1: $(COMMON_SRC) matadd_cuda_Offloading_P1.cu main.c matadd_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec
	
matadd_cuda_Offloading_P2: $(COMMON_SRC) matadd_cuda_Offloading_P2.cu main.c matadd_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec

matadd_cuda_Offloading_P3: $(COMMON_SRC) matadd_cuda_Offloading_P3.cu main.c matadd_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec
# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv *.png *.svg
