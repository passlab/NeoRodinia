include ../common/environment.mk

EXECUTABLES := sum_serial_CPU_P0 sum_omp_CPU_P1 sum_omp_CPU_P2 sum_omp_CPU_P3 sum_omp_CPU_P4 sum_omp_CPU_P5 sum_omp_CPU_P6 sum_omp_Offloading_P1 sum_omp_Offloading_P2 sum_omp_Offloading_P3 sum_omp_Offloading_P4 sum_cuda_Offloading_P1 sum_cuda_Offloading_P2 sum_cuda_Offloading_P3 sum_cuda_Offloading_P4 sum_cuda_Offloading_P5 sum_cuda_Offloading_P6

all: $(EXECUTABLES)

sum_serial_CPU_P0: $(COMMON_SRC) sum_serial_CPU_P0.c main.c sum_common.c
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
sum_omp_CPU_P1: $(COMMON_SRC) sum_omp_CPU_P1.c main.c sum_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

sum_omp_CPU_P2: $(COMMON_SRC) sum_omp_CPU_P2.c main.c sum_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

sum_omp_CPU_P3: $(COMMON_SRC) sum_omp_CPU_P3.c main.c sum_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
sum_omp_CPU_P4: $(COMMON_SRC) sum_omp_CPU_P4.c main.c sum_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
sum_omp_CPU_P5: $(COMMON_SRC) sum_omp_CPU_P5.c main.c sum_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
sum_omp_CPU_P6: $(COMMON_SRC) sum_omp_CPU_P6.c main.c sum_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
sum_omp_Offloading_P1: $(COMMON_SRC) sum_omp_Offloading_P1.c main.c sum_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

sum_omp_Offloading_P2: $(COMMON_SRC) sum_omp_Offloading_P2.c main.c sum_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
sum_omp_Offloading_P3: $(COMMON_SRC) sum_omp_Offloading_P3.c main.c sum_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

sum_omp_Offloading_P4: $(COMMON_SRC) sum_omp_Offloading_P4.c main.c sum_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
sum_cuda_Offloading_P1: $(COMMON_SRC) sum_cuda_Offloading_P1.cu main.c sum_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec
	
sum_cuda_Offloading_P2: $(COMMON_SRC) sum_cuda_Offloading_P2.cu main.c sum_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec
	
sum_cuda_Offloading_P3: $(COMMON_SRC) sum_cuda_Offloading_P3.cu main.c sum_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec
	
sum_cuda_Offloading_P4: $(COMMON_SRC) sum_cuda_Offloading_P4.cu main.c sum_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec
	
sum_cuda_Offloading_P5: $(COMMON_SRC) sum_cuda_Offloading_P5.cu main.c sum_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec
	
sum_cuda_Offloading_P6: $(COMMON_SRC) sum_cuda_Offloading_P6.cu main.c sum_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv *.png
