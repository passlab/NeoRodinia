include ../common/environment.mk

EXECUTABLES := matmul_serial_CPU_P0 matmul_omp_CPU_P1 matmul_omp_CPU_P2 matmul_omp_CPU_P3 matmul_omp_CPU_P4 matmul_omp_Offloading_P1 matmul_omp_Offloading_P2 matmul_omp_Offloading_P3 matmul_omp_Offloading_P4 matmul_cuda_Offloading_P1 matmul_cuda_Offloading_P2 matmul_cuda_Offloading_P3

all: $(EXECUTABLES)

matmul_serial_CPU_P0: $(COMMON_SRC) matmul_serial_CPU_P0.c main.c matmul_common.c
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matmul_omp_CPU_P1: $(COMMON_SRC) matmul_omp_CPU_P1.c main.c matmul_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matmul_omp_CPU_P2: $(COMMON_SRC) matmul_omp_CPU_P2.c main.c matmul_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matmul_omp_CPU_P3: $(COMMON_SRC) matmul_omp_CPU_P3.c main.c matmul_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matmul_omp_CPU_P4: $(COMMON_SRC) matmul_omp_CPU_P4.c main.c matmul_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matmul_omp_Offloading_P1: $(COMMON_SRC) matmul_omp_Offloading_P1.c main.c matmul_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG)  $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matmul_omp_Offloading_P2: $(COMMON_SRC) matmul_omp_Offloading_P2.c main.c matmul_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG)  $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matmul_omp_Offloading_P3: $(COMMON_SRC) matmul_omp_Offloading_P3.c main.c matmul_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG)  $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matmul_omp_Offloading_P4: $(COMMON_SRC) matmul_omp_Offloading_P4.c main.c matmul_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG)  $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matmul_omp_Offloading_P5: $(COMMON_SRC) matmul_omp_Offloading_P5.c main.c matmul_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG)  $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matmul_cuda_Offloading_P1: $(COMMON_SRC) matmul_cuda_Offloading_P1.cu main.c matmul_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec

matmul_cuda_Offloading_P2: $(COMMON_SRC) matmul_cuda_Offloading_P2.cu main.c matmul_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec
	
matmul_cuda_Offloading_P3: $(COMMON_SRC) matmul_cuda_Offloading_P3.cu main.c matmul_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv *.png
