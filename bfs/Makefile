include ../common/environment.mk

EXECUTABLES := bfs_serial_CPU_P0 bfs_omp_CPU_P1 bfs_omp_CPU_P2 bfs_omp_CPU_P3 bfs_omp_Offloading_P1 bfs_omp_Offloading_P2 bfs_omp_Offloading_P3 bfs_omp_Offloading_P4 bfs_cuda_Offloading_P1

all: $(EXECUTABLES)

bfs_serial_CPU_P0: $(COMMON_SRC) bfs_serial_CPU_P0.c main.c bfs_common.c
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
bfs_omp_CPU_P1: $(COMMON_SRC) bfs_omp_CPU_P1.c main.c bfs_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

bfs_omp_CPU_P2: $(COMMON_SRC) bfs_omp_CPU_P2.c main.c bfs_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

bfs_omp_CPU_P3: $(COMMON_SRC) bfs_omp_CPU_P3.c main.c bfs_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
bfs_omp_Offloading_P1: $(COMMON_SRC) bfs_omp_Offloading_P1.c main.c bfs_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

bfs_omp_Offloading_P2: $(COMMON_SRC) bfs_omp_Offloading_P2.c main.c bfs_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
bfs_omp_Offloading_P3: $(COMMON_SRC) bfs_omp_Offloading_P3.c main.c bfs_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

bfs_omp_Offloading_P4: $(COMMON_SRC) bfs_omp_Offloading_P4.c main.c bfs_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
bfs_cuda_Offloading_P1: $(COMMON_SRC) bfs_cuda_Offloading_P1.cu main.c bfs_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" bfs_cuda_Offloading_P1.cu -c -o bfs_cuda_Offloading_P1.o
	$(CC) $(CFLAGS) -c -o main.o main.c -fPIE
	$(CC) $(CFLAGS) -c -o bfs_common.o bfs_common.c -fPIE
	$(NVCC) -Xcompiler "$(CFLAGS)" $(COMMON_DIR)/utils.c bfs_cuda_Offloading_P1.o main.o bfs_common.o -o $@_nvcc_$(OPT_LEVEL)_exec

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv result.txt bfs_cuda.o *.png
