include ../common/environment.mk

EXECUTABLES := b+tree_serial_CPU_P0 b+tree_omp_CPU_P1 b+tree_omp_CPU_P2 b+tree_omp_CPU_P3 b+tree_omp_Offloading_P1 b+tree_omp_Offloading_P2 b+tree_omp_Offloading_P3 b+tree_omp_Offloading_P4 b+tree_cuda_Offloading_P1

all: $(EXECUTABLES)

MATH_LIB = -lm

b+tree_serial_CPU_P0: $(COMMON_SRC) b+tree_serial_CPU_P0.c main.c b+tree_common.c
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
b+tree_omp_CPU_P1: $(COMMON_SRC) b+tree_omp_CPU_P1.c main.c b+tree_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

b+tree_omp_CPU_P2: $(COMMON_SRC) b+tree_omp_CPU_P2.c main.c b+tree_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

b+tree_omp_CPU_P3: $(COMMON_SRC) b+tree_omp_CPU_P3.c main.c b+tree_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
b+tree_omp_Offloading_P1: $(COMMON_SRC) b+tree_omp_Offloading_P1.c main.c b+tree_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

b+tree_omp_Offloading_P2: $(COMMON_SRC) b+tree_omp_Offloading_P2.c main.c b+tree_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
b+tree_omp_Offloading_P3: $(COMMON_SRC) b+tree_omp_Offloading_P3.c main.c b+tree_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

b+tree_omp_Offloading_P4: $(COMMON_SRC) b+tree_omp_Offloading_P4.c main.c b+tree_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
b+tree_cuda_Offloading_P1: $(COMMON_SRC) b+tree_cuda_Offloading_P1.cu main.c b+tree_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" b+tree_cuda_Offloading_P1.cu -c -o b+tree_cuda_Offloading_P1.o
	$(CC) $(CFLAGS) -c -o main.o main.c -fPIE
	$(CC) $(CFLAGS) -c -o b+tree_common.o b+tree_common.c -fPIE
	$(NVCC) -Xcompiler "$(CFLAGS)" $(COMMON_DIR)/utils.c b+tree_cuda_Offloading_P1.o main.o b+tree_common.o -o $@_nvcc_$(OPT_LEVEL)_exec

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv result.txt b+tree_cuda.o *.png *.o
