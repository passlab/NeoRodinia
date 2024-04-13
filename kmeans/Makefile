include ../common/environment.mk

# Source files
SRCS := kmeans_P0.c kmeans_omp_CPU_P1.c kmeans_omp_CPU_P2.c kmeans_omp_CPU_P3.c main.c kmeans_common.c

# Object files
OBJS := $(SRCS:.c=.o)

MATH_LIB = -lm
# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv result.txt kmeans_cuda.o

kmeans_P0: $(COMMON_SRC) kmeans_P0.o main.o
	$(CC) $(CFLAGS) $^ -o kmeans_P0_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
kmeans_omp_CPU_P1: $(COMMON_SRC) kmeans_omp_CPU_P1.c main.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o kmeans_omp_CPU_P1_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

kmeans_omp_CPU_P2: $(COMMON_SRC) kmeans_omp_CPU_P2.c main.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o kmeans_omp_CPU_P2_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

kmeans_omp_CPU_P3: $(COMMON_SRC) kmeans_omp_CPU_P3.c main.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o kmeans_omp_CPU_P3_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
kmeans_omp_CPU_P4: $(COMMON_SRC) kmeans_omp_CPU_P4.c main.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o kmeans_omp_CPU_P4_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
kmeans_omp_Offloading_P1: $(COMMON_SRC) kmeans_omp_Offloading_P1.c main.c
	$(CC) $(CFLAGS) $(OMP_FLAG) -fopenmp-targets=nvptx64 $^ -o kmeans_omp_Offloading_P1_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

kmeans_omp_Offloading_P2: $(COMMON_SRC) kmeans_omp_Offloading_P2.c main.c
	$(CC) $(CFLAGS) $(OMP_FLAG) -fopenmp-targets=nvptx64 $^ -o kmeans_omp_Offloading_P2_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
kmeans_omp_Offloading_P3: $(COMMON_SRC) kmeans_omp_Offloading_P3.c main.c
	$(CC) $(CFLAGS) $(OMP_FLAG) -fopenmp-targets=nvptx64 $^ -o kmeans_omp_Offloading_P3_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

kmeans_omp_Offloading_P4: $(COMMON_SRC) kmeans_omp_Offloading_P4.c main.c
	$(CC) $(CFLAGS) $(OMP_FLAG) -fopenmp-targets=nvptx64 $^ -o kmeans_omp_Offloading_P4_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
kmeans_cuda: $(COMMON_SRC) kmeans_cuda.cu main.c kmeans_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" kmeans_cuda.cu -c -o kmeans_cuda.o
	$(CC) $(CFLAGS) -c -o main.o main.c -fPIE
	$(NVCC) -Xcompiler "$(CFLAGS)" $(COMMON_DIR)/utils.c kmeans_cuda.o main.o -o kmeans_cuda_exec
