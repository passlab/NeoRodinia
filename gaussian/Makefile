include ../common/environment.mk

EXECUTABLES := gaussian_serial_CPU_P0 gaussian_omp_CPU_P1 gaussian_omp_CPU_P2 gaussian_omp_CPU_P3 gaussian_omp_Offloading_P1 gaussian_omp_Offloading_P2 gaussian_omp_Offloading_P3 gaussian_omp_Offloading_P4 gaussian_cuda_Offloading_P1

all: $(EXECUTABLES)

MATH_LIB := -lm

gaussian_serial_CPU_P0: $(COMMON_SRC) gaussian_serial_CPU_P0.c main.c gaussian_common.c
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
gaussian_omp_CPU_P1: $(COMMON_SRC) gaussian_omp_CPU_P1.c main.c gaussian_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

gaussian_omp_CPU_P2: $(COMMON_SRC) gaussian_omp_CPU_P2.c main.c gaussian_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

gaussian_omp_CPU_P3: $(COMMON_SRC) gaussian_omp_CPU_P3.c main.c gaussian_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
gaussian_omp_Offloading_P1: $(COMMON_SRC) gaussian_omp_Offloading_P1.c main.c gaussian_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

gaussian_omp_Offloading_P2: $(COMMON_SRC) gaussian_omp_Offloading_P2.c main.c gaussian_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
gaussian_omp_Offloading_P3: $(COMMON_SRC) gaussian_omp_Offloading_P3.c main.c gaussian_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

gaussian_omp_Offloading_P4: $(COMMON_SRC) gaussian_omp_Offloading_P4.c main.c gaussian_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

gaussian_omp_Offloading_P5: $(COMMON_SRC) gaussian_omp_Offloading_P5.c main.c gaussian_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
gaussian_cuda_Offloading_P1: $(COMMON_SRC) gaussian_cuda_Offloading_P1.cu main.c gaussian_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" gaussian_cuda_Offloading_P1.cu -c -o gaussian_cuda_Offloading_P1.o
	$(CC) $(CFLAGS) -c -o main.o main.c -fPIE
	$(CC) $(CFLAGS) -c -o gaussian_common.o gaussian_common.c -fPIE
	$(NVCC) -Xcompiler "$(CFLAGS)" $(COMMON_DIR)/utils.c gaussian_cuda_Offloading_P1.o main.o gaussian_common.o -o $@_nvcc_$(OPT_LEVEL)_exec

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv result.txt gaussian_cuda.o *.png *.o
