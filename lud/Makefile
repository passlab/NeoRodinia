include ../common/environment.mk

EXECUTABLES := lud_serial_CPU_P0 lud_omp_CPU_P1 lud_omp_CPU_P2 lud_omp_CPU_P3 lud_omp_Offloading_P1 lud_omp_Offloading_P2 lud_omp_Offloading_P3 lud_omp_Offloading_P4 lud_cuda_Offloading_P1

all: $(EXECUTABLES)

MATH_LIB := -lm

lud_serial_CPU_P0: $(COMMON_SRC) lud_serial_CPU_P0.c main.c lud_common.c
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
lud_omp_CPU_P1: $(COMMON_SRC) lud_omp_CPU_P1.c main.c lud_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

lud_omp_CPU_P2: $(COMMON_SRC) lud_omp_CPU_P2.c main.c lud_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

lud_omp_CPU_P3: $(COMMON_SRC) lud_omp_CPU_P3.c main.c lud_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
lud_omp_CPU_P4: $(COMMON_SRC) lud_omp_CPU_P4.c main.c lud_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
lud_omp_Offloading_P1: $(COMMON_SRC) lud_omp_Offloading_P1.c main.c lud_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

lud_omp_Offloading_P2: $(COMMON_SRC) lud_omp_Offloading_P2.c main.c lud_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
lud_omp_Offloading_P3: $(COMMON_SRC) lud_omp_Offloading_P3.c main.c lud_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

lud_omp_Offloading_P4: $(COMMON_SRC) lud_omp_Offloading_P4.c main.c lud_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)

lud_omp_Offloading_P5: $(COMMON_SRC) lud_omp_Offloading_P5.c main.c lud_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(MATH_LIB)
	
lud_cuda_Offloading_P1: $(COMMON_SRC) lud_cuda_Offloading_P1.cu main.c lud_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" lud_cuda_Offloading_P1.cu -c -o lud_cuda_Offloading_P1.o
	$(CC) $(CFLAGS) -c -o main.o main.c -fPIE
	$(CC) $(CFLAGS) -c -o lud_common.o lud_common.c -fPIE
	$(NVCC) -Xcompiler "$(CFLAGS)" $(COMMON_DIR)/utils.c lud_cuda_Offloading_P1.o main.o lud_common.o -o $@_nvcc_$(OPT_LEVEL)_exec

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv result.txt *.o *.png
