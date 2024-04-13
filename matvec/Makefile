include ../common/environment.mk

EXECUTABLES := matvec_serial_CPU_P0 matvec_omp_CPU_P1 matvec_omp_CPU_P2 matvec_omp_CPU_P3 matvec_omp_Offloading_P1 matvec_omp_Offloading_P2 matvec_omp_Offloading_P3 matvec_omp_Offloading_P4 matvec_cuda_Offloading_P1 matvec_cuda_Offloading_P2 matvec_cuda_Offloading_P3

all: $(EXECUTABLES)

matvec_serial_CPU_P0: $(COMMON_SRC) matvec_serial_CPU_P0.c main.c matvec_common.c
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matvec_omp_CPU_P1: $(COMMON_SRC) matvec_omp_CPU_P1.c main.c matvec_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matvec_omp_CPU_P2: $(COMMON_SRC) matvec_omp_CPU_P2.c main.c matvec_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matvec_omp_CPU_P3: $(COMMON_SRC) matvec_omp_CPU_P3.c main.c matvec_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matvec_omp_Offloading_P1: $(COMMON_SRC) matvec_omp_Offloading_P1.c main.c matvec_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matvec_omp_Offloading_P2: $(COMMON_SRC) matvec_omp_Offloading_P2.c main.c matvec_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matvec_omp_Offloading_P3: $(COMMON_SRC) matvec_omp_Offloading_P3.c main.c matvec_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matvec_omp_Offloading_P4: $(COMMON_SRC) matvec_omp_Offloading_P4.c main.c matvec_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matvec_omp_Offloading_P5: $(COMMON_SRC) matvec_omp_Offloading_P5.c main.c matvec_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matvec_omp_Offloading_P6: $(COMMON_SRC) matvec_omp_Offloading_P6.c main.c matvec_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matvec_omp_Offloading_P6.5: $(COMMON_SRC) matvec_omp_Offloading_P6.5.c main.c matvec_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

matvec_omp_Offloading_P7: $(COMMON_SRC) matvec_omp_Offloading_P7.c main.c matvec_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
matvec_cuda_Offloading_P1: $(COMMON_SRC) matvec_cuda_Offloading_P1.cu main.c matvec_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec

matvec_cuda_Offloading_P2: $(COMMON_SRC) matvec_cuda_Offloading_P2.cu main.c matvec_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec
	
matvec_cuda_Offloading_P3: $(COMMON_SRC) matvec_cuda_Offloading_P3.cu main.c matvec_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv *.png
