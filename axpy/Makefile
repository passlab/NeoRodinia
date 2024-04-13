include ../common/environment.mk

EXECUTABLES := axpy_serial_CPU_P0 axpy_omp_CPU_P1 axpy_omp_CPU_P2 axpy_omp_CPU_P3 axpy_omp_Offloading_P1 axpy_omp_Offloading_P2 axpy_omp_Offloading_P3 axpy_omp_Offloading_P4 axpy_cuda_Offloading_P1

all: $(EXECUTABLES)

axpy_serial_CPU_P0: $(COMMON_SRC) axpy_serial_CPU_P0.c main.c axpy_common.c
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
axpy_omp_CPU_P1: $(COMMON_SRC) axpy_omp_CPU_P1.c main.c axpy_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

axpy_omp_CPU_P2: $(COMMON_SRC) axpy_omp_CPU_P2.c main.c axpy_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

axpy_omp_CPU_P3: $(COMMON_SRC) axpy_omp_CPU_P3.c main.c axpy_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
axpy_omp_Offloading_P1: $(COMMON_SRC) axpy_omp_Offloading_P1.c main.c axpy_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

axpy_omp_Offloading_P2: $(COMMON_SRC) axpy_omp_Offloading_P2.c main.c axpy_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

axpy_omp_Offloading_P3: $(COMMON_SRC) axpy_omp_Offloading_P3.c main.c axpy_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

axpy_omp_Offloading_P4: $(COMMON_SRC) axpy_omp_Offloading_P4.c main.c axpy_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
axpy_cuda_Offloading_P1: $(COMMON_SRC) axpy_cuda_Offloading_P1.cu main.c axpy_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv *.png *.svg
