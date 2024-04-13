include ../common/environment.mk

EXECUTABLES := stencil_serial_CPU_P0 stencil_omp_CPU_P1 stencil_omp_CPU_P2 stencil_omp_CPU_P3 stencil_omp_CPU_P4 stencil_omp_Offloading_P1 stencil_omp_Offloading_P2 stencil_omp_Offloading_P3 stencil_omp_Offloading_P4 stencil_cuda_Offloading_P1 stencil_cuda_Offloading_P2

all: $(EXECUTABLES)

stencil_serial_CPU_P0: $(COMMON_SRC) stencil_serial_CPU_P0.c main.c stencil_common.c
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
stencil_omp_CPU_P1: $(COMMON_SRC) stencil_omp_CPU_P1.o main.c stencil_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

stencil_omp_CPU_P2: $(COMMON_SRC) stencil_omp_CPU_P2.o main.c stencil_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

stencil_omp_CPU_P3: $(COMMON_SRC) stencil_omp_CPU_P3.o main.c stencil_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
stencil_omp_CPU_P4: $(COMMON_SRC) stencil_omp_CPU_P4.o main.c stencil_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
stencil_omp_Offloading_P1: $(COMMON_SRC) stencil_omp_Offloading_P1.c main.c stencil_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

stencil_omp_Offloading_P2: $(COMMON_SRC) stencil_omp_Offloading_P2.c main.c stencil_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
stencil_omp_Offloading_P3: $(COMMON_SRC) stencil_omp_Offloading_P3.c main.c stencil_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

stencil_omp_Offloading_P4: $(COMMON_SRC) stencil_omp_Offloading_P4.c main.c stencil_common.c
	$(CC) $(CFLAGS) $(OMP_FLAG) $(OFFLOADING_FLAG) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
stencil_cuda_Offloading_P1: $(COMMON_SRC) stencil_cuda_Offloading_P1.cu main.c stencil_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec

stencil_cuda_Offloading_P2: $(COMMON_SRC) stencil_cuda_Offloading_P2.cu main.c stencil_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@_nvcc_$(OPT_LEVEL)_exec

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) *exec *.csv *.png
