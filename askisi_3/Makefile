CC=gcc
CFLAGS=-fopenmp

.PHONY: clean

all: clean gpu_omp

gpu_omp: gpu_omp.c

clean:
	@if [ -f "gpu_omp" ]; then rm gpu_omp; fi