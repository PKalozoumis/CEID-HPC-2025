CC=gcc
CFLAGS=-fopenmp -Wall -foffload=nvptx-none -fcf-protection=none -fno-stack-protector -O3

.PHONY: clean

all: gpu_omp

cpu_calculations.o: cpu_calculations.c
	${CC} ${CFLAGS} -fopt-info-vec -c cpu_calculations.c -o cpu_calculations.o

#================================================================================

gpu_omp.o: gpu_omp.c
	${CC} ${CFLAGS} -c gpu_omp.c -o gpu_omp.o

gpu_omp: cpu_calculations.o gpu_omp.o
	${CC} ${CFLAGS} cpu_calculations.o gpu_omp.o -o gpu_omp

#================================================================================

clean:
	rm -f gpu_omp.o
	rm -f cpu_calculations.o
	rm -f gpu_omp.o
	rm -f cpu_calculations.o
	rm -f gpu_omp
	rm -f cpu_calculations
	rm -f gpu_omp
	rm -f cpu_calculations