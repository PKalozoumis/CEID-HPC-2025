CC=gcc
CFLAGS=-fopenmp -Wall  -fcf-protection=none -fno-stack-protector -foffload=nvptx-none

.PHONY: clean

all: clean gpu_omp

cpu_calculations.o: cpu_calculations.c
	${CC} ${CFLAGS} -c cpu_calculations.c -o cpu_calculations.o

gpu_omp.o: gpu_omp.c
	${CC} ${CFLAGS} -c gpu_omp.c -o gpu_omp.o

gpu_omp: cpu_calculations.o gpu_omp.o
	${CC} ${CFLAGS} cpu_calculations.o gpu_omp.o -o gpu_omp

gpu_omp_bad: gpu_omp.c
	gcc -fopenmp -fcf-protection=none -fno-stack-protector -foffload=nvptx-none -o gpu_omp_bad gpu_omp_bad.c

clean:
	rm -f gpu_omp.o
	rm -f cpu_calculations.o