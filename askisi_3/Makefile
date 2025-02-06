CC=nvcc
CFLAGS=-Xcompiler -fopenmp -Xcompiler -Wall -Xcompiler -fcf-protection=none -arch=sm_75

.PHONY: clean

all: gpu_omp gpu_omp_alt

cpu_calculations.o: ../askisi_2/cuda/cpu_calculations.c
	${CC} ${CFLAGS} -Xcompiler -O1 -Xcompiler -fopt-info-vec -c ../askisi_2/cuda/cpu_calculations.c -o cpu_calculations.o

#================================================================================

gpu_omp.o: gpu_omp.c
	${CC} ${CFLAGS} -O3 -c gpu_omp.c -o gpu_omp.o

gpu_omp: cpu_calculations.o gpu_omp.o
	${CC} ${CFLAGS} cpu_calculations.o gpu_omp.o -o gpu_omp

#================================================================================

gpu_omp_alt.o: gpu_omp_alt.c
	${CC} ${CFLAGS} -O3 -c gpu_omp_alt.c -o gpu_omp_alt.o

gpu_omp_alt: gpu_omp_alt.o
	${CC} ${CFLAGS} -o gpu_omp_alt gpu_omp_alt.o cpu_calculations.o

#================================================================================

clean:
	rm -f gpu_omp.o
	rm -f cpu_calculations.o
	rm -f gpu_omp.o
	rm -f gpu_omp_alt.o
	rm -f cpu_calculations.o
	rm -f gpu_omp
	rm -f cpu_calculations
	rm -f gpu_omp
	rm -f gpu_omp_alt
	rm -f cpu_calculations