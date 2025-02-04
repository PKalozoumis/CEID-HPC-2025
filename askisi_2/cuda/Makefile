.PHONY: clean

all: clean cuda

cpu_calculations.o: cpu_calculations.c
	gcc -fopenmp -c cpu_calculations.c -o cpu_calculations.o

cuda: cpu_calculations.o cuda.cu
	nvcc -Xcompiler -fopenmp cuda.cu cpu_calculations.o -o cuda

clean:
	rm -f cuda cpu_calculations.o cpu_calculations