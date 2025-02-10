all: bench

bench: bench.o weno_simd.o weno_avx.o weno_original.o
	gcc -o bench bench.o weno_simd.o weno_avx.o weno_original.o -lm

bench.o: bench.c weno.h
	gcc -O1 -fopt-info-vec-all -o bench.o -c bench.c
	@echo

weno_simd.o: weno_simd.c
	gcc -O1 -fopenmp -fopt-info-vec -o weno_simd.o -c weno_simd.c
	@echo

weno_avx.o: weno_avx.c
	gcc -O1 -fopt-info-vec -o weno_avx.o -c weno_avx.c

weno_original.o: weno_original.c
	gcc -O1 -fopt-info-vec -o weno_original.o -c weno_original.c

clean:
	@rm -f bench
	@rm -f bench.o
	@rm -f weno_simd.o

