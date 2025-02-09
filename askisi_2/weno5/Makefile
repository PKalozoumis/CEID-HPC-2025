all: bench

bench: bench.o weno_simd.o weno_avx.o
	gcc -o bench bench.o weno_simd.o weno_avx.o -lm

bench.o: bench.c weno.h
	gcc -O2 -fopt-info-vec -o bench.o -c bench.c
	@echo

weno_simd.o: weno_simd.c
	gcc -O1 -fopenmp -fopt-info-vec -o weno_simd.o -c weno_simd.c
	@echo

weno_avx.o: weno_avx.c
	gcc -O1 -mavx512vl -fopt-info-vec -o weno_avx.o -c weno_avx.c

clean:
	@rm -f bench
	@rm -f bench.o
	@rm -f weno_simd.o

