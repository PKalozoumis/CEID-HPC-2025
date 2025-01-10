all: bench bench_simd bench_avx

bench: bench.c weno.h
	gcc -O3 -fopt-info-vec -o bench bench.c -lm

bench_simd: bench_simd.c weno_simd.h
	gcc -O3 -fopenmp -fopt-info-vec -o bench_simd bench_simd.c -lm

bench_avx: bench_avx.c weno_avx.h
	gcc -O3 -mavx512vl -fopt-info-vec -o bench_avx bench_avx.c -lm

clean:
	rm -f bench_simd
	rm -f bench_avx
	rm -f bench

