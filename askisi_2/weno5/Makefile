all: bench bench_avx

bench: bench.c weno.h
	gcc -O3 -fopt-info-vec -o bench bench.c -lm

bench_avx: bench_avx.c weno_avx.h
	gcc -mavx512f -o bench_avx bench_avx.c -lm

clean:
	rm -f bench
	rm -f bench_avx

