#include <stdio.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef WENOEPS
#define WENOEPS 1.e-6
#endif

#include "weno.h"
#include "weno_simd.h"
#include "weno_avx.h"

typedef enum {SIMD, AVX, BOTH} MODE;

//================================================================================================================

float *myalloc(const int NENTRIES, const int verbose)
{
	const int initialize = 1;
	enum
	{
		alignment_bytes = 64
	};
	float *tmp = NULL;

	const int result = posix_memalign((void **)&tmp, alignment_bytes, sizeof(float) * NENTRIES);
	assert(result == 0);

	if (initialize)
	{
		for (int i = 0; i < NENTRIES; ++i)
			tmp[i] = drand48();

		if (verbose)
		{
			for (int i = 0; i < NENTRIES; ++i)
				printf("tmp[%d] = %f\n", i, tmp[i]);
			printf("==============\n");
		}
	}
	return tmp;
}

//================================================================================================================

double get_wtime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec * 1e-6;
}

//================================================================================================================

void check_error(const double tol, float ref[], float val[], const int N)
{
	static const int verbose = 0;

	for (int i = 0; i < N; ++i)
	{
		assert(!isnan(ref[i]));
		assert(!isnan(val[i]));

		const double err = ref[i] - val[i];
		const double relerr = err / fmaxf(FLT_EPSILON, fmaxf(fabs(val[i]), fabs(ref[i])));

		if (verbose)
			printf("+%1.1e,", relerr);

		if (fabs(relerr) >= tol && fabs(err) >= tol)
			printf("\nPosition %d: Ref: %e Val: %e -> Err: %e Relerr: %e\n", i, ref[i], val[i], err, relerr);

		assert(fabs(relerr) < tol || fabs(err) < tol);
	}

	if (verbose)
		printf("\t");
}

//================================================================================================================

typedef const float* const cptrc;
void (*implementation[2])(cptrc, cptrc, cptrc, cptrc, cptrc, float* const, const int) = {weno_minus_simd, weno_minus_avx};

//Each benchmarks is performed for a specific NENTRIES
//We test all implementations
void benchmark(const int NENTRIES_, const int NTIMES, const int verbose, char *benchmark_name)
{
	const int NENTRIES = 4 * (NENTRIES_ / 4);

	printf("*************** BENCHMARK **************************\n");
	printf("nentries set to %e\n", (float)NENTRIES);

	float *const a = myalloc(NENTRIES, verbose);
	float *const b = myalloc(NENTRIES, verbose);
	float *const c = myalloc(NENTRIES, verbose);
	float *const d = myalloc(NENTRIES, verbose);
	float *const e = myalloc(NENTRIES, verbose);
	float *const f = myalloc(NENTRIES, verbose);
	float *const gold = myalloc(NENTRIES, verbose); //The reference implementation, for error checking
	float *const result = myalloc(NENTRIES, verbose); //Result of our own implementation

	double times[3] = {0.0, 0.0, 0.0};

	//For each implementation...
	for (int i = 0; i < 1; i++)
	{
		//Run the implementation many times
		for (int j = 0; j < 8; j++)
		{
			double t = get_wtime();
			weno_minus_reference(a, b, c, d, e, gold, NENTRIES);
			t = get_wtime() - t;
			times[0] += t;

			//printf("Testing implementation: %s\n", i == 0 ? "weno_simd" : "weno_avx");
			t = get_wtime();
			implementation[1](a, b, c, d, e, result, NENTRIES);
			t = get_wtime() - t;
			times[i+1] += t;
		}

		const double tol = 1e-5;
		printf("minus: verifying accuracy with tolerance %.5e...", tol);
		check_error(tol, gold, result, NENTRIES);
		printf("passed!\n\n");
	}

	printf("Reference time: %.03lfs\n", times[0]);
	printf("SIMD time: %.03lfs\n", times[1]);
	printf("AVX time: %.03lfs\n\n", times[2]);

	free(a);
	free(b);
	free(c);
	free(d);
	free(e);
	free(gold);
	free(result);
}

//================================================================================================================

//The benchmarks that were initially provided
void existing_benchmark(int argc, char *argv[])
{
	const int debug = 0;

	if (debug)
	{
		benchmark(4, 1, 1, "debug");
		return;
	}

	/* performance on cache hits */
	{
		const double desired_kb = 32 * 8 * 0.5;								/* we want to fill 50% of the dcache */
		const int nentries = floor(desired_kb * 1024. / 7 / sizeof(float)); // floor(desired_kb * 1024. / 7 / sizeof(float));
		const int ntimes = (int)floor(2. / (1e-7 * nentries));

		for (int i = 0; i < 4; ++i)
		{
			printf("*************** PEAK-LIKE BENCHMARK (RUN %d) **************************\n", i);
			benchmark(nentries, ntimes, 0, "cache");
		}
	}

	/* performance on data streams */
	{
		const double desired_mb = 128 * 16;
		const int nentries = (int)floor(desired_mb * 1024. * 1024. / 7 / sizeof(float));

		for (int i = 0; i < 4; ++i)
		{
			printf("*************** STREAM-LIKE BENCHMARK (RUN %d) **************************\n", i);
			benchmark(nentries, 1, 0, "stream");
		}
	}
}

//================================================================================================================

int main(int argc, char *argv[])
{
	printf("Hello, weno benchmark!\n\n");

	//Parse implemenation from argument

	double t = get_wtime();

	int N[8] = {5000, 10000, 50000, 100000, 500000, 1e6, 0.5e8, 1e8};

	for (int i = 0; i < 8; i++)
	{
		const int ntimes = (int)floor(2. / (1e-7 * N[i]));
		benchmark(N[i], ntimes, 0, "goofy benchmark");
	}

	printf("Total time: %.03lfs\n\n", get_wtime() - t);

	return 0;
}
