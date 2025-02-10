#include <stdio.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

#ifndef WENOEPS
#define WENOEPS 1.e-6
#endif

#include "weno.h"
#include "weno_simd.h"
#include "weno_avx.h"
#include "weno_original.h"

void print_array(float* arr, int N)
{
	for (int i = 0; i < N; i++)
		printf("%f ", arr[i]);

	printf("\n");
}

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

void allocate(int NENTRIES, float** a, float** b, float** c, float** d, float** e, float** f, float** gold, float** result)
{
	*a = myalloc(NENTRIES, 0);
	*b = myalloc(NENTRIES, 0);
	*c = myalloc(NENTRIES, 0);
	*d = myalloc(NENTRIES, 0);
	*e = myalloc(NENTRIES, 0);
	*f = myalloc(NENTRIES, 0);
	*gold = myalloc(NENTRIES, 0); //The reference implementation, for error checking
	*result = myalloc(NENTRIES, 0); //Result of our own implementation
}

typedef const float* const cptrc;
void (*implementation[])(cptrc, cptrc, cptrc, cptrc, cptrc, float* const, const int) = {weno_minus_original, weno_minus_reference, weno_minus_simd, weno_minus_avx};

//Each benchmarks is performed for a specific NENTRIES
//We test all implementations
double benchmark(const int NENTRIES, float** data, float* out, const int verbose, int impl)
{
	double t = get_wtime();
	implementation[impl](data[0], data[1], data[2], data[3], data[4], out, NENTRIES);
	t = get_wtime() - t;

	return t;
}

//================================================================================================================

//The benchmarks that were initially provided
void existing_benchmark(int argc, char *argv[])
{
	const int debug = 0;

	if (debug)
	{
		//benchmark(4, 1, 1, "debug");
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
			//benchmark(nentries, ntimes, 0, "cache");
		}
	}

	/* performance on data streams */
	{
		const double desired_mb = 128 * 16;
		const int nentries = (int)floor(desired_mb * 1024. * 1024. / 7 / sizeof(float));

		for (int i = 0; i < 4; ++i)
		{
			printf("*************** STREAM-LIKE BENCHMARK (RUN %d) **************************\n", i);
			//benchmark(nentries, 1, 0, "stream");
		}
	}
}

//================================================================================================================

int main(int argc, char *argv[])
{
	printf("Hello, weno benchmark!\n\n");

	//Open shared memory from Python driver program
    //===========================================================================
    int fd;
    double* shmem = NULL;
    
    if (argc == 2) //2nd argument will be the shared memory name
    {
        fd = shm_open(argv[1], O_RDWR, 0);

        if (fd == -1)
            {perror("Could not open shared memory"); exit(1);}

        shmem = (double*)mmap(NULL, 5*4*sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

        if (close(fd) == -1)
            {perror("Could not close file descriptor"); exit(1);}

        if (shmem == MAP_FAILED)
            {perror("mmap failure"); exit(1);}
    }

	//Start benchmarking
	//===========================================================================

	double t = get_wtime();

	int N[] = {1e6, 1e7, 1.5e7 ,0.5e8, 1e8};
	double times[] = {0.0, 0.0, 0.0, 0.0};

	for (int i = 0; i < sizeof(N)/sizeof(int); i++)
	{
		float* a;
		float* b;
		float* c;
		float* d;
		float* e;
		float* f;
		float* gold;
		float* result;

		const int NENTRIES = 16 * (N[i] / 16);
		printf("Entires: %d After rounding down: %d\n\n", N[i], NENTRIES);

		allocate(NENTRIES, &a, &b, &c, &d, &e, &f, &gold, &result);
		float* data[8] = {a, b, c, d, e, f};

		double local_times[] = {0.0, 0.0, 0.0, 0.0};

        //printf("*************** BENCHMARK **************************\n");
	    printf("-> Running Reference (original) with NENTRIES = %e\n",(float)NENTRIES);

		//Run original reference implementation
		//---------------------------------------------------------------------------------
		for (int j = 0; j < 8; j++)
			local_times[0] += benchmark(NENTRIES, data, gold, 0, 0);

		times[0] += local_times[0];
 

        //printf("*************** BENCHMARK **************************\n");
	    printf("-> Running Reference (automatic vectorization) with NENTRIES = %e\n",(float)NENTRIES);
		//Run reference implementation with automatic vectorization
		//---------------------------------------------------------------------------------
		for (int j = 0; j < 8; j++)
			local_times[1] += benchmark(NENTRIES, data, gold, 0, 1);

		times[1] += local_times[1];


        //printf("*************** BENCHMARK **************************\n");
	    printf("-> Running OpenMP SIMD with NENTRIES = %e\n",(float)NENTRIES);

		//Run OpenMP SIMD
		//---------------------------------------------------------------------------------
		for (int j = 0; j < 8; j++)
			local_times[2] += benchmark(NENTRIES, data, result, 0, 2);

		times[2] += local_times[2];

		double tol = 1e-5;
		printf("minus: verifying OpenMP SIMD accuracy with tolerance %.5e...", tol);
		check_error(tol, gold, result, NENTRIES);
		printf("passed!\n\n");

        //printf("*************** BENCHMARK **************************\n");
	    printf("-> Running AVX with NENTRIES = %e\n",(float)NENTRIES);

		//Run AVX
		//---------------------------------------------------------------------------------
		for (int j = 0; j < 8; j++)
			local_times[3] += benchmark(NENTRIES, data, result, 0, 3);

		times[3] += local_times[3];

		tol = 1;
		printf("minus: verifying AVX accuracy with tolerance %.5e...", tol);
		check_error(tol, gold, result, NENTRIES);
		printf("passed!\n\n");

		//---------------------------------------------------------------------------------

		printf("Reference time (original): %.03lfs\n", local_times[0]);
		printf("Reference time (automatic vectorization): %.03lfs\n", local_times[1]);
		printf("SIMD time: %.03lfs\n", local_times[2]);
		printf("AVX time: %.03lfs\n", local_times[3]);
        printf("\n===============================================================\n\n");

		free(a);
		free(b);
		free(c);
		free(d);
		free(e);
        free(f);
		free(gold);
		free(result);

		//Copy local times to the shared memory
		if (shmem != NULL)
			memcpy((void*)(shmem + i*4), (void*)local_times, 4*sizeof(double));
	}

	printf("Total Reference time (original): %.03lfs\n", times[0]);
	printf("Total Reference time (automatic vectorization): %.03lfs\n", times[1]);
	printf("Total SIMD time: %.03lfs\n", times[2]);
	printf("Total AVX time: %.03lfs\n\n", times[3]);

	printf("Total time: %.03lfs\n\n", get_wtime() - t);

	if (shmem != NULL)
    {
        if (munmap(shmem, sizeof(double)) == -1)
            {perror("unmap failure"); exit(1);}
    }

	return 0;
}
