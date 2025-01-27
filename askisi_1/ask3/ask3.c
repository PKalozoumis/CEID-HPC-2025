#include <stdio.h>
#include <omp.h>
#include <sys/time.h>
#include <stdlib.h>

extern double work(int i);

double getTime(void)
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}

void initialize(double *A, int N)
{
	#pragma omp parallel
	#pragma omp single
	{
		double t = getTime();
		for (int i = 0; i < N; i++)
		{
			#pragma omp task firstprivate(i) shared(A)
			{
				printf("Task %d is being executed by thread %d\n", i, omp_get_thread_num());

				A[i] = work(i);
			}
		}

		#pragma omp taskwait
		printf("%lf\n", getTime() - t);
	}
}

int main(int argc, char *argv[])
{
	omp_set_dynamic(0);
	omp_set_num_threads(1);
	int N = 100;
	double *A = (double *)malloc(N * sizeof(double));

	initialize(A, N);

	return 0;
}