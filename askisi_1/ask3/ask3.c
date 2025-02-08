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

//===============================================================================================

void initialize(double *A, int N)
{

    #pragma omp parallel
    #pragma omp single
    {
        double t = getTime();
        for (int i = 0; i < N; i+=2)
        {
            #pragma omp task firstprivate(i) shared(A)
            {
                A[i] = work(i);

				if (i < N-1)
				{
                    printf("Task (%d, %d) is being executed by thread %d\n", i, i+1, omp_get_thread_num());
                    A[i+1] = work(i+1);
				}
				else
				{
					printf("Task %d is being executed by thread %d\n", i, omp_get_thread_num());
				}
            }
        }
        #pragma omp taskwait

        printf("\n-> Time: %lf seconds\n", getTime() - t);

	}
}

//===============================================================================================

int main(int argc, char *argv[])
{
	omp_set_dynamic(0);

    int N = 20;
	double *A = (double *)malloc(N * sizeof(double));

    for(int i=2;i<=16;i+=2){
        printf("-> Threads: %d\n\n",i);
        omp_set_num_threads(i);
	    initialize(A, N);
        printf("============================================\n");
    }

	return 0;
}