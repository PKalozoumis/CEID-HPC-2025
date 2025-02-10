#include "cpu_calculations.h"
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

//========================================================================================================

static void cpu_test(float *A, float *B, float *C, float *D, int N, float* E, float* F){
    
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float resAC = 0.0f, resBD = 0.0f, resAD = 0.0f, resBC = 0.0f;

            for (int k = 0; k < N; k++)
            {
                resAC += A[i * N + k] * C[k * N + j];
                resBD += B[i * N + k] * D[k * N + j];
                resAD += A[i * N + k] * D[k * N + j];
                resBC += B[i * N + k] * C[k * N + j];
            }

            E[i*N + j] = resAC - resBD;
            F[i*N + j] = resAD + resBC; 
        }
    }
}

//========================================================================================================

double cpu_calculation(float *A, float *B, float *C, float *D, int N, float* E, float* F)
{
    printf("Performing CPU calculations...\n");
    fflush(stdout);
    double t = get_wtime();
    
    cpu_test(A,B,C,D,N,E,F);
    
    t = get_wtime()-t;
    printf("Total time for CPU calculations: %.03lfs\n\n", t);
    fflush(stdout);

    return t;
}
//========================================================================================================

double matrix_comparison(float* cpuE, float* cpuF, float* gpuE, float* gpuF, int N){
    printf("Comparing results... ");
    fflush(stdout);
    double t = get_wtime();

    int error=0;
    double tolerance = 1e-1;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if (fabs(cpuE[i*N + j] - gpuE[i * N + j]) > tolerance || 
                fabs(cpuF[i*N + j] - gpuF[i * N + j]) > tolerance){
                error=1;
                break;
            }
        }
        if(error==1){
            break;
        }
    }

    if(error==0){
        printf("Successful comparison\n");
        fflush(stdout);
    }else{
        printf("Comparison failed\n");
        fflush(stdout);
    }

    t = get_wtime()-t;
    printf("Total time for result comparison in CPU: %.03lfs\n\n", t);
    fflush(stdout);

    return t;
}

//========================================================================================================

static void initialize_matrix_(float** matrix, int N)
{
    srand(time(NULL) + 1000 * omp_get_thread_num());

    if (posix_memalign((void**)matrix, 32, N*N*sizeof(float)) != 0)
        {perror("Could not allocate aligned memory"); exit(EXIT_FAILURE);}

    for (int i = 0; i < N * N; i++)
    {
        (*matrix)[i] = (rand() / (float)RAND_MAX) * 10;
    }
}

//========================================================================================================

double get_wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

//========================================================================================================

void print_matrix(float *matrix, int N)
{

    printf("==================================================================\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.02f", matrix[i * N + j]);

            if (j < N - 1)
                printf("\t");
        }

        printf("\n");
    }

    fflush(stdout);
}

//========================================================================================================

double initialize_matrices(float** A, float** B, float** C, float** D, int N)
{
    printf("Initializing matrices in host...\n");
    fflush(stdout);
    double t = get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            initialize_matrix_(A, N);

            #pragma omp task
            initialize_matrix_(B, N);

            #pragma omp task
            initialize_matrix_(C, N);

            #pragma omp task
            initialize_matrix_(D, N);
        }
    }

    t = get_wtime()-t;
    printf("Total time for parallel initialization: %.03lfs\n\n", t);
    fflush(stdout);

    return t;
}