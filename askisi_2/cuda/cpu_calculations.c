#include "cpu_calculations.h"
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

//========================================================================================================

void cpu_matrix_add(float* restrict AB, float* restrict CD, float* restrict result, int N)
{
    #pragma omp parallel for simd aligned(AB, CD, result: 32) collapse(2)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i*N + j] = AB[i*N + j] + CD[i*N + j];
        }
    }
}

//========================================================================================================

void cpu_matrix_sub(float* restrict AB, float* restrict CD, float* restrict result, int N)
{
    #pragma omp parallel for simd aligned(AB, CD, result: 32) collapse(2)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i*N + j] = AB[i*N + j] - CD[i*N + j];
        }
    }
}

//========================================================================================================

void cpu_matrix_mull(float* restrict A, float* restrict B, float* restrict result, int N)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float temp = 0;

            #pragma omp simd
            for (int t = 0; t < N; t++)
                temp += A[i * N + t] * B[t * N + j];

            result[i*N + j] = temp;
        }
    }
}

//========================================================================================================

double cpu_calculation(float *A, float *B, float *C, float *D, int N, float* E, float* F)
{
    printf("Performing CPU calculations...\n");
    fflush(stdout);
    double t = get_wtime();
    
    float *AC, *BD, *AD, *BC;

    if (posix_memalign((void**)&AC, 32, N*N*sizeof(float)) != 0)
        {perror("Could not allocate aligned memory"); exit(EXIT_FAILURE);}
    if (posix_memalign((void**)&BD, 32, N*N*sizeof(float)) != 0)
        {perror("Could not allocate aligned memory"); exit(EXIT_FAILURE);}
    if (posix_memalign((void**)&AD, 32, N*N*sizeof(float)) != 0)
        {perror("Could not allocate aligned memory"); exit(EXIT_FAILURE);}
    if (posix_memalign((void**)&BC, 32, N*N*sizeof(float)) != 0)
        {perror("Could not allocate aligned memory"); exit(EXIT_FAILURE);}
        

    cpu_matrix_mull(A, C, AC, N);
    cpu_matrix_mull(B, D, BD, N);
    cpu_matrix_mull(A, D, AD, N);
    cpu_matrix_mull(B, C, BC, N);

    cpu_matrix_sub(AC, BD, E, N);
    cpu_matrix_add(AD, BC, F, N);

    free(AC);
    free(BD);
    free(AD);
    free(BC);

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

void initialize_matrix_(float** matrix, int N)
{
    srand(time(NULL) + 1000 * omp_get_thread_num());
    // printf("%d\n", omp_get_thread_num());

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
            {
                printf("\t");
            }
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