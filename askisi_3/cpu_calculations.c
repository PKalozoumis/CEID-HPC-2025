#include "cpu_calculations.h"
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

//========================================================================================================

void cpu_matrix_add(float* AB, float* CD, float* result, int N)
{
    #pragma omp parallel loop
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i*N + j] = AB[i*N + j] + CD[i*N + j];
        }
    }
}

//========================================================================================================

void cpu_matrix_sub(float* AB, float* CD, float* result, int N)
{
    #pragma omp parallel loop
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i*N + j] = AB[i*N + j] - CD[i*N + j];
        }
    }
}

//========================================================================================================

void cpu_matrix_mull(float *A, float *B, float* result, int N)
{
    #pragma omp parallel loop
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i*N + j] = 0;
            for (int t = 0; t < N; t++)
            {
                result[i*N + j] += A[i * N + t] * B[t * N + j];
            }
        }
    }
}

//========================================================================================================

void cpu_calculation(float *A, float *B, float *C, float *D, int N, float* E, float* F)
{
    printf("Performing CPU calculations...\n");
    double t = get_wtime();

    float *AC, *BD, *AD, *BC;

    AC = (float*)malloc(N*N*sizeof(float));
    BD = (float*)malloc(N*N*sizeof(float));
    AD = (float*)malloc(N*N*sizeof(float));
    BC = (float*)malloc(N*N*sizeof(float));

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

    printf("Total time for CPU calculations: %.03lfs\n\n", get_wtime()-t);
}

//========================================================================================================

void matrix_comparison(float* cpuE, float* cpuF, float* gpuE, float* gpuF, int N){
    printf("Comparing results... ");
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
    }else{
        printf("Comparison failed\n");
    }

    printf("Total time for result comparison in CPU: %.03lfs\n\n", get_wtime()-t);
}

//========================================================================================================

void initialize_matrix(float** matrix, int N)
{
    srand(time(NULL) + 1000 * omp_get_thread_num());
    // printf("%d\n", omp_get_thread_num());

    *matrix = (float *)malloc(N * N * sizeof(float));

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
}