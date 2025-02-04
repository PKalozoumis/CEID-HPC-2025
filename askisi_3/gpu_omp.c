#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cpu_calculations.h"

//========================================================================================================

void multiply_matrix(float *R, float *M1, float *M2, int N)
{
    #pragma omp target data use_device_addr(R, M1, M2)
    #pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < N; k++)
                sum += M1[i * N + k] * M2[k * N + j];

            R[i * N + j] = sum;
        }
    }
}

//========================================================================================================

void add_matrix(float *R, float *M1, float *M2, int N)
{
    #pragma omp target data use_device_addr(R, M1, M2)
    #pragma omp target teams distribute parallel for simd collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            R[i*N + j] = M1[i*N + j] + M2[i*N + j];
}

//========================================================================================================

void sub_matrix(float *R, float *M1, float *M2, int N)
{
    #pragma omp target data use_device_addr(R, M1, M2)
    #pragma omp target teams distribute parallel for simd collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            R[i*N + j] = M1[i*N + j] - M2[i*N + j];

}

//========================================================================================================

int main(int argc, char **argv)
{
    printf("Detected devices: %d\n\n", omp_get_num_devices());

    if (argc == 1)
    {
        printf("Give matrix size\n");
        return 0;
    }

    int N = atoi(argv[1]);

    // Initialize matrices in host
    //===========================================================================

    float *A, *B, *C, *D;

    printf("Initializing matrices in host...\n");
    fflush(stdout);
    double t = get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            initialize_matrix(&A, N);

            #pragma omp task
            initialize_matrix(&B, N);

            #pragma omp task
            initialize_matrix(&C, N);

            #pragma omp task
            initialize_matrix(&D, N);
        }
    }

    printf("Total time for parallel initialization: %.03lfs\n\n", get_wtime()-t);

    //===========================================================================

    int arraySize = N*N*sizeof(float);;

    float *E = (float *)malloc(arraySize);
    float *F = (float *)malloc(arraySize);

    printf("Performing GPU calculations...\n");
    fflush(stdout);

    t = get_wtime();
    
        #pragma omp target teams distribute parallel for collapse(2) map(to: A[0:N*N], B[0:N*N]) map(from: C)
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            { 
               
                float sum = 0.0f;
                //float resAC = 0.0f, resBD = 0.0f, resAD = 0.0f, resBC = 0.0f;

                //#pragma omp simd reduction(+:resAC, resBD, resAD, resBC)
                
                for (int k = 0; k < N; k++)
                {
                     sum += A[i][k] * B[k][j];
                    //resBD += B[i * N + k] * D[k * N + j];
                    //resAD += A[i * N + k] * D[k * N + j];
                    //resBC += B[i * N + k] * C[k * N + j];
                }
                C[i][j] = sum;

                //E[i*N + j] = resAC - resBD;
                //F[i*N + j] = resAD + resBC; 
            }
        }
    

    printf("Total time for GPU calculations: %.03lfs\n\n", get_wtime() - t);
    fflush(stdout);

    //print_matrix(E, N);

    // Verify results
    //-------------------------------------------------------------------------------
    float* Echeck = (float*)malloc(arraySize);
    float* Fcheck = (float*)malloc(arraySize);

    printf("Performing CPU calculations...\n");
    fflush(stdout);
    t = get_wtime();
    cpu_calculation(A, B, C, D, N, Echeck, Fcheck);
    printf("Total time for CPU calculations: %.03lfs\n\n", get_wtime()-t);
    fflush(stdout);
    
    printf("Comparing results... ");
    fflush(stdout);
    t = get_wtime();
    matrix_comparison(Echeck,Fcheck,E,F,N);
    printf("Total time for result comparison in CPU: %.03lfs\n", get_wtime()-t);
    fflush(stdout);

    // Free memory
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(F);

    free(Echeck);
    free(Fcheck);
}