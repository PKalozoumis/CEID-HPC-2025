#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../askisi_2/cuda/cpu_calculations.h"


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
    initialize_matrices(&A, &B, &C, &D, N);

    //===========================================================================

    int arraySize = N*N*sizeof(float);;

    float *E = (float *)malloc(arraySize);
    float *F = (float *)malloc(arraySize);

    printf("Performing GPU calculations...\n");
    fflush(stdout);

    double t = get_wtime();
    
        #pragma omp target teams distribute parallel for collapse(2) map(to: A[0:N*N], B[0:N*N], C[0:N*N], D[0:N*N]) map(from: E[0:N*N],F[0:N*N])
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
    

    printf("Total time for GPU calculations: %.03lfs\n\n", get_wtime() - t);
    fflush(stdout);

    //print_matrix(E, N);

    // Verify results
    //-------------------------------------------------------------------------------
    float* Ecpu = (float*)malloc(arraySize);
    float* Fcpu = (float*)malloc(arraySize);

    cpu_calculation(A, B, C, D, N, Ecpu, Fcpu);
    
    matrix_comparison(Ecpu,Fcpu,E,F,N);

    // Free memory
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(F);

    free(Ecpu);
    free(Fcpu);
}