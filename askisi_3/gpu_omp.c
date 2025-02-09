#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cpu_calculations.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

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

    //Open shared memory from Python driver program
    //===========================================================================
    int fd;
    double* shmem = NULL;
    
    if (argc == 3) //3rd argument will be the shared memory name
    {
        fd = shm_open(argv[2], O_RDWR, 0);

        if (fd == -1)
            {perror("Could not open shared memory"); exit(1);}

        shmem = (double*)mmap(NULL, 4*sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

        if (close(fd) == -1)
            {perror("Could not close file descriptor"); exit(1);}

        if (shmem == MAP_FAILED)
            {perror("mmap failure"); exit(1);}
    }

    // Initialize matrices in host
    //===========================================================================
    float *A, *B, *C, *D;
    double t = initialize_matrices(&A, &B, &C, &D, N);
    if (shmem != NULL) shmem[0] = t;

    int arraySize = N*N*sizeof(float);

    //Start CPU calculations
    //===========================================================================
    float* Ecpu = (float*)malloc(arraySize);
    float* Fcpu = (float*)malloc(arraySize);

    t = cpu_calculation(A, B, C, D, N, Ecpu, Fcpu);
    if (shmem != NULL) shmem[1] = t;

    //Start GPU caculations
    //===========================================================================
    float *E = (float *)calloc(N*N, sizeof(float));
    float *F = (float *)calloc(N*N, sizeof(float));

    printf("Performing GPU calculations...\n");
    fflush(stdout);

    t = get_wtime();
    
    #pragma omp target data map(to: A[0:N*N], B[0:N*N], C[0:N*N], D[0:N*N]) map(from: E[0:N*N], F[0:N*N]) 
    {
        #pragma omp target teams distribute parallel for collapse(2)
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

    t = get_wtime() - t;
    if (shmem != NULL) shmem[2] = t;

    printf("Total time for GPU calculations: %.03lfs\n\n", t);
    fflush(stdout);

    // Verify results
    //-------------------------------------------------------------------------------
    t = matrix_comparison(Ecpu,Fcpu,E,F,N);
    if (shmem != NULL) shmem[3] = t;

    // Free memory
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(F);
    free(Ecpu);
    free(Fcpu);

    if (shmem != NULL)
    {
        if (munmap(shmem, sizeof(double)) == -1)
            {perror("unmap failure"); exit(1);}
    }
}