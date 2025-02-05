#include <stdio.h>
#include <omp.h>
#include <cuda.h>
#include "cpu_calculations.h"

//========================================================================================================

void gpu_time()
{
    static cudaEvent_t start, stop;

    if (start == NULL) //Start measuring
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
    }
    else //Stop measuring
    {
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Total time for GPU calculations: %.03lfs\n\n", milliseconds/1000);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        start = stop = NULL;
    }
}

//========================================================================================================

__global__ void single_kernel_calculations(float* A, float* B, float* C, float* D, float* E, float* F, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N)
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

//========================================================================================================

__global__ void multiply_matrix(float* R, float* M1, float* M2, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("(%d, %d)\n", i, j);

    if (i < N && j < N)
    {
        for (int k = 0; k < N; k++)
            R[i * N + j] += M1[i * N + k] * M2[k * N + j];
    }
}

//========================================================================================================

__global__ void add_matrix(float* R, float* M1, float* M2, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = i * N + j;

    if (i < N && j < N)
    {
        R[pos] = M1[pos] + M2[pos];
    }
}

//========================================================================================================

__global__ void sub_matrix(float *R, float *M1, float *M2, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = i * N + j;

    if (i < N && j < N)
    {
        R[pos] = M1[pos] - M2[pos];
    }
}

//========================================================================================================

int main(int argc, char **argv)
{
    int single_kernel_mode = 0;

    if (argc == 1)
    {
        printf("Give matrix size\n");
        return 0;
    }

    if (argc == 3)
    {
        //Mode 0 -> Multiple kernels
        //Mode 1 -> Single kernel
        //Mode 2 -> Run both

        if (strcmp(argv[2], "0") != 0 && strcmp(argv[2], "1") != 0 && strcmp(argv[2], "2") != 0)
        {
            printf("Invalid mode %s\n", argv[2]);
            return 0;
        }

        single_kernel_mode = atoi(argv[2]);
    }

    int N = atoi(argv[1]);

    // Initialize matrices in host
    //===========================================================================
    float *A, *B, *C, *D;
    initialize_matrices(&A, &B, &C, &D, N);

    int arraySize = N * N * sizeof(float);

    int blockSize = 16;
    dim3 block(blockSize, blockSize); // 16x16 = 256 threads per block. A multiple of 32, the warp size
    dim3 grid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    //Start CPU calculations
    //===========================================================================
    float *Ecpu, *Fcpu;

    posix_memalign((void**)&Ecpu, 32, arraySize);
    posix_memalign((void**)&Fcpu, 32, arraySize);

    cpu_calculation(A, B, C, D, N, Ecpu, Fcpu);

    //Initialize GPU memory
    //===========================================================================
    float *devA, *devB, *devC, *devD, *devE, *devF, *devAC, *devBD, *devAD, *devBC;

    cudaMalloc(&devA, arraySize);
    cudaMalloc(&devB, arraySize);
    cudaMalloc(&devC, arraySize);
    cudaMalloc(&devD, arraySize);
    cudaMalloc(&devE, arraySize);
    cudaMalloc(&devF, arraySize);

    cudaMemcpy(devA, A, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, C, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devD, D, arraySize, cudaMemcpyHostToDevice);

    if (single_kernel_mode != 1)
    {
        cudaMalloc(&devAC, arraySize);
        cudaMalloc(&devBD, arraySize);
        cudaMalloc(&devAD, arraySize);
        cudaMalloc(&devBC, arraySize);

        cudaMemset(devAC, 0, arraySize);
        cudaMemset(devBD, 0, arraySize);
        cudaMemset(devAD, 0, arraySize);
        cudaMemset(devBC, 0, arraySize);
    }

    //Start GPU calculations
    //===========================================================================
    float* E = (float *)malloc(arraySize);
    float* F = (float *)malloc(arraySize);

    printf("Performing GPU calculations...\n\n");

    //Start multiple kernel calculations
    if (single_kernel_mode != 1)
    {
        printf("> Running version with multiple kernels...\n-----------------------------------------------------\n");

        gpu_time();

        multiply_matrix<<<grid, block>>>(devAC, devA, devC, N);
        multiply_matrix<<<grid, block>>>(devBD, devB, devD, N);
        multiply_matrix<<<grid, block>>>(devAD, devA, devD, N);
        multiply_matrix<<<grid, block>>>(devBC, devB, devC, N);

        sub_matrix<<<grid, block>>>(devE, devAC, devBD, N);
        add_matrix<<<grid, block>>>(devF, devAD, devBC, N);

        gpu_time();

        cudaFree(devAC);
        cudaFree(devBD);
        cudaFree(devAD);
        cudaFree(devBC);

        //Compare results for multiple kernels
        cudaMemcpy(E, devE, arraySize, cudaMemcpyDeviceToHost);
        cudaMemcpy(F, devF, arraySize, cudaMemcpyDeviceToHost);
        matrix_comparison(Ecpu, Fcpu, E, F, N);
    }

    //Start single kernel calculations
    if (single_kernel_mode != 0)
    {
        printf("> Running version with single kernel...\n-----------------------------------------------------\n");

        gpu_time();
        single_kernel_calculations<<<grid, block>>>(devA, devB, devC, devD, devE, devF, N);
        gpu_time();

        //Compare results for single kernel
        cudaMemcpy(E, devE, arraySize, cudaMemcpyDeviceToHost);
        cudaMemcpy(F, devF, arraySize, cudaMemcpyDeviceToHost);
        matrix_comparison(Ecpu, Fcpu, E, F, N);
    }

    //Free memory
    //========================================================================================
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFree(devD);

    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(F);
    free(Ecpu);
    free(Fcpu);
}