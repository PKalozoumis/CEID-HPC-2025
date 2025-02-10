#include <stdio.h>
#include <omp.h>
#include <cuda.h>
#include "cpu_calculations.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

//========================================================================================================

double gpu_time()
{
    static cudaEvent_t start, stop;

    if (start == NULL) //Start measuring
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        return 0;
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

        return milliseconds/1000.0;
    }
}

//========================================================================================================

// Calculations on GPU with one kernel
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

// Kernel multipling matrices
__global__ void multiply_matrix(float* R, float* M1, float* M2, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N)
    {
        for (int k = 0; k < N; k++)
            R[i * N + j] += M1[i * N + k] * M2[k * N + j];
    }
}

//========================================================================================================

// Kernel adding matrices
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

// Kernel subtracting matrices
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
    uint8_t MODE_CPU = 1<<0;
    uint8_t MODE_GPU_MULTIPLE = 1<<1; //Test multiple kernel implementation
    uint8_t MODE_GPU_SINGLE = 1<<2; //Test single kernel implementation
    uint8_t mode = MODE_CPU | MODE_GPU_SINGLE;

    if (argc == 1)
    {
        printf("Give matrix size\n");
        return 0;
    }

    if (argc >= 3)
    {
        mode = atoi(argv[2]);

        if ((mode > 7) || (mode < 1))
        {
            printf("Invalid mode. Must be in 1-7\n");
            exit(1);
        }
    }

    int N = atoi(argv[1]);

    //Open shared memory from Python driver program
    //===========================================================================
    int fd;
    double* shmem = NULL;
    
    if (argc == 4) //4th argument will be the shared memory name
    {
        fd = shm_open(argv[3], O_RDWR, 0);

        if (fd == -1)
            {perror("Could not open shared memory"); exit(1);}

        shmem = (double*)mmap(NULL, 6*sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

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

    int arraySize = N * N * sizeof(float);

    int blockSize = 16;
    dim3 block(blockSize, blockSize); // 16x16 = 256 threads per block. A multiple of 32, the warp size
    dim3 grid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    //Start CPU calculations
    //===========================================================================
    float *Ecpu, *Fcpu;

    posix_memalign((void**)&Ecpu, 32, arraySize);
    posix_memalign((void**)&Fcpu, 32, arraySize);

    if (mode & MODE_CPU)
    {
        t = cpu_calculation(A, B, C, D, N, Ecpu, Fcpu);
        if (shmem != NULL) shmem[1] = t;
    }

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

    if (mode & MODE_GPU_MULTIPLE)
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

    if ((mode & MODE_GPU_MULTIPLE) || (mode & MODE_GPU_SINGLE))
        printf("Performing GPU calculations...\n\n");

    //Start multiple kernel calculations
    if (mode & MODE_GPU_MULTIPLE)
    {
        printf("> Running version with multiple kernels...\n-----------------------------------------------------\n");

        gpu_time();

        multiply_matrix<<<grid, block>>>(devAC, devA, devC, N);
        multiply_matrix<<<grid, block>>>(devBD, devB, devD, N);
        multiply_matrix<<<grid, block>>>(devAD, devA, devD, N);
        multiply_matrix<<<grid, block>>>(devBC, devB, devC, N);

        sub_matrix<<<grid, block>>>(devE, devAC, devBD, N);
        add_matrix<<<grid, block>>>(devF, devAD, devBC, N);

        t = gpu_time();
        if (shmem != NULL) shmem[2] = t;

        cudaFree(devAC);
        cudaFree(devBD);
        cudaFree(devAD);
        cudaFree(devBC);

        if (mode & MODE_CPU)
        {
            //Compare results for multiple kernels
            cudaMemcpy(E, devE, arraySize, cudaMemcpyDeviceToHost);
            cudaMemcpy(F, devF, arraySize, cudaMemcpyDeviceToHost);
            t = matrix_comparison(Ecpu, Fcpu, E, F, N);
            if (shmem != NULL) shmem[3] = t;
        }
    }

    //Start single kernel calculations
    if (mode & MODE_GPU_SINGLE)
    {
        printf("> Running version with single kernel...\n-----------------------------------------------------\n");

        gpu_time();
        single_kernel_calculations<<<grid, block>>>(devA, devB, devC, devD, devE, devF, N);
        t = gpu_time();
        if (shmem != NULL) shmem[4] = t;

        if (mode & MODE_CPU)
        {
            //Compare results for single kernel
            cudaMemcpy(E, devE, arraySize, cudaMemcpyDeviceToHost);
            cudaMemcpy(F, devF, arraySize, cudaMemcpyDeviceToHost);
            t = matrix_comparison(Ecpu, Fcpu, E, F, N);
            if (shmem != NULL) shmem[5] = t;
        }
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

    if (shmem != NULL)
    {
        if (munmap(shmem, sizeof(double)) == -1)
            {perror("unmap failure"); exit(1);}
    }
}