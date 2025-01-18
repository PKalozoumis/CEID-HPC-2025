#include <stdio.h>
#include <omp.h>
#include <cuda.h>

void initialize_matrix(float** matrix, int N)
{
    srand(time(NULL) + 1000*omp_get_thread_num());
    //printf("%d\n", omp_get_thread_num());

    *matrix = (float*)malloc(N*N*sizeof(float));

    for (int i = 0; i < N*N; i++)
    {
        (*matrix)[i] = (rand() / (float)RAND_MAX) * 10;
    }
}

void print_matrix(float* matrix, int N)
{

    printf("==================================================================\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.02f", matrix[i*N + j]);

            if (j < N-1)
            {
                printf("\t");
            }
        }
        printf("\n");
    }
}


__global__ void multiply_matrix(float* R, float* M1, float* M2, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("(%d, %d)\n", i, j);

    if (i < N && j < N)
    {
        for (int k = 0; k < N; k++)
            R[i*N + j] += M1[i*N + k]*M2[k*N + j];
    }
}

int main(int argc, char** argv)
{
    if (argc == 1)
    {
        printf("Give matrix size\n");
        return 0;
    }

    int N = atoi(argv[1]);

    //Initialize matrices in host
    //===========================================================================

    float *A, *B, *C, *D;

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

    /*
    printf("\nA\n");
    print_matrix(A, N);
    printf("\nB\n");
    print_matrix(B, N);
    printf("\nC\n");
    print_matrix(C, N);
    printf("\nD\n");
    print_matrix(D, N);*/

    float *devA, *devB, *devC, *devD, *devR;

    int arraySize = N*N*sizeof(float);

    cudaMalloc(&devA, arraySize);
    cudaMalloc(&devB, arraySize);
    cudaMalloc(&devC, arraySize);
    cudaMalloc(&devD, arraySize);
    cudaMalloc(&devR, arraySize);
    cudaMemset(devR, 0, arraySize);

    cudaMemcpy(devA, A, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, C, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devD, D, arraySize, cudaMemcpyHostToDevice);

    print_matrix(A, N);
    print_matrix(B, N);

    int blockSize = 16;
    dim3 block(blockSize, blockSize); //16x16 = 256 threads per block. A multiple of 32, the warp size
    dim3 grid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    multiply_matrix<<<grid, block>>>(devR, devA, devB, N);

    float* result = (float*)malloc(arraySize);
    cudaMemcpy(result, devR, arraySize, cudaMemcpyDeviceToHost);

    print_matrix(result, N);
}