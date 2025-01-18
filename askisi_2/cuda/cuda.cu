#include <stdio.h>
#include <omp.h>
#include <cuda.h>

void initialize_matrix(float** matrix, int N)
{
    srand(time(NULL) + 1000*omp_get_thread_num());
    printf("%d\n", omp_get_thread_num());

    *matrix = (float*)malloc(N*N*sizeof(float));

    for (int i = 0; i < N*N; i++)
    {
        (*matrix)[i] = (rand() / (float)RAND_MAX) * 1000;
    }
}

void print_matrix(float* matrix, int N)
{
    for (int i = 0; i < N*N; i++)
    {
        printf("%f\t", matrix[i]);

        if ((i > 0 && i % N == 0))
            printf("\n");
    }
}


__global__ void multiply_matrix(float* R, float* M1, float* M2, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    


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

    float *devA, *devB, *devC, *devD;

    int arraySize = N*N*sizeof(float);

    cudaMalloc(&devA, arraySize);
    cudaMalloc(&devB, arraySize);
    cudaMalloc(&devC, arraySize);
    cudaMalloc(&devD, arraySize);

    cudaMemcpy(devA, A, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, C, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devD, D, arraySize, cudaMemcpyHostToDevice);
}