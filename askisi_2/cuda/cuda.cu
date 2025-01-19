#include <stdio.h>
#include <omp.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>

//========================================================================================================

void alloc_2d(float*** A, int N)
{
    *A = (float **)malloc(N * sizeof(float *));

    for (int i = 0; i < N; i++)
    {
        (*A)[i] = (float*)malloc(N * sizeof(float));
    }
}

//========================================================================================================

void initialize_matrix(float **matrix, int N)
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

void print_matrix_cpu(float** matrix, int N)
{
    printf("==================================================================\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.02f", matrix[i][j]);

            if (j < N - 1)
            {
                printf("\t");
            }
        }
        printf("\n");
    }
}

//========================================================================================================

__global__ void multiply_matrix(float *R, float *M1, float *M2, int N)
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

__global__ void add_matrix(float *R, float *M1, float *M2, int N)
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

void cpu_matrix_add(float **AB, float **CD, float **result ,int n)
{

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[i][j] = AB[i][j] + CD[i][j];
        }
    }
}

//========================================================================================================

void cpu_matrix_sub(float **AB, float **CD, float **result ,int n)
{

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[i][j] = AB[i][j] - CD[i][j];
        }
    }
}

//========================================================================================================

void cpu_matrix_mull(float *A, float *B, float** result, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[i][j] = 0;
            for (int t = 0; t < n; t++)
            {
                result[i][j] += A[i * n + t] * B[t * n + j];
            }
        }
    }
}

//========================================================================================================

void free_memMatrix(float **matrix, int n){
    for(int i=0;i<n;i++){
        free(matrix[i]);
    }

    free(matrix);
}

//========================================================================================================

void cpu_calculation(float *A, float *B, float *C, float *D, int n, float** E, float** F)
{

    float **AC, **BD, **AD, **BC;

    AC = (float **)malloc(n * sizeof(float *));
    BD = (float **)malloc(n * sizeof(float *));
    AD = (float **)malloc(n * sizeof(float *));
    BC = (float **)malloc(n * sizeof(float *));

    for (int i = 0; i < n; i++)
    {
        AC[i] = (float *)malloc(n * sizeof(float));
        BD[i] = (float *)malloc(n * sizeof(float));
        AD[i] = (float *)malloc(n * sizeof(float));
        BC[i] = (float *)malloc(n * sizeof(float));
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            cpu_matrix_mull(A, C, AC, n);

            #pragma omp task
            cpu_matrix_mull(B, D, BD, n);

            #pragma omp task
            cpu_matrix_mull(A, D, AD, n);

            #pragma omp task
            cpu_matrix_mull(B, C, BC, n);
        }
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            cpu_matrix_sub(AC, BD, E, n);

            #pragma omp task
            cpu_matrix_add(AD, BC, F, n);
        }
    }

     #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            free_memMatrix(AC, n);

            #pragma omp task
            free_memMatrix(BD, n);
            
            #pragma omp task
            free_memMatrix(AD, n);

            #pragma omp task
            free_memMatrix(BC, n);
        }
    }
}

//========================================================================================================

void matrix_comparison(float **cpuE, float **cpuF,float* gpuE, float* gpuF, int n){
    
    int error=0;
    double tolerance = 1e-0;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if (fabs(cpuE[i][j] - gpuE[i * n + j]) > tolerance || 
                fabs(cpuF[i][j] - gpuF[i * n + j]) > tolerance){
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
    
}

//========================================================================================================

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        printf("Give matrix size\n");
        return 0;
    }

    int N = atoi(argv[1]);

    // Initialize matrices in host
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

    float *devA, *devB, *devC, *devD, *devAC, *devBD, *devAD, *devBC;

    int arraySize = N * N * sizeof(float);

    cudaMalloc(&devA, arraySize);
    cudaMalloc(&devB, arraySize);
    cudaMalloc(&devC, arraySize);
    cudaMalloc(&devD, arraySize);
    cudaMalloc(&devAC, arraySize);
    cudaMalloc(&devBD, arraySize);
    cudaMalloc(&devAD, arraySize);
    cudaMalloc(&devBC, arraySize);

    cudaMemset(devAC, 0, arraySize);
    cudaMemset(devBD, 0, arraySize);
    cudaMemset(devAD, 0, arraySize);
    cudaMemset(devBC, 0, arraySize);

    cudaMemcpy(devA, A, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, C, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devD, D, arraySize, cudaMemcpyHostToDevice);

    int blockSize = 16;
    dim3 block(blockSize, blockSize); // 16x16 = 256 threads per block. A multiple of 32, the warp size
    dim3 grid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);

    multiply_matrix<<<grid, block>>>(devAC, devA, devC, N);
    multiply_matrix<<<grid, block>>>(devBD, devB, devD, N);
    multiply_matrix<<<grid, block>>>(devAD, devA, devD, N);
    multiply_matrix<<<grid, block>>>(devBC, devB, devC, N);

    sub_matrix<<<grid, block>>>(devAC, devAC, devBD, N);
    add_matrix<<<grid, block>>>(devAD, devAD, devBC, N);


    cudaDeviceSynchronize(); //!!!!!!!!!!!!!!!!!!!

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Total time for GPU calculations: %.03lfs\n", milliseconds/1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Get back final results

    float *E = (float *)malloc(arraySize);
    float *F = (float *)malloc(arraySize);

    cudaMemcpy(E, devAC, arraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(F, devAD, arraySize, cudaMemcpyDeviceToHost);

    //print_matrix(E, N);
    //print_matrix(F, N);

    // Verify results
    //-------------------------------------------------------------------------------
    float **Echeck, **Fcheck;
    alloc_2d(&Echeck, N);
    alloc_2d(&Fcheck, N);

    cpu_calculation(A, B, C, D, N, Echeck, Fcheck);

    //print_matrix_cpu(Echeck, N);
    //print_matrix_cpu(Fcheck, N);

    matrix_comparison(Echeck,Fcheck,E,F,N);

    // Free memory

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFree(devD);
    cudaFree(devAC);
    cudaFree(devBD);
    cudaFree(devAD);
    cudaFree(devBC);

    free(A);
    free(B);
    free(C);
    free(D);

    free_memMatrix(Echeck, N);
    free_memMatrix(Fcheck, N);
}