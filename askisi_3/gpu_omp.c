#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

//========================================================================================================

double get_wtime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec * 1e-6;
}

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

//========================================================================================================

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

void free_memMatrix(float **matrix, int n){
    for(int i=0;i<n;i++){
        free(matrix[i]);
    }

    free(matrix);
}

//========================================================================================================

void cpu_calculation(float *A, float *B, float *C, float *D, int n, float** E, float** F)
{

    printf("Initializing in the host\n");
    fflush(stdout);

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
    
    printf("Starting CPU calculations\n");
    fflush(stdout);

    double t = get_wtime();

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

    printf("Time: %lf\n", get_wtime() - t);

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
    double tolerance = 1e-1;
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

    printf("Initializing in the host\n");
    fflush(stdout);

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

    //===========================================================================

    int arraySize = N*N;

    float *E = (float *)malloc(arraySize * sizeof(float));
    float *F = (float *)malloc(arraySize * sizeof(float));
    float *AC = (float*)malloc(arraySize * sizeof(float));
    float *BD = (float*)malloc(arraySize * sizeof(float));
    float *AD = (float*)malloc(arraySize * sizeof(float));
    float *BC = (float*)malloc(arraySize * sizeof(float));

    double t = get_wtime();

    printf("Starting GPU calculations\n");
    fflush(stdout);
    
    #pragma omp target data map(to: A[0:N*N],B[0:N*N],C[0:N*N],D[0:N*N]) map(from: E[0:N*N],F[0:N*N]) map(alloc: AC[0:N*N], BD[0:N*N], AD[0:N*N], BC[0:N*N])
    {
        multiply_matrix(AC, A, C, N);
        multiply_matrix(BD, B, D, N);
        multiply_matrix(AD, A, D, N);
        multiply_matrix(BC, B, C, N);

        sub_matrix(E, AC, BD, N);
        add_matrix(F, AD, BC, N);
    }

    printf("Time: %lf\n", get_wtime() - t);

    //print_matrix(E, N);

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
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(F);
    free(AC);
    free(BD);
    free(AD);
    free(BC);

    free_memMatrix(Echeck, N);
    free_memMatrix(Fcheck, N);
}