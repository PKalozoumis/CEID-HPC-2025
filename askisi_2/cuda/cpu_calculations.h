#ifndef CPU_CALCULATIONS_H
#define CPU_CALCULATIONS_H

#ifdef __cplusplus
extern "C" {
#endif

void initialize_matrix_(float** matrix, int N);
double initialize_matrices(float** A, float** B, float** C, float** D, int N);
double get_wtime();
void print_matrix(float *matrix, int N);
double cpu_calculation(float *A, float *B, float *C, float *D, int N, float* E, float* F);
double matrix_comparison(float* cpuE, float* cpuF, float* gpuE, float* gpuF, int N);
void cpu_test(float *A, float *B, float *C, float *D, int N, float* E, float* F);

#ifdef __cplusplus
}
#endif

#endif