#ifndef CPU_CALCULATIONS_H
#define CPU_CALCULATIONS_H

#ifdef __cplusplus
extern "C" {
#endif

void initialize_matrix_(float** matrix, int N);
void initialize_matrices(float** A, float** B, float** C, float** D, int N);
double get_wtime();
void print_matrix(float *matrix, int N);
void cpu_matrix_add(float* AB, float* CD, float* result, int N);
void cpu_matrix_sub(float* AB, float* CD, float* result, int N);
void cpu_matrix_mull(float *A, float *B, float* result, int N);
void cpu_calculation(float *A, float *B, float *C, float *D, int N, float* E, float* F);
void matrix_comparison(float* cpuE, float* cpuF, float* gpuE, float* gpuF, int N);

#ifdef __cplusplus
}
#endif

#endif