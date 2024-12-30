#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numThreads = omp_get_num_threads();

    omp_set_num_threads(numThreads);
    omp_set_dynamic(0);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "file.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_Offset base;
	MPI_File_get_position(file, &base);

    //Read number of threads
    uint8_t totalThreads = 
}