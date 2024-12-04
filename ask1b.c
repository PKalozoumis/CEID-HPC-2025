#include <mpi.h>
#include <stdio.h>
#include <omp.h>

int rank, size;
int prev = 0;

void MPI_Exscan_pt2pt(int *in, int *out)
{
    int prev = 0;

    if (rank != 0)
    {
        MPI_Recv(&prev, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    *out = prev;

    if (rank != size - 1)
    {
        int next = prev + *in;
        MPI_Send(&next, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
}

void MPI_Exscan_omp(int *in, int *out)
{

    int threadnum = omp_get_thread_num();

    if (rank != 0 && threadnum == 0)
    {
        MPI_Recv(&prev, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

#pragma omp barrier

    *out = prev;

    for (int i = 0; i < threadnum; i++)
        *out += in[i];

    if (rank != size - 1 && threadnum == omp_get_num_threads() - 1)
    {
        int next = *out + in[threadnum];
        // printf("Process %d sends %d from thread %d\n", rank, next, threadnum);
        // printf("in: %d diergasia: %d out: %d\n",next, rank, *out);
        MPI_Send(&next, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
    /*
    else if (rank == size-1 && threadnum==omp_get_num_threads()-1)
    {
        int next = *out + in[threadnum];
        printf("Process %d sends %d from thread %d\n", rank, next, threadnum);
    }
    */
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    omp_get_num_procs();

    if (rank == 0)
    {
        // printf("Processes: %d\n", size);
        // printf("Processors: %d\n", omp_get_num_procs());
    }

    omp_set_num_threads(4);
    omp_set_dynamic(0);

    int num[][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}};

    int outdata = 0;
    int share = 0;

#pragma omp parallel firstprivate(outdata)
    {
        MPI_Exscan_omp(num[rank], &outdata);
        printf("outdata: %d\n", outdata);
    }

    MPI_Finalize();

    return 0;
}