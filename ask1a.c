#include <mpi.h>
#include <stdio.h>

int rank, size;

void MPI_Exscan_pt2pt(int* in, int* out, int rank)
{
    int prev = 0;

    if(rank!=0){
        MPI_Recv(&prev, 1, MPI_INT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    *out = prev;

    if(rank!=size-1){
        int next = prev + *in;
        MPI_Send(&next, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
        printf("Processes: %d\n", size);

    int num[] = {10, 20, 30, 40};
    
    int indata = num[rank];
    int outdata = 0;

    MPI_Exscan_pt2pt(&indata, &outdata, rank);

    printf("Process: %d Indata: %d Result: %d\n", rank, indata, outdata);

    MPI_Finalize();

    return 0;
}