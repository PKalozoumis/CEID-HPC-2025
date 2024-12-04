#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int rank, size;
int prev = 0;

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

    MPI_Send(&next, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  }
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

  int N = 2;

  int outdata = 0;
  int writeSize[4];

  for (int i = 0; i < 4; i++)
    writeSize[i] = N*N*N;
  
#pragma omp parallel firstprivate(outdata)
  {
    int data[N][N][N];
    srand(time(NULL) + rank + omp_get_thread_num());

    // Initialize the matrix
    for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
      {
        for (int k = 0; k < N; k++)
        {
          data[i][j][k] = rand() % 1000;
        }
      }
    }

    

    MPI_Exscan_omp(writeSize, &outdata);
    printf("outdata: %d\n", outdata);
  }

  MPI_Finalize();

  return 0;
}