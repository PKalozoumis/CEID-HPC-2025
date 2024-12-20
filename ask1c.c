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

  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, "file.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
  
#pragma omp parallel firstprivate(outdata)
  {
    float data[N][N][N];
    srand(time(NULL) + rank + omp_get_thread_num());

    // Initialize the matrix
    for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
      {
        for (int k = 0; k < N; k++)
        {
          data[i][j][k] = (rand() % 1000) / (float)RAND_MAX;
        }
      }
    }

    MPI_Exscan_omp(writeSize, &outdata);
    printf("outdata: %d\n", outdata);


    //Start writing to file
    //==============================================================
    #pragma omp barrier

    #pragma omp single
    MPI_Barrier(MPI_COMM_WORLD);

    
    MPI_Offset offset = outdata;
    printf("offset %ld\n",offset);
    //MPI_File_write_at(file, offset, &data, writeSize[omp_get_thread_num()], MPI_FLOAT, MPI_STATUS_IGNORE);

    int write = rank*4 + omp_get_thread_num();
    printf("write %d\n",write);
    MPI_File_write_at(file, offset, &write, 1, MPI_INT, MPI_STATUS_IGNORE);
  }


  MPI_File_close(&file);
  MPI_Finalize();

  return 0;
}