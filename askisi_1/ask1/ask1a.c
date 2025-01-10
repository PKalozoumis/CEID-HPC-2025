#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<unistd.h>

#define MAX 1000
#define MIN 1

int rank, size;

//============================================================================================

void MPI_Exscan_pt2pt(int *in, int *out, int rank)
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

//============================================================================================

void validation(int *indata, int outdata_check)
{

  int outdata = 0;
  int error = 0;

  MPI_Exscan(indata, &outdata, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (outdata_check != outdata)
  {
    error = 1;
  }

  int error_gather[size];

  MPI_Gather(&error, 1, MPI_INT, error_gather, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {

    int validation = 0;

    for (int i = 0; i < size; i++)
    {
      if (error_gather[i] == 1)
      {
        validation = 1;
      }
    }

    if (validation == 1)
    {
      printf("-> Unsuccessful verification\n");
    }
    else
    {
      printf("-> Successful verification\n");
    }
  }
}

//============================================================================================

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
    printf("Processes: %d\n", size);

  // int num[] = {10, 20, 30, 40};

  srand(time(NULL) + rank);
  int indata = (rand() % (MAX - MIN + 1)) + MIN;
  int outdata = 0;

  MPI_Exscan_pt2pt(&indata, &outdata, rank);

  printf("Process: %d Indata: %d Result: %d\n", rank, indata, outdata);
  
  sleep(0.1);
  MPI_Barrier(MPI_COMM_WORLD);

  validation(&indata, outdata);

  MPI_Finalize();

  return 0;
}