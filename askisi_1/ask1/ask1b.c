#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

#define MAX 1000
#define MIN 1

int rank, size;
int prev = 0;

//============================================================================================

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

//============================================================================================

void MPI_Exscan_omp(int *in, int *out)
{
  static int prev = 0; // The offset in the file for the entire process
  int threadnum = omp_get_thread_num();

  // The first thread of the process must take the value that the last thread of the previous process passed on
  // In the case where this is the first process, there is nothing to take, and we assume prev = 0
  if (rank != 0 && threadnum == 0)
  {
    MPI_Recv(&prev, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

// Ensure that the value of prev was read by the fist thread of the process
#pragma omp barrier
  *out = prev;

  // All threads of the same process are aware of the other threads' counts
  // Each thread knows its inittial position in the file by summing up all the counts before it
  for (int i = 0; i < threadnum; i++)
    *out += in[i];

  // The last thread of the process must send its value to the next process
  // If I am the last process, then there is nothing to send
  if (rank != size - 1 && threadnum == omp_get_num_threads() - 1)
  {
    int next = *out + in[threadnum];
    MPI_Send(&next, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  }
}

//============================================================================================

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  omp_get_num_procs();

  int threads;

  if (rank == 0)
  {
    printf("Processes: %d\n", size);
    printf("Maximum number of processors: %d\n", omp_get_num_procs());
    do
    {
      printf("How many threads should each process have (Give a intenger number): ");
      scanf("%d", &threads);
    } while (threads < 0);

    MPI_Bcast(&threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Bcast(&threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  omp_set_num_threads(threads);
  omp_set_dynamic(0);

  /*int num[][4] = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16}};*/

  int indata[threads];
  srand(time(NULL) + rank + omp_get_thread_num());

  for (int i = 0; i < threads; i++)
  {
    indata[i] = (rand() % (MAX - MIN + 1)) + MIN;
  }

  int outdata = 0;
  int share = 0;

#pragma omp parallel firstprivate(outdata)
  {
    MPI_Exscan_omp(indata, &outdata);
    printf("Process: %d Indata: %d Result: %d\n", rank, indata[omp_get_thread_num()], outdata);
  }

  MPI_Finalize();

  return 0;
}