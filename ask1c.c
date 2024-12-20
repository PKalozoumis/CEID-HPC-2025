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

//================================================================

int main(int argc, char *argv[])
{

  int errFlag = 0;
  int *flags;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(argc != 3){
  
    if (rank == 0)
    {
      if (argc == 1)
      {
        printf("Give num of threads.\n");
      }
      else if (argc == 2)
      {
        printf("Give size of data.\n");
      }
    }
      
    return 0;
  }

  int numThreads = atoi(argv[1]);
  int N = atoi(argv[2]);

  int arraySize = N*N*N;

  if(numThreads <1){
    if (rank == 0)
      printf("Give positive num of threads.\n");
    return 0;
  }

  if (N < 1)
  {
    if (rank == 0)
      printf("Size of data must be greater than zero.\n");

    return 0;
  }

  if(numThreads>omp_get_num_procs()){
    if (rank == 0)
      printf("The maximum threads are %d. Using %d threads.\n",omp_get_num_procs(),omp_get_num_procs());
      
    numThreads = omp_get_num_procs();
  }

  uint8_t totalThreads = numThreads * size;

  omp_set_num_threads(numThreads);
  omp_set_dynamic(0);

  int outdata = 0;
  int writeSize[numThreads];

  for (int i = 0; i < numThreads; i++)
    writeSize[i] = arraySize;

  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, "file_erotima_c.bin", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);
  MPI_File_set_size(file, 0);

  MPI_Offset base;
  MPI_File_get_position(file, &base);

  // Write file header (size: 33 bytes)
  //================================================================
  if (rank == 0)
  {
    MPI_File_write_at(file, base, &totalThreads, 1, MPI_BYTE, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Offset offset = sizeof(uint8_t) + numThreads * sizeof(int) * rank;
  MPI_File_write_at(file, offset, &writeSize, numThreads, MPI_INT, MPI_STATUS_IGNORE);
  base += 1 + totalThreads * sizeof(int);

#pragma omp parallel firstprivate(outdata)
  {
    float* data = (float *)malloc(arraySize*sizeof(float));
    srand(time(NULL) + rank + omp_get_thread_num());

    // Initialize the matrix
    for (int i = 0; i < arraySize; i++)
    {
      data[i] = (rand() / (float)RAND_MAX) * 1000;
    }

    MPI_Exscan_omp(writeSize, &outdata);

    // Start writing to file
    //==============================================================
    #pragma omp barrier

    #pragma omp single
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Offset offset = base + outdata * sizeof(float);
    // printf("offset %ld\n",offset);
    MPI_File_write_at(file, offset, data, writeSize[omp_get_thread_num()], MPI_FLOAT, MPI_STATUS_IGNORE);

    //==============================================================

    #pragma omp barrier

    #pragma omp single
    MPI_Barrier(MPI_COMM_WORLD);

    float* buffer = (float*)malloc(arraySize*sizeof(float));
    MPI_File_read_at(file, offset, buffer, arraySize, MPI_FLOAT, MPI_STATUS_IGNORE);

    for (int i = 0; i < arraySize; i++)
    {
      if (buffer[i] != data[i])
      {
        printf("%lf %lf\n", buffer[i], data[i]);
        errFlag = 1;
        break;
      }
    }

  }

  
  if (rank == 0)
  {
    flags = (int *)malloc(size * sizeof(int));
  }

  MPI_Gather(&errFlag, 1, MPI_INT, flags, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    int success = 1;
    for (int i = 0; i < size; i++)
    {
      if (flags[i] == 1)
      {
        printf("Verification failed.\n");
        success = 0;
        break;
      }
    }

    free(flags);

    if (success)
    {
      printf("Successful verification.\n");
    }
  }

  MPI_File_close(&file);
  MPI_Finalize();

  return 0;
}