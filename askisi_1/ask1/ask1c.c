#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int rank, size;

//============================================================================================

void MPI_Exscan_omp(int in, int *out)
{
    //These static variables are allocated for the entire process
    static int prev = 0; // The offset in the file for the entire process
    static int *process_data;
    
    int threadnum = omp_get_thread_num();

    #pragma omp single
    process_data = (int *)malloc(omp_get_num_threads() * sizeof(int));

    process_data[threadnum] = in;

    #pragma omp barrier

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
        *out += process_data[i];

    // The last thread of the process must send its value to the next process
    // If I am the last process, then there is nothing to send
    if (rank != size - 1 && threadnum == omp_get_num_threads() - 1)
    {
        int next = *out + process_data[threadnum];
        MPI_Send(&next, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
}

//============================================================================================

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	// Argument parsing
	//============================================================================================
	if (argc != 3)
	{
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

	int arraySize = N * N * N;

	if (numThreads < 1)
	{
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

	if (numThreads > omp_get_num_procs())
	{
		if (rank == 0)
			printf("-> The maximum threads are %d. Using %d threads.\n", omp_get_num_procs(), omp_get_num_procs());

		numThreads = omp_get_num_procs();
	}

	//============================================================================================

	uint8_t totalThreads = numThreads * size; // Threads across all processes

	omp_set_num_threads(numThreads);
	omp_set_dynamic(0);

	int threadPosition = 0; //Position where each thread writes in the file
	int writeSize[numThreads]; //How much each thread writes

	for (int i = 0; i < numThreads; i++)
		writeSize[i] = arraySize;

	MPI_File file;
	MPI_File_open(MPI_COMM_WORLD, "file_erotima_c.bin", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);
	MPI_File_set_size(file, 0);

	MPI_Offset base;
	MPI_File_get_position(file, &base);

	// Write file header
	//================================================================
	if (rank == 0)
	{
		MPI_File_write_at(file, base, &totalThreads, 1, MPI_BYTE, MPI_STATUS_IGNORE);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Offset offset = sizeof(uint8_t) + numThreads * sizeof(int) * rank;
	MPI_File_write_at(file, offset, &writeSize, numThreads, MPI_INT, MPI_STATUS_IGNORE);
	base += 1 + totalThreads * sizeof(int);

	// Parallel region
	//============================================================================================
	int errFlag = 0;
	
	#pragma omp parallel firstprivate(threadPosition)
	{
		float *data = (float *)malloc(arraySize * sizeof(float)); //Data that each thread will write to the file
		srand(time(NULL) + rank * omp_get_num_threads() + omp_get_thread_num());

		// Initialize the matrix
		//============================================================================================
		for (int i = 0; i < arraySize; i++)
		{
			data[i] = (rand() / (float)RAND_MAX) * 1000;
		}

		// Each thread of each process determines its position on the file
		MPI_Exscan_omp(arraySize, &threadPosition);

		// Start writing to file
		//============================================================================================
		#pragma omp barrier
		#pragma omp single
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Offset offset = base + threadPosition * sizeof(float);

		MPI_File_write_at(file, offset, data, writeSize[omp_get_thread_num()], MPI_FLOAT, MPI_STATUS_IGNORE);

		// Read from file
		//============================================================================================
		#pragma omp barrier
		#pragma omp single
		MPI_Barrier(MPI_COMM_WORLD);

		float *buffer = (float *)malloc(arraySize * sizeof(float));
		MPI_File_read_at(file, offset, buffer, arraySize, MPI_FLOAT, MPI_STATUS_IGNORE);

		//Check for errors hile reading
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
	// End of parallel region

	MPI_File_close(&file);

	// Error handling
	//============================================================================================
	int *flags;

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
				printf("-> Verification failed.\n");
				success = 0;
				break;
			}
		}

		free(flags);

		if (success)
		{
			printf("-> Successful verification.\n");
		}
	}

	MPI_Finalize();

	return 0;
}