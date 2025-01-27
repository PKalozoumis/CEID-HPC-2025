#include <zfp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

int rank, size;

//============================================================================================

void MPI_Exscan_omp(int in, int *out)
{
	static int prev = 0; // The offset in the file for the entire process
	static int *process_data;
	int threadnum = omp_get_thread_num();

	#pragma omp single
	process_data = (int *)malloc(omp_get_num_threads() * sizeof(int));

	#pragma omp barrier
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

size_t compress_array(float *data, size_t N, unsigned char **compressedData)
{
	// int precision: Affects truncation. Default is 24 bits. Cannot exceed 32 bits for floats
	// double tolerance: The accuracy. How close the compressed data is to the original. Controls the MAX ERROR allowed during the compression process. Default is 0
	/*
	  for (int i = 0; i < N; i++)
	  {
		printf("original data[%d]: %f\n", i, data[i]);
	  }
	*/
	zfp_type type = zfp_type_float; // Specify float type
	zfp_field *field = zfp_field_3d(data, type, N, N, N);
	zfp_stream *zfp = zfp_stream_open(NULL);

	zfp_stream_set_accuracy(zfp, 0.001);

	// Allocate buffer for compressed data
	size_t bufsize = zfp_stream_maximum_size(zfp, field);
	*compressedData = (unsigned char *)malloc(bufsize);
	if (compressedData == NULL)
	{
		printf("Memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	// Associate buffer with ZFP stream
	bitstream *stream = stream_open(*compressedData, bufsize);
	zfp_stream_set_bit_stream(zfp, stream);
	zfp_stream_rewind(zfp);

	// Compress
	size_t compressedSize = zfp_compress(zfp, field);
	if (!compressedSize)
	{
		fprintf(stderr, "ZFP compression failed\n");
		exit(EXIT_FAILURE);
	}

	// Clean up
	zfp_field_free(field);
	zfp_stream_close(zfp);
	stream_close(stream);

	return compressedSize;
}

//============================================================================================

void decompress_array(unsigned char *compressedData, size_t compressedSize, float *originalData, size_t N)
{
	zfp_type type = zfp_type_float;
	zfp_field *field = zfp_field_3d(originalData, type, N, N, N);
	if (field == NULL)
	{
		fprintf(stderr, "Failed to create ZFP field for decompression\n");
		exit(EXIT_FAILURE);
	}
	zfp_stream *zfp = zfp_stream_open(NULL);
	if (zfp == NULL)
	{
		fprintf(stderr, "Failed to create ZFP stream for decompression\n");
		zfp_field_free(field); // Free the field before exiting
		exit(EXIT_FAILURE);
	}
	zfp_stream_set_accuracy(zfp, 0.001);

	size_t bufsize = zfp_stream_maximum_size(zfp, field);

	bitstream *stream = stream_open(compressedData, bufsize);
	if (stream == NULL)
	{
		fprintf(stderr, "Failed to create bitstream for decompression\n");
		zfp_field_free(field); // Free the field before exiting
		zfp_stream_close(zfp); // Close the stream before exiting
		exit(EXIT_FAILURE);
	}
	zfp_stream_set_bit_stream(zfp, stream);
	zfp_stream_rewind(zfp);

	if (!zfp_decompress(zfp, field))
	{
		fprintf(stderr, "ZFP decompression failed\n");
		exit(EXIT_FAILURE);
	}
	/*
	  for (int i = 0; i < N; i++) // Print first 10 values for quick verification
	  {
		printf("Decompressed data[%d]: %f\n", i, originalData[i]);
	  }
	*/
	zfp_field_free(field);
	zfp_stream_close(zfp);
	stream_close(stream);
}

//================================================================

int main(int argc, char *argv[])
{

	int errFlag = 0;
	int *flags;

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
			printf("The maximum threads are %d. Using %d threads.\n", omp_get_num_procs(), omp_get_num_procs());

		numThreads = omp_get_num_procs();
	}

	//============================================================================================
	printf("Size: %ld\n", arraySize * sizeof(float));

	uint8_t totalThreads = numThreads * size; // Threads across all processes

	omp_set_num_threads(numThreads);
	omp_set_dynamic(0);

	int outdata = 0;
	int writeSize[numThreads];

	MPI_File file;
	MPI_File_open(MPI_COMM_WORLD, "file_erotima_d.bin", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);
	MPI_File_set_size(file, 0);

	MPI_Offset base;
	MPI_File_get_position(file, &base);

	// Write file header
	//============================================================================================
	if (rank == 0)
	{
		MPI_File_write_at(file, base, &totalThreads, 1, MPI_BYTE, MPI_STATUS_IGNORE);
	}

	#pragma omp barrier
	#pragma omp single
	MPI_Barrier(MPI_COMM_WORLD);

	// Parallel region
	//============================================================================================
	#pragma omp parallel firstprivate(outdata)
	{
		int threadNum = omp_get_thread_num();
		float *data = (float *)malloc(arraySize * sizeof(float));
		if (data == NULL)
		{
			printf("Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}

		// Initialize the matrix
		//============================================================================================
		srand(time(NULL) + rank * omp_get_num_threads() + omp_get_thread_num());

		for (int i = 0; i < arraySize; i++)
		{
			data[i] = (rand() / (float)RAND_MAX) * 1000;
		}

		// Compress the matrix
		//============================================================================================
		unsigned char *compressedData = NULL;
		size_t compressedSize = compress_array(data, N, &compressedData);

		printf("compressedSize: %ld\n", compressedSize);

		writeSize[threadNum] = compressedSize;

		#pragma omp barrier
		#pragma omp single
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Exscan_omp(compressedSize, &outdata);

		// Continue writing header
		//============================================================================================

		// One thread from each process must write in the header how many bytes each thread will write
		#pragma omp single
		{
			MPI_Offset header_offset = 1 + numThreads * sizeof(int) * rank;
			MPI_File_write_at(file, header_offset, &writeSize, numThreads, MPI_INT, MPI_STATUS_IGNORE);
			base = 1 + totalThreads * sizeof(int);
		}

		// printf("Base: %ld\n", base);
		// printf("Start at: %d\n", outdata);

		// Start writing to file
		//============================================================================================
		#pragma omp barrier
		#pragma omp single
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Offset offset = base + outdata;
		// printf("offset %ld\n",offset);
		MPI_File_write_at(file, offset, compressedData, compressedSize, MPI_BYTE, MPI_STATUS_IGNORE);

		// Read from file
		//============================================================================================
		#pragma omp barrier
		#pragma omp single
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_File_read_at(file, offset, compressedData, compressedSize, MPI_BYTE, MPI_STATUS_IGNORE);

		float *decompressedData = (float *)malloc(arraySize * sizeof(float));
		if (decompressedData == NULL)
		{
			printf("Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}

		decompress_array(compressedData, compressedSize, decompressedData, N);

		for (int i = 0; i < arraySize; i++)
		{
			if (fabs(decompressedData[i] - data[i]) > 2)
			{
				printf("decompressed: %lf, data: %lf, i: %d\n", decompressedData[i], data[i], i);
				errFlag = 1;

				break;
			}
		}

		free(data);
		free(compressedData);
		free(decompressedData);
	}
	// End of parallel region

	// Error handling
	//============================================================================================

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