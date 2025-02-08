#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MAX 9
#define MIN 1

int rank, size;
int prev = 0;

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

void validation(int *indata, int *outdata, int thread_count)
{
    int indata_share_proc[thread_count * size];
    int outdata_share_proc[thread_count * size];

    // Send local indatas to the root process
    MPI_Gather(indata, thread_count, MPI_INT, indata_share_proc, thread_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(outdata, thread_count, MPI_INT, outdata_share_proc, thread_count, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int error = 0;
        int temp = 0;

        for (int i = 1; i < size * thread_count; i++)
        {
            // Compare the data for each process
            // and its threads
            temp += indata_share_proc[i - 1];

            if (outdata_share_proc[i] != temp)
            {
                error = 1;
                break;
            }
        }

        // Print the result of the validation
        if (error == 1)
        {
            printf("Verification failed\n");
        }
        else
        {
            printf("Successful verification\n");
        }
    }
}

//============================================================================================

//It just prints results in the correct order
void print_ordered(const char* str)
{
    int signal = 0;

    #pragma omp single
    if (rank > 0)
        MPI_Recv(&signal, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int num_threads = omp_get_num_threads();
    int thread_num = omp_get_thread_num();

    //Flags tell which process is allowed to proceed
    static int* flags;

    //Array of flags for the process
    #pragma omp single
    flags = (int*)calloc(num_threads, sizeof(int));

    //First thread is free to proceed
    flags[0] = 1;

    //Busy wait until the previous thread releases you
    while (!flags[thread_num]){};

    printf("%s", str);
    fflush(stdout);

    //Release the next thread of the process
    if (thread_num < num_threads-1)
        flags[thread_num + 1] = 1;

    //Once all threads are done, release the next process
    #pragma omp barrier
    #pragma omp single
    {
        printf("\n");
        fflush(stdout);

        if (rank < size - 1)
            MPI_Ssend(&signal, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[])
{
    // Initialize the environment, obtain the rank 
    // and the total number of processes
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int threads;

    if (argc == 1)
    {
        if (rank == 1)
            printf("Give num of threads.\n");

        MPI_Finalize();
        return 0;
    }

    threads = atoi(argv[1]);

    if (rank == 0)
    {
        printf("Processes: %d\nThreads: %d\n\n", size, threads);
    }

    omp_set_num_threads(threads);
    omp_set_dynamic(0);

    // Setup the data
    int outdata = 0;
    int indata_share_thread[threads];
    int outdata_share_thread[threads];

    #pragma omp parallel firstprivate(outdata)
    {
        // Initialize random number generator, unique for each process
        // and each thread
        srand(time(NULL) + rank * omp_get_num_threads() + omp_get_thread_num());

        int indata = (rand() % (MAX - MIN + 1)) + MIN;

        // Array used for data validation
        indata_share_thread[omp_get_thread_num()] = indata;

        // Implementation of hybrid Exscan function
        MPI_Exscan_omp(indata, &outdata);

        // Array used for data validation
        outdata_share_thread[omp_get_thread_num()] = outdata;

        // Print the results
        char msg[100];
        sprintf(msg, "Process: %.02d Thread: %.02d Indata: %d Result: %d\n", rank, omp_get_thread_num(), indata, outdata);
        //printf("Process: %d Thread: %d Indata: %d Result: %d\n", rank, omp_get_thread_num(), indata, outdata);
        print_ordered(msg);
        
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Validation the results
    validation(indata_share_thread, outdata_share_thread, threads);

    MPI_Finalize();

    return 0;
}