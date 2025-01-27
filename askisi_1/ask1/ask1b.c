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

void validation(int *indata, int *outdata, int thread)
{

    int indata_share_proc[thread * size];
    int outdata_share_proc[thread * size];

    // Send local indatas to the root process
    MPI_Gather(indata, thread, MPI_INT, indata_share_proc, thread, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(outdata, thread, MPI_INT, outdata_share_proc, thread, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int error = 0;
        int temp = 0;

        for (int i = 1; i < size * thread; i++)
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
    // Initialize the environment, obtain the rank 
    // and the total number of processes
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int threads;

    if (rank == 0)
    {
        // Print the number of processes
        printf("Processes: %d\n", size);
        printf("Maximum number of processors: %d\n", omp_get_num_procs());

        // Give the number of thread each process will have
        do
        {
            printf("How many threads should each process have (Give a intenger number): ");
            scanf("%d", &threads);
        } while (threads < 0);
    }

    // Broadcast the number of the thread to all processes
    MPI_Bcast(&threads, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Set the number of threads that will be used to execute 
    // a parallel region 
    omp_set_num_threads(threads);

    // Disable the dynamic adjuctment of the number of thread available
    // for the execution of a parallel region
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

        // Create the data
        int indata = (rand() % (MAX - MIN + 1)) + MIN;

        // Array used for data validation
        indata_share_thread[omp_get_thread_num()] = indata;

        // Implementation of hybrid Exscan function
        MPI_Exscan_omp(indata, &outdata);

        // Array used for data validation
        outdata_share_thread[omp_get_thread_num()] = outdata;

        // Print the results
        printf("Process: %d Thread: %d Indata: %d Result: %d\n", rank, omp_get_thread_num(), indata, outdata);
    }

    // Validation the results
    validation(indata_share_thread, outdata_share_thread, threads);

     // Terminate the environment
    MPI_Finalize();

    return 0;
}