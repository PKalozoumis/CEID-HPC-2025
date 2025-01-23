#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// Define the minimum and the maximum
// integer of the random number generator
#define MAX 1000
#define MIN 1

int rank, size;

//============================================================================================

void MPI_Exscan_pt2pt(int *in, int *out, int rank)
{
    int prev = 0;

    // All processes except the 0 wait to receive the sum of the previous process
    if (rank != 0)
    {
        MPI_Recv(&prev, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // The result of a process
    *out = prev;

    //Each process except the last one send the sum to the next
    if (rank != size - 1)
    {
        // Calculation of sum to send to the next process
        int next = prev + *in;
        MPI_Send(&next, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
}

//============================================================================================

void validation(int *indata, int outdata_check)
{

    int outdata = 0;
    int error = 0;

    // Calling MPI_Exscan with the same data
    MPI_Exscan(indata, &outdata, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Compare the data
    if (outdata_check != outdata)
    {
        error = 1;
    }

    int error_gather[size];

    // Send the result of the validation to the process 0
    MPI_Gather(&error, 1, MPI_INT, error_gather, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {

        int validation = 0;

        // Checking every process validation
        for (int i = 0; i < size; i++)
        {
            if (error_gather[i] == 1)
            {
                validation = 1;
            }
        }

        // Print the result validation
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
    // Initialize the environment, obtain the rank 
    // and the total number of processes
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Print the total number of processes
    if (rank == 0)
        printf("Processes: %d\n", size);

    // Initialize random number generator,
    // unique for each process
    srand(time(NULL) + rank);

    // Create the data
    int indata = (rand() % (MAX - MIN + 1)) + MIN;
    int outdata = 0;

    // Implementation of Exscan function
    MPI_Exscan_pt2pt(&indata, &outdata, rank);

    // Print the result of our Excan function
    printf("Process: %d Indata: %d Result: %d\n", rank, indata, outdata);

    // Synchronization with Barrier
    MPI_Barrier(MPI_COMM_WORLD);

    // Validation the results
    validation(&indata, outdata);

    // Terminate the environment
    MPI_Finalize();

    return 0;
}