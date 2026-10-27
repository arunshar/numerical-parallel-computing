#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    int received = -1;
    MPI_Sendrecv(&rank, 1, MPI_INT, right, 10,
                 &received, 1, MPI_INT, left, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int ok = received == left, all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI ring exchange: ranks = %d, all_correct = %d\n", size, all_ok);
    MPI_Finalize();
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
