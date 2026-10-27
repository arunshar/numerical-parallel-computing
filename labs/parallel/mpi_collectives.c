#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank == 0) fprintf(stderr, "This example requires four ranks.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int values[4] = {3, 7, 11, 15}, local = 0, root_sum = 0, all_sum = 0;
    MPI_Scatter(values, 1, MPI_INT, &local, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local, &root_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&local, &all_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int dims[2] = {2, 2}, periods[2] = {1, 0};
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    int coords[2], source, destination;
    MPI_Cart_coords(grid, rank, 2, coords);
    MPI_Cart_shift(grid, 1, 1, &source, &destination);
    int expected_left = coords[1] == 0 ? MPI_PROC_NULL : rank - 1;
    int expected_right = coords[1] == 1 ? MPI_PROC_NULL : rank + 1;
    int ok = all_sum == 36 && source == expected_left && destination == expected_right;
    if (rank == 0) ok = ok && root_sum == 36;
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI collectives: sum = %d, topology_correct = %d\n", all_sum, all_ok);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
