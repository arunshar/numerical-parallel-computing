#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    long sum = 0;
    int actual_threads = 0;
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(4) default(none) shared(actual_threads) reduction(+:sum)
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int i = 1; i <= 1000; ++i) sum += i;
    }
    printf("OpenMP sum = %ld, threads = %d\n", sum, actual_threads);
    return sum == 500500 ? EXIT_SUCCESS : EXIT_FAILURE;
}
