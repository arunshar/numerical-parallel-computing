#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { int rank; long partial; } Worker;

static void *sum_part(void *pointer) {
    Worker *worker = pointer;
    worker->partial = 0;
    for (int i = worker->rank + 1; i <= 1000; i += 4)
        worker->partial += i;
    return NULL;
}

int main(void) {
    pthread_t threads[4];
    Worker workers[4];
    int created = 0;
    for (int i = 0; i < 4; ++i) {
        workers[i] = (Worker){i, 0};
        if (pthread_create(&threads[i], NULL, sum_part, &workers[i]) != 0) {
            for (int j = 0; j < created; ++j) pthread_join(threads[j], NULL);
            fprintf(stderr, "Thread creation failed.\n");
            return EXIT_FAILURE;
        }
        ++created;
    }
    long total = 0;
    for (int i = 0; i < 4; ++i) {
        if (pthread_join(threads[i], NULL) != 0) return EXIT_FAILURE;
        total += workers[i].partial;
    }
    printf("pthread sum = %ld\n", total);
    return total == 500500 ? EXIT_SUCCESS : EXIT_FAILURE;
}
