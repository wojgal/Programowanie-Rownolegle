#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <omp.h>

#define N 100000001
#define CHUNK_SIZE 10000

int main() {
    bool *isPrime = (bool *)malloc(N * sizeof(bool));
    if (isPrime == NULL) {
        printf("Błąd alokacji pamięci.\n");
        return 1;
    }

    int i, j;

    // Inicjalizacja tablicy
    for (i = 2; i < N; i++) {
        isPrime[i] = true;
    }

    // Zrównoleglenie obliczeń przy użyciu OpenMP
    #pragma omp parallel shared(isPrime) private(i, j)
    {
        int numThreads = omp_get_num_threads();
        int threadId = omp_get_thread_num();
        int numChunks = N / CHUNK_SIZE;
        int chunkStart = threadId * numChunks;
        int chunkEnd = chunkStart + numChunks;

        // Obszar obliczeń dla każdego wątku
        for (i = 2 + chunkStart; i < N && i < 2 + chunkEnd; i++) {
            if (isPrime[i]) {
                for (j = i * i; j < N; j += i) {
                    isPrime[j] = false;
                }
            }
        }
    }

    // Wyświetlanie liczb pierwszych
    for (i = 2; i < N; i++) {
        if (isPrime[i]) {
            printf("%d ", i);
        }
    }

    free(isPrime);  // Zwolnienie pamięci

    return 0;
}
