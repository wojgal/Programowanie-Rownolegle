#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <omp.h>

#define N 100000001

int main() {
    bool *isPrime = (bool *)malloc(N * sizeof(bool));
    if (isPrime == NULL) {
        printf("Błąd alokacji pamięci.\n");
        return 1;
    }

    bool *tmp = (bool *)malloc(N * sizeof(bool));
    if (tmp == NULL) {
        printf("Błąd alokacji pamięci.\n");
        free(isPrime);
        return 1;
    }

    int i, j;

    // Inicjalizacja tablicy
    for (i = 2; i < N; i++) {
        isPrime[i] = true;
    }

    // Zrównoleglenie obliczeń przy użyciu OpenMP
    #pragma omp parallel shared(isPrime, tmp) private(i, j)
    {
        int numThreads = omp_get_num_threads();
        int threadId = omp_get_thread_num();
        int chunkSize = (N - 2) / numThreads;
        int start = 2 + threadId * chunkSize;
        int end = start + chunkSize;

        // Obliczenia na tablicy pomocniczej
        for (i = start; i < end; i++) {
            tmp[i] = true;
        }

        for (i = 2; i * i < N; i++) {
            if (isPrime[i]) {
                for (j = i * i; j < N; j += i) {
                    tmp[j] = false;
                }
            }
        }

        // Kopiowanie wyników do tablicy isPrime
        #pragma omp barrier
        for (i = start; i < end; i++) {
            isPrime[i] = tmp[i];
        }
    }

    // Wyświetlanie liczb pierwszych
    for (i = 2; i < N; i++) {
        if (isPrime[i]) {
            printf("%d ", i);
        }
    }

    free(isPrime);  // Zwolnienie pamięci
    free(tmp);  // Zwolnienie pamięci

    return 0;
}
