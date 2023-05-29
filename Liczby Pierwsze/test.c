#include <stdio.h>
#include <stdbool.h>
#include <omp.h>

#define N 100000001

int main() {
    bool isPrime[N];
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
        int chunkSize = (N - 2) / numThreads;
        int start = 2 + threadId * chunkSize;
        int end = start + chunkSize;

        // Obszar obliczeń dla każdego wątku
        for (i = start; i < end; i++) {
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

    return 0;
}
