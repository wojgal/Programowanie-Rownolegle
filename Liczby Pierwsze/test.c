#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

int main() {
    __uint64_t max = 2000000000;
    int max_sqrt = sqrt(max);
    __uint64_t primes_amount = 0;
    int thread_number = 8;

    bool* sieve_eratosthenes = (bool *)calloc(max, sizeof(bool));

    // Ustawienie sita na wartości 1
    #pragma omp parallel for schedule(guided)
    for (__uint64_t i = 0; i < max; i++) {
        sieve_eratosthenes[i] = true;
    }

    // Sprawdzenie liczb czy są pierwsze metodą sita Eratostenesa
    #pragma omp parallel num_threads(thread_number)
    {
        int tid = omp_get_thread_num();
        __uint64_t start = tid * (max / thread_number) + 1;
        if (start < 2) {
            start = 2;
        }
        __uint64_t end = (tid + 1) * (max / thread_number);

        for (__uint64_t x = start; x <= end; x++) {
            if (sieve_eratosthenes[x]) {
                for (__uint64_t y = 2 * x; y <= max; y += x) {
                    sieve_eratosthenes[y] = false;
                }
            }
        }
    }

    #pragma omp parallel for schedule(guided) reduction(+:primes_amount)
    for (__uint64_t i = 2; i < max; i++) {
        if (sieve_eratosthenes[i]) {
            primes_amount++;
        }
    }

    printf("[Dodawanie Domenowe] Ilość liczb pierwszych: %ld\n", primes_amount);
    free(sieve_eratosthenes);
    return 0;
}
