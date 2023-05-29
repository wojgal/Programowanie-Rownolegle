#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

void print(bool* sieve_eratosthenes, int min, int max) {
    // Jak wcześniej
    // ...
}

void get_start_primes(bool* start_primes, int min) {
    // Jak wcześniej
    // ...
}

int main() {
    int min = 2;
    int max = 1000000000;
    int max_sqrt = sqrt(max);
    int thread_number = 8;
    bool* start_primes = (bool*)calloc(min - 2, sizeof(bool));
    bool* sieve_eratosthenes = (bool*)calloc(max - min + 1, sizeof(bool));
    bool print_result = false;

    if (min > 2) {
        get_start_primes(start_primes, min);
    }

    // Usunięcie linii: omp_set_num_threads(8);
    // Liczba wątków zostanie automatycznie określona przez OpenMP

    #pragma omp parallel
    {
        // Każdy wątek ma swoją kopię zmiennych
        bool* thread_start_primes = (bool*)calloc(min - 2, sizeof(bool));
        bool* thread_sieve_eratosthenes = (bool*)calloc(max - min + 1, sizeof(bool));

        // Każdy wątek inicjalizuje lokalne kopie start_primes i sieve_eratosthenes
        for (int i = 2; i < min; i++) {
            thread_start_primes[i - 2] = 1;
        }

        for (int i = 2; i <= sqrt(min); i++) {
            if (thread_start_primes[i - 2]) {
                for (int j = 2 * i; j < min; j += i) {
                    thread_start_primes[j - 2] = 0;
                }
            }
        }

        // Każdy wątek wykonuje obliczenia na swoim zakresie
        #pragma omp for schedule(dynamic)
        for (int x = 0; x < thread_number; x++) {
            int start = x * (max / thread_number) + 1;
            if (start < 2) {
                start = 2;
            }
            int end = (x + 1) * (max / thread_number);
            if (x == thread_number - 1) {
                end = max;
            }

            for (int i = start; i <= end; i++) {
                if (i < min) {
                    if (thread_start_primes[i - 2]) {
                        int remainder = min % i;
                        for (int j = min - remainder; j <= max; j += i) {
                            thread_sieve_eratosthenes[j - min] = 0;
                        }
                    }
                }
                else {
                    if (thread_sieve_eratosthenes[i - min]) {
                        for (int j = 2 * i; j <= max; j += i) {
                            thread_sieve_eratosthenes[j - min] = 0;
                        }
                    }
                }
            }
        }

        // Połączenie wyników z każdego wątku w tablicach współdzielonych
        #pragma omp critical
        {
            for (int i = 0; i <= max - min; i++) {
                sieve_eratosthenes[i] |= thread_sieve_eratosthenes[i];
            }
        }

        // Zwolnienie pamięci zaalokowanej dla lokalnych tablic
        free(thread_start_primes);
        free(thread_sieve_eratosthenes);
    }

    if (print_result) {
        print(sieve_eratosthenes, min, max);
    }

    free(start_primes);
    free(sieve_eratosthenes);

    return 0;
}
