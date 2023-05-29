#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

void print(bool* sieve_eratosthenes, int min, int max) {
    int primes_amount = 0;
    int counter = 0;

    printf("Liczby pierwsze dla przedzialu %d - %d:\n\n", min, max);
    for (int i = 0; i <= max - min; i++) {
        if (sieve_eratosthenes[i]) {
            printf("%d ", i + min);
            primes_amount++;
            counter++;
        }
        if (counter == 10) {
            counter = 0;
            printf("\n");
        }
    }

    printf("\n\nIlosc liczb pierwszych w przedziale %d - %d wynosi: %d\n", min, max, primes_amount);
}

void get_start_primes(bool* start_primes, int min) {
    for (int i = 0; i < min - 2; i++) {
        start_primes[i] = true;
    }

    for (int i = 2; i <= sqrt(min); i++) {
        if (start_primes[i - 2]) {
            for (int j = 2 * i; j <= min; j += i) {
                start_primes[j - 2] = false;
            }
        }
    }
}

int main() {
    int min = 2;
    int max = 1000000000;
    int max_sqrt = sqrt(max);
    int thread_number = 8;
    bool* start_primes = (bool*)calloc(max - 2, sizeof(bool));
    bool* sieve_eratosthenes = (bool*)calloc(max - min + 1, sizeof(bool));
    bool print_result = true;

    if (min > 2) {
        get_start_primes(start_primes, min);
    }

    #pragma omp parallel
    {
        bool* thread_start_primes = (bool*)calloc(max - 2, sizeof(bool));
        bool* thread_sieve_eratosthenes = (bool*)calloc(max - min + 1, sizeof(bool));

        #pragma omp for schedule(dynamic)
        for (int x = 0; x < thread_number; x++) {
            int start = x * (max / thread_number) + 2;
            if (start < 2) {
                start = 2;
            }
            int end = (x + 1) * (max / thread_number);
            if (x == thread_number - 1) {
                end = max;
            }

            for (int i = start; i <= end; i++) {
                if (i < min) {
                    if (start_primes[i - 2]) {
                        int remainder = min % i;
                        for (int j = min - remainder; j <= max; j += i) {
                            thread_sieve_eratosthenes[j - min] = false;
                        }
                    }
                }
                else {
                    if (thread_sieve_eratosthenes[i - min]) {
                        for (int j = 2 * i; j <= max; j += i) {
                            thread_sieve_eratosthenes[j - min] = false;
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i <= max - 2; i++) {
                start_primes[i] |= thread_start_primes[i];
            }
            for (int i = 0; i <= max - min; i++) {
                sieve_eratosthenes[i] |= thread_sieve_eratosthenes[i];
            }
        }

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
