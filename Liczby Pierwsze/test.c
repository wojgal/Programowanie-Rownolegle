#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

int main() {
    __uint64_t min = 2;
    __uint64_t max = 2000000000;
    int max_sqrt = sqrt(max);
    __uint64_t primes_amount = 0;

    bool* sieve_eratosthenes = (bool *)calloc((max_sqrt + 1), sizeof(bool));

    // Ustawienie sita na wartości 1
    #pragma omp parallel for schedule(guided) shared(sieve_eratosthenes)
    for (__uint64_t i = min; i <= max_sqrt; i++) {
        sieve_eratosthenes[i] = 1;
    }

    #pragma omp parallel for schedule(dynamic) shared(sieve_eratosthenes)
    for (__uint64_t i = min; i <= max_sqrt; i++) {
        if (sieve_eratosthenes[i]) {
            for (__uint64_t j = 2 * i; j <= max_sqrt; j += i) {
                sieve_eratosthenes[j] = 0;
            }
        }
    }

    #pragma omp parallel for schedule(guided) reduction(+:primes_amount)
    for (__uint64_t i = min; i < max; i++) {
        if (i <= max_sqrt && sieve_eratosthenes[i]) {
            primes_amount++;
        }
        else if (i > max_sqrt) {
            bool is_prime = true;
            for (__uint64_t j = 2; j <= max_sqrt; j++) {
                if (sieve_eratosthenes[j] && i % j == 0) {
                    is_prime = false;
                    break;
                }
            }
            if (is_prime) {
                primes_amount++;
            }
        }
    }

    printf("[Dodawanie Funkcyjne] Ilosc liczb pierwszych: %ld", primes_amount);
    free(sieve_eratosthenes);
    return 0;
}
