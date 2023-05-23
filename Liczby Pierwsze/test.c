#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

int main() {
    //Tworzenie potzebnych zmiennych
    __uint64_t max = 2000000000;
    int max_sqrt = sqrt(max);
    __uint64_t primes_amount = 0;
    int thread_number = 8;

    bool* sieve_eratosthenes = (bool *)calloc((max), sizeof(bool));

    //Ustawienie sita na wartosci 1
    #pragma omp parallel for schedule(guided) shared(sieve_eratosthenes)
    for (__uint64_t i = 0; i < max; i++) {
        sieve_eratosthenes[i] = 1;
    }

    //Sprawdzenie liczb czy sa pierwsze metoda sita Eratostenesa
#pragma omp parallel
{
    for (int i = 0; i < thread_number; i++) {
        __uint64_t start = i * (max / thread_number) + 1;

        if(start < 2){
            start = 2;
        }
        
        __uint64_t end = (i + 1) * (max / thread_number);

        for (__uint64_t x = start; x <= end; x++) {
            if (sieve_eratosthenes[x]) {
                for (__uint64_t y = 2 * x; y <= max; y += x) {
                    sieve_eratosthenes[y] = 0;
                }
            }
        }
    }

    #pragma omp barrier

    #pragma omp parallel for schedule(guided) private(i) reduction(+:primes_amount)
    for (__uint64_t i = 2; i < max; i++) {
        if (sieve_eratosthenes[i]) {
            primes_amount++;
        }
    }
}

    printf("[Dodawanie Domenowe] Ilosc liczb pierwszych: %ld", primes_amount);
    free(sieve_eratosthenes);
    return 0;
}
