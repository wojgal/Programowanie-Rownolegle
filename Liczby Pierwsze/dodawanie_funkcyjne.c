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

    bool* sieve_eratosthenes = (bool *)calloc((max), sizeof(bool));

    //Ustawienie sita na wartosci 1
    #pragma omp parallel for schedule(guided) shared(sieve_eratosthenes)
    for(__uint64_t i = min; i < max + 1; i++){
        sieve_eratosthenes[i] = 1;
    }

    #pragma omp parallel for schedule(dynamic) shared(sieve_eratosthenes)
    for(__uint64_t i = min; i <= max_sqrt; i++){
        if(sieve_eratosthenes[i]){
            for(__uint64_t j = 2*i; j < max; j += i){
                sieve_eratosthenes[j] = 0;
            }
        }
    }

    #pragma omp parallel for schedule(guided) reduction(+:primes_amount)
    for(__uint64_t i = min; i < max; i ++){
        if(sieve_eratosthenes[i]){
            #pragma omp atomic
            primes_amount++;
        }
    }

    printf("[Dodawanie Funkcyjne] Ilosc liczb pierwszych: %ld", primes_amount);
    free(sieve_eratosthenes);
    return 0;
}
