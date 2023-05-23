#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

int main(){
    //Tworzenie potrzebnych zmiennych
    __uint64_t max = 2000000000;
    int max_sqrt = sqrt(max);
    __uint64_t primes_amount = 0;

    bool* sieve_eratosthenes = (bool *)calloc(max, sizeof(bool));

    //Ustawienie sita na wartosci 1
    for(__uint64_t i = 0; i < max; i++){
        sieve_eratosthenes[i] = 1;
    }

    //Sprawdzenie liczb czy sa pierwsze metoda sita Eratostenesa
    for(__uint64_t i = 2; i <= max_sqrt; i++){
        if(sieve_eratosthenes[i]){
            for(__uint64_t j = 2*i; j <= max; j += i){
                sieve_eratosthenes[j] = 0;
            }
        }
    }

    //Podliczanie ilosci liczb pierwszych
    for(__uint64_t i = 2; i < max; i++){
        if(sieve_eratosthenes[i]){
            primes_amount++;
        }
    }

    printf("[Dodawanie Sekwencyjne] Ilosc liczb pierwszych: %ld", primes_amount);
    free(sieve_eratosthenes);
    return 0;
    
}
