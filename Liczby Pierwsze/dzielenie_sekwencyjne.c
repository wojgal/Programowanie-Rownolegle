#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

int main(){
    //Tworzenie potrzebnych zmiennych
    int max = 100000000;
    int max_sqrt = sqrt(max);
    int primes_amount = 0;

    int* sieve_eratosthenes = malloc(max * sizeof(int));

    //Ustawienie sita na wartosci 1
    for(int i = 0; i < max; i++){
        sieve_eratosthenes[i] = 1;
    }

    //Sprawdzenie liczb czy sa pierwsze metoda sita Eratostenesa
    for(int i = 2; i <= max_sqrt; i++){
        if(sieve_eratosthenes[i]){
            for(int j = 2*i; j <= max; j += i){
                sieve_eratosthenes[j] = 0;
            }
        }
    }

    //Podliczanie ilosci liczb pierwszych
    for(int i = 2; i < max; i++){
        if(sieve_eratosthenes[i]){
            primes_amount++;
        }
    }

    printf("[Dodawanie Sekwencyjne] Ilosc liczb pierwszych: %d", primes_amount);
    
    return 0;
    
}
