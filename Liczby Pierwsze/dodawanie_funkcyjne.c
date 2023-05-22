#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int max = 100000000;
    int max_sqrt = sqrt(max);
    int primes_amount = 0;
    int i, j;

    int* sieve_eratosthenes = malloc(max * sizeof(int));

    //Ustawienie sita na wartosci 1
    for(i = 0; i < max + 1; i++){
        sieve_eratosthenes[i] = 1;
    }

    #pragma omp parallel for shared(sieve_eratosthenes, primes_amount) private(i, j)
    for(i = 2; i <= max_sqrt; i++){
        if(sieve_eratosthenes){
            for(j = 2*i; j < max; j += i){
                sieve_eratosthenes[j] = 0;
            }
            primes_amount++;
        }
    }

    return 0;
}
