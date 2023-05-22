#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

void calculate_sieve(int* sieve_eratosthenes, int start, int end) {
    int i, j;

    #pragma omp parallel for shared(primes) private(i, j)
    for(i = start; i <= end; i++) {
        if(sieve_eratosthenes[i]) {
            for(j = 2*i; j <= end; j += i){
                sieve_eratosthenes[j] = 0;
            }
        }
    }
}

int main() {
    //Tworzenie potzebnych zmiennych
    int max = 100000000;
    int max_sqrt = sqrt(max);
    int primes_amount = 0;
    int thread_number = omp_get_num_threads();

    int* sieve_eratosthenes = malloc(max * sizeof(int));

    //Ustawienie sita na wartosci 1
    for(int i = 0; i < max; i++){
        sieve_eratosthenes[i] = 1;
    }

    //Sprawdzenie liczb czy sa pierwsze metoda sita Eratostenesa
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < thread_number; i++) {
        int start = i * (max / thread_number) + 1;
        int end = (i + 1) * (max / thread_number);

        calculate_sieve(sieve_eratosthenes, start, end);
    }

    #pragma omp for schedule(guided)
    for(int i = 2; i < max; i ++){
        if(sieve_eratosthenes[i]){
            #pragma omp atomic
            primes_amount++;
        }
    }


    return 0;
}
