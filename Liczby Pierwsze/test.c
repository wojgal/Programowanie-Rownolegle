#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

void print(bool* sieve_eratosthenes, int min, int max) {
    int primes_amount = 0;
    int counter = 0;

    printf("Liczby pierwsze dla przedzialu %d - %d:\n\n", min, max);
    for (int i = 0; i <= max - min; i++) {
        if (sieve_eratosthenes[i]) {
           // printf("%d ", i + min);
            primes_amount++;
            counter++;
        }
        if (counter == 10) {
            counter = 0;
          //  printf("\n");
        }
    }

    printf("\n\nIlosc liczb pierwszych w przedziale %d - %d wynosi: %d\n", min, max, primes_amount);
}

void get_start_primes(bool* start_primes, int min) {
    for (int i = 2; i < min; i++) {
        start_primes[i - 2] = 1;
    }

    for (int i = 2; i <= sqrt(min); i++) {
        if (start_primes[i - 2]) {
            for (int j = 2 * i; j < min; j += i) {
                start_primes[j - 2] = 0;
            }
        }
    }
}


int main() {

    int min = 2;
    int max = 1000000000;
    int i, j;

    bool* sieve_eratosthenes = (bool*)malloc(max-min+1 * sizeof(bool));
    bool* start_primes = (bool*)calloc(min - 2, sizeof(bool));
    
    //Zmienna odpowiadajca za wypisywanie 
    bool print_result = true;

    if (min > 2) {
        get_start_primes(start_primes, min);
    }

    // Inicjalizacja tablicy
#pragma omp parallel for schedule(guided)
    for (i = min; i <= max; i++) {
        sieve_eratosthenes[i-min] = 1;
    }

    omp_set_num_threads(1);
    // Zrównoleglenie obliczeń przy użyciu OpenMP
#pragma omp parallel shared(sieve_eratosthenes) private(i, j)
    {
        int threadId = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        printf("%d", numThreads);
        int start = threadId * (max/numThreads)+1;
        int end = (threadId+1)*(max/numThreads);

        if (threadId==0) {
            start = min;
        }

        if (threadId == numThreads - 1) {
            end = max;
        }

        for (i = start; i <= end; i++) {
            if (sieve_eratosthenes[i-min]) {
                //printf("[%d] prime: %d\n", threadId, i);
                for (j = 2*i; j <= max; j += i) {
                    sieve_eratosthenes[j-min] = 0;
                }
            }
        }
    }

    if (print_result) {
        print(sieve_eratosthenes, min, max);
    }
   
    free(sieve_eratosthenes);  // Zwolnienie pamięci
    free(start_primes);

    return 0;
}
