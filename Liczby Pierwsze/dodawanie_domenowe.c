#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

int main() {
    //Tworzenie potzebnych zmiennych
    int max = 100000000;
    int max_sqrt = sqrt(max);
    int primes_amount = 0;
    int thread_number = 8;

    int* sieve_eratosthenes = (int*)malloc(max * sizeof(int));

    //Ustawienie sita na wartosci 1
    for (int i = 0; i < max; i++) {
        sieve_eratosthenes[i] = 1;
    }

    //Sprawdzenie liczb czy sa pierwsze metoda sita Eratostenesa
//#pragma omp parallel for schedule(static)
    for (int i = 0; i < thread_number; i++) {
        int start = i * (max / thread_number) + 1;

        if(start < 2){
            start = 2;
        }
        
        int end = (i + 1) * (max / thread_number);
        int x, y;

//#pragma omp parallel for private(x, y)
        for (x = start; x <= end; x++) {
            if (sieve_eratosthenes[x]) {
                for (y = 2 * x; y <= max; y += x) {
                    sieve_eratosthenes[y] = 0;
                }
            }
        }
    }

//#pragma omp parallel for reduction(+:primes_amount)
    for (int i = 2; i < max; i++) {
        if (sieve_eratosthenes[i]) {
            primes_amount++;
        }
    }

    printf("[Dodawanie Domenowe] Ilosc liczb pierwszych: %d", primes_amount);

    return 0;
}
