#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

void calculate_sieve(int* sieve_eratosthenes, int start, int end) {
    int i, j;

#pragma omp parallel for shared(sieve_eratosthenes) private(i, j)
    for (i = start; i <= end; i++) {
        if (sieve_eratosthenes[i]) {
            for (j = 2 * i; j <= end; j += i) {
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
    int thread_number = 8;

    int* sieve_eratosthenes = new int[max];

    //Ustawienie sita na wartosci 1
    for (int i = 0; i < max; i++) {
        sieve_eratosthenes[i] = 1;
    }

    //Sprawdzenie liczb czy sa pierwsze metoda sita Eratostenesa
#pragma omp parallel for schedule(static)
    for (int i = 0; i < thread_number; i++) {
        int start = i * (max / thread_number) + 1;
        int end = (i + 1) * (max / thread_number);
        int x, y;
#pragma omp parallel for shared(sieve_eratosthenes) private(x, y)
        for (x = start; x <= end; x++) {
            if (sieve_eratosthenes[x]) {
                for (y = 2 * x; y <= end; y += x) {
                    sieve_eratosthenes[y] = 0;
                }
            }
        }
    }

#pragma omp parallel for reduction(+:primes_amount)
    for (int i = 2; i < max; i++) {
        if (sieve_eratosthenes[i]) {
            primes_amount++;
        }
    }

    printf("[Dodawanie Domenowe] Ilosc liczb pierwszych: %d", primes_amount);

    return 0;
}
