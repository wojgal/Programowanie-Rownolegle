#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

//Funkcja wypisujaca wyniki
void print(bool* sieve_eratosthenes, int min, int max) {
    int primes_amount = 0;
    int counter = 0;

    printf("Liczby pierwsze dla przedzialu %d - %d:\n\n", min, max);
    for (int i = 0; i <= max - min; i++) {
        if (sieve_eratosthenes[i]) {
            printf("%d ", i + min);
            primes_amount++;
            counter++;
        }
        if (counter == 10) {
            counter = 0;
            printf("\n");
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
    //Tworzenie potzebnych zmiennych
    int min = 50;
    int max = 100;

    bool* sieve_eratosthenes = (bool*)calloc(max - min + 1, sizeof(bool));

    //Zmienna odpowiadajca za wypisywanie 
    bool print_result = true;

    omp_set_num_threads(8);
    //Sprawdzenie liczb czy sa pierwsze metoda sita Eratostenesa
#pragma omp parallel shared(sieve_eratosthenes)
{
        int thread_number = omp_get_num_threads();
        int x = omp_get_thread_num();

        int start = x * (max / thread_number) + 1;

        if (start < 2) {
            start = 2;
        }
        
        int end = (x + 1) * (max / thread_number);
        if (x == thread_number - 1) {
            end = max;
        }

        printf("[%d] %d %d", x, start, end);
        bool* thread_sieve = (bool*)calloc(end - start + 1, sizeof(bool));
        bool* start_primes = (bool*)calloc(start - min + 1, sizeof(bool));

        get_start_primes(start_primes, start);


        for (int i = start; i <= end; i++) {
            //Liczby przed zakresem
            if (i < start) {
                if (start_primes[i - start]) {
                    int remainder = start % i;

                    for (int j = start - remainder; j <= end; j += i) {
                        thread_sieve[j - start] = 0;
                    }
                }
            }
            //Liczby po zakresie
            else {
                if (sieve_eratosthenes[i - start]) {
                    for (int j = 2 * i; j <= end; j += i) {
                        thread_sieve[j - start] = 0;
                    }
                }
            }
        }

        for(int i = 0; i < end - start + 1;){
            sieve_eratosthenes[i + start] = thread_sieve[i];
        }

        free(thread_sieve);
        free(start_primes);
    }

    if (print_result) {
        print(sieve_eratosthenes, min, max);
    }
    
    free(sieve_eratosthenes);

    return 0;
}
