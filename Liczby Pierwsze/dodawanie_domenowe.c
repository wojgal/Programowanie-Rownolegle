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
    for (int i = 0; i <= max - min + 1; i++) {
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
    int min = 53;
    int max = 100;
    int max_sqrt = sqrt(max);
    int thread_number = 8;
    bool* start_primes;

    bool* sieve_eratosthenes = (bool*)calloc(max - min + 1, sizeof(bool));

    //Zmienna odpowiadajca za wypisywanie 
    bool print_result = true;

    if (min > 2) {
        start_primes = (bool*)calloc(min - 2, sizeof(bool));
        get_start_primes(start_primes, min);
    }

    //Ustawienie sita na wartosci 1
    for (int i = min; i <= max; i++) {
        sieve_eratosthenes[i-min] = 1;
    }

    //Sprawdzenie liczb czy sa pierwsze metoda sita Eratostenesa
//#pragma omp parallel for schedule(static)
    for (int x = 0; x < thread_number; x++) {
        int start = x * (max / thread_number) + 1;

        if (start < 2) {
            start = 2;
        }
        printf("%d ", start);
        int end = (x + 1) * (max / thread_number);
        if (x == thread_number-1){
            end = max;
        }
        printf("%d \n", end);


        for (int i = start; i <= end; i++) {
            //Liczby przed zakresem
            if (i < min) {
                if (start_primes[i - 2]) {
                    int remainder = min % i;

                    for (int j = min - remainder; j <= max; j += i) {
                        sieve_eratosthenes[j - min] = 0;
                    }
                }
            }
            //Liczby po zakresie
            else {
                if (sieve_eratosthenes[i - min]) {
                    for (int j = 2 * i; j <= max; j += i) {
                        sieve_eratosthenes[j - min] = 0;
                    }
                }
            }
        }
    }

    if (print_result) {
        print(sieve_eratosthenes, min, max);
    }

    return 0;
}
