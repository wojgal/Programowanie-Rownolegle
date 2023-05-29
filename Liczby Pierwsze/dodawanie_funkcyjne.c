#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

//Funkcja wypisujaca wyniki
void print(bool* sieve_eratosthenes, int min, int max){
    int primes_amount = 0;
    int counter = 0;

    printf("Liczby pierwsze dla przedzialu %d - %d:\n\n", min, max);
    for(int i = 0; i <= max - min; i++){
        if(sieve_eratosthenes[i]){
            printf("%d ", i + min);
            primes_amount++;
            counter++;
        }
        if(counter == 10){
            counter = 0;
            printf("\n");
        }
    }

    printf("\n\nIlosc liczb pierwszych w przedziale %d - %d wynosi: %d\n", min, max, primes_amount);
}

void get_start_primes(bool *start_primes, int min){
    #pragma omp parallel for schedule(guided)
    for(int i = 2; i < min; i++){
        start_primes[i - 2] = 1;
    }
    #pragma omp parallel for schedule(dynamic)
    for(int i = 2; i <= (int)sqrt(min); i++){
        if(start_primes[i - 2]){
            for(int j = 2 * i; j < min; j += i){
                start_primes[j - 2] = 0;
            }
        }
    }
}

int main() {
   for(int x=0;x<100;x++){
    int min = 2;
    int max = 100000000;
    int max_sqrt = sqrt(max);

    bool* sieve_eratosthenes = (bool *)calloc((max), sizeof(bool));
    bool* start_primes = (bool *)calloc(min - 2, sizeof(bool));
    omp_set_num_threads(8);
    //Zmienna odpowiadajca za wypisywanie 
    bool print_result = false;

    if(min > 2){
        //start_primes = (bool *)calloc(min - 2, sizeof(bool));
        get_start_primes(start_primes, min);
    }

    //Ustawienie sita na wartosci 1
    #pragma omp parallel for schedule(guided) shared(sieve_eratosthenes)
    for(int i = min; i <= max; i++){
        sieve_eratosthenes[i-min] = 1;
    }
    #pragma omp parallel for schedule(dynamic) shared(sieve_eratosthenes)
    for(int i = 2; i <= max_sqrt; i++){
        //Liczby przed zakresem
        if(i < min){
            if(start_primes[i - 2]){
                int remainder = min % i;

                for(int j = min - remainder; j <= max; j += i){
                    sieve_eratosthenes[j - min] = 0;
                }
            }
        }
        //Liczby po zakresie
        else{
            if(sieve_eratosthenes[i - min]){
                for(int j = 2*i; j <= max; j += i){
                    sieve_eratosthenes[j - min] = 0;
                }
            }
        }
    }
    
    if(print_result){
        print(sieve_eratosthenes, min, max);
    }

    free(sieve_eratosthenes);
    free(start_primes);
}
    return 0;
}
