#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

//Funkcja sprawdzajaca dzieleniem, czy liczba jest pierwsza
int check_prime(int number){
    if(number < 2){
        return 0;
    }

    for(int i = 2; i <= sqrt(number); i++){
        if(number % i == 0){
            return 0;
        }
    }

    return 1;
}

//Funkcja wypisujaca wyniki
void print(bool* sieve_eratosthenes, int min, int max){
    int primes_amount = 0;
    int counter = 0;

    printf("Liczby pierwsze dla przedzialu %d - %d:\n\n", min, max);
    for(int i = 0; i <= max - min + 1; i++){
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

int main(){
    //Tworzenie potrzebnych zmiennych
    int min = 2;
    int max = 100000000;
    bool* sieve_eratosthenes = (bool *)calloc(max-min+1, sizeof(bool));

    //Zmienna odpowiadajca za wypisywanie 
    bool print_result = false;

    //Sprawdzanie liczb czy sa pierwsze 
    for(int i = min; i <= max; i++){
        sieve_eratosthenes[i - min] = check_prime(i)
    }

    if(print_result){
        print(sieve_eratosthenes, min, max);
    }

    return 0;
}
