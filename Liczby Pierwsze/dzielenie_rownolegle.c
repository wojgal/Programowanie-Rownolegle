#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

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

int main(){
    //Tworzenie potrzebnych zmiennych
    int min = 2;
    int max = 100000000;
    int max_sqrt = sqrt(max);
    int primes_amount = 0;

    //Sprawdzanie liczb czy sa pierwsze 
    #pragma omp parallel shared(primes_amount)
    #pragma omp parallel for schedule(guided)
    for(int i = 0; i <= max; i++){
        primes_amount += check_prime(i);
    }

    printf("[Dzielenie Sekwencyjne] Ilosc liczb pierwszych: %d", primes_amount);

    return 0;
}