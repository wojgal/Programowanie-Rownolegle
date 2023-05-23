#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

//Funkcja sprawdzajaca dzieleniem, czy liczba jest pierwsza
int check_prime(int number){
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
    int primes_amount = 0;

    //Sprawdzanie liczb czy sa pierwsze 
    #pragma omp parallel for schedule(guided) reduction(+:primes_amount)
    for(int i = min; i <= max; i++){
        primes_amount += check_prime(i);
    }

    printf("[Dzielenie Rownolegle] Ilosc liczb pierwszych: %d", primes_amount);

    return 0;
}
