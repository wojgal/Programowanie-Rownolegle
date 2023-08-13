#include <iostream>
//#include <cuda_runtime.h>

// N - dlugosc tablicy
// R - dlugosc promienia zliczania
// BS - wielkosc bloku

void calculate(int* input_tab, int* output_tab, int N, int R){
    int output_tab_size = N - 2 * R;

    if(output_tab_size < 0){
        std::cout << "Error, nie spelnino warunku N > 2R.\n";
        return;
    }

    //i oraz j, sa to iteracje po czesciach tablicy, ktore nadaj sie do zliczenia
    //r oraz c, wspomagaja nawigacje miejsc, ktore sumujemy
    for(int i = R; i < N - R; i++){
        for(int j = R; j < N - R; j++){
            int sum = 0;

            for(int r = -R; r <= R; r++){
                for(int c = -R; c <= R; c++){
                    sum += input_tab[(i + r) * N + j + c];
                }
            }

            output_tab[(i - R) * output_tab_size + j - R] = sum;
        }
    }
}



// Wypelnianie tablicy liczbami 0 - 100
void fill_table(int* table, int tab_size){
    for(int i = 0; i < tab_size; i++){
        table[i] = i % 100;
    }
}



void print_table(int* table, int tab_size){
    for(int i = 0; i < tab_size; i++){
        std::cout << table[i] << " ";
    }
}



int main(){
    const int N = 6;
    const int R = 2;

    const int input_tab_size = N * N;
    const int output_tab_size = (N - 2 * R) * (N - 2 * R);
    
    int input_tab[input_tab_size];
    int output_tab[output_tab_size];

    fill_table(input_tab, input_tab_size);

    calculate(input_tab, output_tab, N, R);

    print_table(output_tab, output_tab_size);

    return 0;
}
