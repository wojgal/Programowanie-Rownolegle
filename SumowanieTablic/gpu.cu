#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// N - dlugosc tablicy
// R - dlugosc promienia zliczania
// BS - wielkosc bloku

__global__ void calculate(const int* input_tab, int* output_tab, int N, int R) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int output_tab_size = N - 2 * R;

    if (row >= R && row < N - R && col >= R && col < N - R) {
        int sum = 0;

        for (int r = -R; r <= R; ++r) {
            for (int c = -R; c <= R; ++c) {
                sum += input_tab[(row + r) * N + col + c];
            }
        }

        output_tab[(row - R) * output_tab_size + (col - R)] = sum;
    }
    else{
        std::cout << "Error, nie spelnino warunku N > 2R.\n";
        return;
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




int main() {
    const int N = 6;
    const int R = 2;

    const int input_tab_size = N * N;
    const int output_tab_size = (N - 2 * R) * (N - 2 * R);

    // Alokuje pamięć na GPU
    int* device_input;
    int* device_output;

    cudaHostAlloc((void**)&device_input, input_tab_size * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&device_output, output_tab_size * sizeof(int), cudaHostAllocMapped);

    fill_table(device_input, input_tab_size);

    // Konfiguracja wątków i bloków
    dim3 blockSize(16, 16);
    dim3 gridSize((N - 2 * R + blockSize.x - 1) / blockSize.x, (N - 2 * R + blockSize.y - 1) / blockSize.y);

    // Wywołanie kernela na GPU
    calculate<<<gridSize, blockSize>>>(device_input, device_output, N, R);

    print_table(device_output, output_tab_size);

    // Zwolnienie pamięci na GPU
    cudaFreeHost(device_input);
    cudaFreeHost(device_output);

    return 0;
}
