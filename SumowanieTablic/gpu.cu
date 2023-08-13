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
    const int N = 8;
    const int R = 1;

    const int input_tab_size = N * N;
    const int output_tab_size = (N - 2 * R) * (N - 2 * R);

    int* host_input = new int[input_tab_size];
    int* host_output = new int[output_tab_size];

    fill_table(host_input, input_tab_size);

    // Alokuje pamięć na GPU
    int* device_input;
    int* device_output;
    cudaMalloc((void**)&device_input, input_tab_size * sizeof(int));
    cudaMalloc((void**)&device_output, output_tab_size * sizeof(int));

    // Kopiuje dane z CPU do GPU
    cudaMemcpy(device_input, host_input, input_tab_size * sizeof(int), cudaMemcpyHostToDevice);

    // Konfiguracja wątków i bloków
    dim3 blockSize(16, 16);
    dim3 gridSize((N - 2 * R + blockSize.x - 1) / blockSize.x, (N - 2 * R + blockSize.y - 1) / blockSize.y);

    // Wywołanie kernela na GPU
    calculate<<<gridSize, blockSize>>>(device_input, device_output, N, R);

    // Kopiowanie wyników z GPU do CPU
    cudaMemcpy(host_output, device_output, output_tab_size * sizeof(int), cudaMemcpyDeviceToHost);

    print_table(host_output, output_tab_size);

    // Zwolnienie pamięci na GPU
    cudaFree(device_input);
    cudaFree(device_output);

    // Zwolnienie pamięci na CPU
    delete[] host_input;
    delete[] host_output;

    return 0;
}
