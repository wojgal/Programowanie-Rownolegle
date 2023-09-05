#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define BS 32
#define N 500
#define R 20
#define K 1

// N - dlugosc tablicy
// R - dlugosc promienia zliczania
// BS - wielkosc bloku

__global__ void calculateGlobal(int* input_tab, int* output_tab, int Nx, int Rx, int Kx) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int output_tab_size = Nx - 2 * Rx;

    if (col < output_tab_size && row < output_tab_size) {
        int sum = 0;

        // Zliczanie sumy elementów w zasięgu promienia R
        for (int i = -Rx; i <= Rx; i++) {
            for (int j = -Rx; j <= Rx; j++) {
                sum += input_tab[(col + j + Rx) * Nx + row + Rx + j];
            }
        }

        // Zapisywanie wyników sum do tablicy wynikowej
        output_tab[col * output_tab_size + row] = sum;
    }
}

__global__ void calculateShared(int* input_tab, int* output_tab, int Nx, int Rx, int Kx) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int output_tab_size = Nx - 2 * Rx;
    const int shared_input_tab_size = BS + 2 * R + 1;
    int calculation_radius_range = 2 * Rx + 1;

    if (col < output_tab_size && row < output_tab_size) {
        __shared__ int shared_input_tab[shared_input_tab_size][shared_input_tab_size];

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            for (int i = 0; i < shared_input_tab_size; i++) {
                for (int j = 0; j < shared_input_tab_size; j++) {
                    shared_input_tab[i][j] = input_tab[(col + i) * Nx + row + j];
                }
            }
        }

        //Synchronizacja wątków po wczytaniu danych do pamięci współdzielonej
        __syncthreads();

        int sum = 0;
        for (int i = 0; i < calculation_radius_range; i++) {
            for (int j = 0; j < calculation_radius_range; j++) {
                sum += shared_input_tab[threadIdx.x + j][threadIdx.y + i];
            }
        }
        output_tab[col * output_tab_size + row] = sum;

    }
}

// Wypelnianie tablicy liczbami 0 - 100
void fill_table(int* table, int tab_size) {
    for (int i = 0; i < tab_size; i++) {
        table[i] = i % 100;
    }
}



void print_table(int* table, int tab_size) {
    for (int i = 0; i < tab_size; i++) {

        if (i % int(sqrt(tab_size)) == 0) {
            std::cout << "\n";
        }
        std::cout << table[i] << " ";
    }
}




int main() {
    const int input_tab_size = N * N;
    const int output_tab_size = (N - 2 * R) * (N - 2 * R);

    // Alokuje pamięć na GPU
    int* device_input;
    int* device_output;

    cudaHostAlloc((void**)&device_input, input_tab_size * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&device_output, output_tab_size * sizeof(int), cudaHostAllocMapped);

    fill_table(device_input, input_tab_size);

    // Konfiguracja wątków i bloków
    dim3 blockSize(BS, BS);
    dim3 gridSize((N - 2 * R + blockSize.x - 1) / blockSize.x, (N - 2 * R + blockSize.y - 1) / blockSize.y);

    // Wywołanie kernela na GPU
    calculateGlobal <<<gridSize, blockSize>>> (device_input, device_output, N, R, K);

    cudaDeviceSynchronize();

    //print_table(device_output, output_tab_size);

     // Zwolnienie pamięci na GPU
    cudaFreeHost(device_input);
    cudaFreeHost(device_output);

    return 0;
}
