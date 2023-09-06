#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define BS 16
#define N 50
#define R 20
#define K 3

// N - dlugosc tablicy
// R - dlugosc promienia zliczania
// BS - wielkosc bloku

__global__ void calculateGlobal(int* input_tab, int* output_tab, int Nx, int Rx, int Kx) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * Kx - 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int output_tab_size = Nx - 2 * Rx;

    for (int k_iter = 0; k_iter < Kx; k_iter++) {
        row++;

        if (col < output_tab_size && row < output_tab_size) {
            int sum = 0;

            // Zliczanie sumy elementów w zasięgu promienia R
            for (int i = -Rx; i <= Rx; i++) {
                for (int j = -Rx; j <= Rx; j++) {
                    sum += input_tab[(row + j + Rx) * Nx + col + Rx + i];
                }
            }

            // Zapisywanie wyników sum do tablicy wynikowej
            output_tab[row * output_tab_size + col] = sum;
        }
    }
}

__global__ void calculateShared(int* input_tab, int* output_tab, int Nx, int Rx, int Kx) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int output_tab_size = Nx - 2 * Rx;
    int calculation_radius_range = 2 * Rx + 1;
    const int shared_input_tab_size = BS + 2 * R + 1;

    for (int k_iter = 0; k_iter < Kx; k_iter++) {
        int row_offset = row;
        int col_offset = col + k_iter * output_tab_size;

        if (col_offset < output_tab_size && row_offset < output_tab_size) {
            // Inicjalizacja tablicy pamięci współdzielonej
            __shared__ int shared_input_tab[shared_input_tab_size * shared_input_tab_size];

            // Wczytanie danych do pamięci współdzielonej przez wątek (0, 0)
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                for (int i = 0; i < shared_input_tab_size; i++) {
                    for (int j = 0; j < shared_input_tab_size; j++) {
                        shared_input_tab[i * shared_input_tab_size + j] = input_tab[(row_offset + i) * Nx + col_offset + j];
                    }
                }
            }

            // Synchronizacja wątków po wczytaniu danych do pamięci współdzielonej
            __syncthreads();

            // Zliczanie sumy elementów w zasięgu promienia R
            int sum = 0;

            for (int i = 0; i < calculation_radius_range; i++) {
                for (int j = 0; j < calculation_radius_range; j++) {
                    sum += shared_input_tab[(threadIdx.y + i) * shared_input_tab_size + threadIdx.x + j];
                }
            }

            // Zapisywanie wyników sum do tablicy wynikowej
            output_tab[row_offset * output_tab_size + col_offset] = sum;
        }
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
