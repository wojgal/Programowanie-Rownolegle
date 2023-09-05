#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define BS 8
#define N 20
#define R 2
#define K 1

// N - dlugosc tablicy
// R - dlugosc promienia zliczania
// BS - wielkosc bloku

__global__ void calculate(const int* input_tab, int* output_tab, int Nx, int Rx, int Kx) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * Kx - 1;

    int output_tab_size = Nx - 2 * Rx;

    int output_row_offset = (row - Rx) * output_tab_size;

    for (int k_iter = 0; k_iter < Kx; k_iter++) {
        col++;

        if (row >= Rx && row < Nx - Rx && col >= Rx && col < Nx - Rx) {
            int sum = 0;

            for (int r = -Rx; r <= Rx; r++) {
                for (int c = -Rx; c <= Rx; c++) {
                    sum += input_tab[(row + r) * Nx + col + c];
                }
            }
            output_tab[output_row_offset + col - Rx] = sum;
        }
    }
}

__global__ void calculateShared(int* input_tab, int* output_tab, int Nx, int Rx, int Kx) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int output_tab_size = Nx - 2 * Rx;
    const int shared_input_tab_size = BS + 2 * R + 1;

    if (col < output_tab_size && row < output_tab_size) {
        __shared__ int shared_input_tab[shared_input_tab_size][shared_input_tab_size];

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            for (int i = 0; i < shared_input_tab_size; i++) {
                for (int j = 0; j < shared_input_tab_size; j++) {
                    shared_input_tab[i][j] = input_tab[(col + i) * Nx + row + j];
                }
            }
        }
        __syncthreads();

        int sum = 0;
        for (int j = 0; j < 2 * Rx + 1; j++) {
            for (int i = 0; i < 2 * Rx + 1; i++) {
                sum += shared_input_tab[(threadIdx.x + i)][threadIdx.y + j];
            }
        }
        output_tab[(col) * (output_tab_size)+(row)] = sum;

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
    calculateShared << <gridSize, blockSize >> > (device_input, device_output, N, R, K);

    cudaDeviceSynchronize();

    print_table(device_output, output_tab_size);

     // Zwolnienie pamięci na GPU
    cudaFreeHost(device_input);
    cudaFreeHost(device_output);

    return 0;
}
