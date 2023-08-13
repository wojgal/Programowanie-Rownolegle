#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void calculateOutputParallel(const int* input, int* output, int N, int R) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int outputSize = N - 2 * R;

    if (row >= R && row < N - R && col >= R && col < N - R) {
        int sum = 0;
        for (int r = -R; r <= R; ++r) {
            for (int c = -R; c <= R; ++c) {
                sum += input[(row + r) * N + col + c];
            }
        }
        output[(row - R) * outputSize + (col - R)] = sum;
    }
}

void print_table(int* table, int tab_size) {
    for (int i = 0; i < tab_size; i++) {
        std::cout << table[i] << " ";
    }
}

int main() {
    int N = 6; // Rozmiar tablicy
    int R = 2; // Promień
    int inputSize = N * N;
    int outputSize = (N - 2 * R) * (N - 2 * R);
    int* hostInput = new int[inputSize];
    int* hostOutputGPU = new int[outputSize];

    // Wypełnij tablicę wejściową przykładowymi wartościami
    for (int i = 0; i < inputSize; ++i) {
        hostInput[i] = i;
    }

    // Alokuje pamięć na GPU
    int* deviceInput;
    int* deviceOutput;
    cudaMalloc((void**)&deviceInput, inputSize * sizeof(int));
    cudaMalloc((void**)&deviceOutput, outputSize * sizeof(int));

    // Kopiuje dane z CPU do GPU
    cudaMemcpy(deviceInput, hostInput, inputSize * sizeof(int), cudaMemcpyHostToDevice);

    // Konfiguracja wątków i bloków
    dim3 blockSize(16, 16);
    dim3 gridSize((N - 2 * R + blockSize.x - 1) / blockSize.x, (N - 2 * R + blockSize.y - 1) / blockSize.y);

    // Wywołanie kernela na GPU
    calculateOutputParallel << <gridSize, blockSize >> > (deviceInput, deviceOutput, N, R);

    // Kopiowanie wyników z GPU do CPU
    cudaMemcpy(hostOutputGPU, deviceOutput, outputSize * sizeof(int), cudaMemcpyDeviceToHost);

    print_table(hostOutputGPU, outputSize);

    // Zwolnienie pamięci na GPU
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    // Zwolnienie pamięci na CPU
    delete[] hostInput;
    delete[] hostOutputGPU;

    return 0;
}
