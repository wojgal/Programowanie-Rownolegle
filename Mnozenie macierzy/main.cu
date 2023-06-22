#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 16

struct BlockStatus {
    int state;
};

__global__ void matrixMultiplication(float *A, float *B, float *C, BlockStatus *status)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int blockRow = row / BLOCK_SIZE;
    int blockCol = col / BLOCK_SIZE;
    int blockIndex = blockRow * (N / BLOCK_SIZE) + blockCol;

    while (status[blockIndex].state != 2)
    {
        // Oczekiwanie na zakończenie pobierania danych przez inny blok
        __threadfence();
    }

    float sum = 0.0f;
    for (int i = 0; i < N; i++)
    {
        sum += A[row * N + i] * B[i * N + col];
    }

    C[row * N + col] = sum;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        // Aktualizacja stanu bloku wątków
        atomicExch(&status[blockIndex].state, 2);
    }

    // Synchronizacja wszystkich wątków w bloku przed kontynuacją
    __threadfence();
}

void initializeMatrices(float* A, float* B, int size)
{
    srand(time(NULL));

    for (int i = 0; i < size; i++)
    {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main()
{
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    int size = N * N * sizeof(float);

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    initializeMatrices(A, B, N * N);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    BlockStatus *d_status;
    cudaMalloc(&d_status, sizeof(BlockStatus) * (N / BLOCK_SIZE) * (N / BLOCK_SIZE));
    cudaMemset(d_status, 0, sizeof(BlockStatus) * (N / BLOCK_SIZE) * (N / BLOCK_SIZE));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(N / blockSize.x, N / blockSize.y);

    matrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, d_status);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_status);

    return 0;
}
