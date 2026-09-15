#include <cuda_runtime.h>
#include <math.h>

__global__ void cal_mean(const float* input, float* mean, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float local = 0.0f;
        for (int j = 0; j < N; j++) {
            local += input[i * N + j];
        }
        mean[i] = local / N;
    }
}

__global__ void variance(const float* input, float* var, float* mean, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float local = 0.0f;
        float temp;
        for (int j = 0; j < N; j++) {
            temp = input[i * N + j] - mean[i];
            local += temp * temp;
        }
        var[i] = local / N;
    }
}

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, float* mean, float* variance, int M, int N, float eps) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        output[i * N + j] = ((input[i * N + j] - mean[i]) / sqrtf(variance[i] + eps))*gamma[j] + beta[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    dim3 threads(16,16);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    float* mean;
    cudaMalloc(&mean, M*sizeof(float));
    cal_mean<<<blocks, threads>>>(input, mean, M, N);
    
    float* var;
    cudaMalloc(&var, M*sizeof(float));
    variance<<<blocks, threads>>>(input, var, mean, M, N);
    
    
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, mean, var, M, N, eps);
    cudaDeviceSynchronize();
}
