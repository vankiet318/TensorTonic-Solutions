#include <cuda_runtime.h>
#include <float.h>

__global__ void reduce_max(const float* input, float* output, int N) {
    __shared__ float temp[256];
    int idx = threadIdx.x;
    float v = -FLT_MAX;
    
    for (int i = idx; i < N; i += blockDim.x) {
        v = fmaxf(v, input[i]);
    }

    temp[idx] = v;

    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (idx < st) {
            temp[idx] = fmaxf(temp[idx], temp[idx + st]);
        }
        __syncthreads();
    }
    if (idx == 0) {
        *output = temp[0];
    }
}

__global__ void exp_sum(const float* input, float* maxv, float* sum, float* output, int N) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    float max = *maxv;
    float local = 0.0f;
    for (int i = tid; i < N; i+= blockDim.x) {
        float e = expf(input[i] - max);
        output[i] = e;
        local += e;
    }
    temp[tid] = local;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) {
            temp[tid] += temp[st + tid];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *sum = temp[0];
    }
}

__global__ void softmax_kernel( const float* input, float* output, float* sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] /= *sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float *d_max, *d_sum;
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    
    reduce_max<<<1, threads>>>(input, d_max, N);
    exp_sum<<<1, threads>>>(input, d_max, d_sum, output, N);
    softmax_kernel<<<blocks, threads>>>(input, output, d_sum, N);
    
    cudaDeviceSynchronize();
    cudaFree(d_max);
    cudaFree(d_sum);
}