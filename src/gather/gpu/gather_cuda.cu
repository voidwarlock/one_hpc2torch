#include <cuda.h>
#include <cub/cub.cuh>
#include <cstdio> 
#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <mma.h>
#include <string.h>
#include <complex.h>

struct KernelParams {
    int64_t rank;
    int64_t axis;
    int64_t shape_axis;
    int64_t stride_axis;
    int64_t num_indices;
};

template<typename T>
__global__ void gather_kernel(const KernelParams params, const T* __restrict__ inputTensor, const int64_t* __restrict__ indexTensor, T* __restrict__ outputTensor) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    int64_t index = __ldg(&indexTensor[(idx / params.stride_axis) % params.num_indices]);

    int64_t linearIdx = params.shape_axis * params.stride_axis * (idx / (params.num_indices * params.stride_axis)) + index * params.stride_axis + idx % params.stride_axis;

    outputTensor[idx] = inputTensor[linearIdx];

}


__global__ void gather_kernel_float(const KernelParams params, const float* __restrict__ inputTensor, const int64_t* __restrict__ indexTensor, float* __restrict__ outputTensor) {

    if(params.axis != params.rank - 1)
    {
        const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < params.num_indices * params.stride_axis / 2) {

            int64_t index = __ldg(&indexTensor[(idx * 2 / params.stride_axis) % params.num_indices]);

            int64_t linearIdx = params.shape_axis * params.stride_axis * (idx * 2 / (params.num_indices * params.stride_axis)) + index * params.stride_axis + (idx * 2) % params.stride_axis;

            const float2* input = reinterpret_cast<const float2*>(inputTensor);
            float2* output = reinterpret_cast<float2*>(outputTensor);

            output[idx] = input[linearIdx / 2];
        }
    }
    else
    {
        const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        int64_t index = __ldg(&indexTensor[(idx / params.stride_axis) % params.num_indices]);

        int64_t linearIdx = params.shape_axis * params.stride_axis * (idx / (params.num_indices * params.stride_axis)) + index * params.stride_axis + idx % params.stride_axis;

        outputTensor[idx] = inputTensor[linearIdx];

    }
}

__global__ void gather_kernel_half(const KernelParams params, const __half* __restrict__ inputTensor, const int64_t* __restrict__ indexTensor, __half* __restrict__ outputTensor) {

    if(params.axis != params.rank - 1)
    {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < params.num_indices * params.stride_axis / 2) {

            int64_t index = __ldg(&indexTensor[(idx * 2 / params.stride_axis) % params.num_indices]);

            int64_t linearIdx = params.shape_axis * params.stride_axis * (idx * 2 / (params.num_indices * params.stride_axis)) + index * params.stride_axis + (idx * 2) % params.stride_axis;

            const __half2* input = reinterpret_cast<const __half2*>(inputTensor);
            __half2* output = reinterpret_cast<__half2*>(outputTensor);

            output[idx] = input[linearIdx / 2];
        }
    }
    else
    {
        const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        int64_t index = __ldg(&indexTensor[(idx / params.stride_axis) % params.num_indices]);

        int64_t linearIdx = params.shape_axis * params.stride_axis * (idx / (params.num_indices * params.stride_axis)) + index * params.stride_axis + idx % params.stride_axis;

        outputTensor[idx] = inputTensor[linearIdx];

    }

}

extern "C" void launchGatherKernel(
        const int64_t rank,
        const int64_t axis,
        void* inputTensor,
        const int64_t* indexTensor,
        void* outputTensor,
        const int64_t shape_axis,
        const int64_t stride_axis,
        int64_t output_size,
        int64_t num_indices,
        cudaStream_t stream = 0,
        const char* dtype = "float"
    ) {

        KernelParams params = {rank, axis, shape_axis, stride_axis, num_indices};
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;


        if (strcmp(dtype, "float") == 0){
            gather_kernel_float<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float), stream>>>(params, static_cast<float*>(inputTensor), indexTensor, static_cast<float*>(outputTensor));
        }
        else if (strcmp(dtype, "half") == 0) {
            gather_kernel_half<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(__half), stream>>>(params, static_cast<__half*>(inputTensor), indexTensor, static_cast<__half*>(outputTensor));
        }
        else if (strcmp(dtype, "double") == 0) {
            gather_kernel<double><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double), stream>>>(params, static_cast<double*>(inputTensor), indexTensor, static_cast<double*>(outputTensor));
        }
        else if (strcmp(dtype, "int32") == 0) {
            gather_kernel<int32_t><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int32_t), stream>>>(params, static_cast<int32_t*>(inputTensor), indexTensor, static_cast<int32_t*>(outputTensor));
        }
        else if (strcmp(dtype, "int64") == 0) {
            gather_kernel<int64_t><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int64_t), stream>>>(params, static_cast<int64_t*>(inputTensor), indexTensor, static_cast<int64_t*>(outputTensor));
        }
        else if (strcmp(dtype, "int8") == 0) {
            gather_kernel<int8_t><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int8_t), stream>>>(params, static_cast<int8_t*>(inputTensor), indexTensor, static_cast<int8_t*>(outputTensor));
        }
        else if (strcmp(dtype, "int16") == 0) {
            gather_kernel<int16_t><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int16_t), stream>>>(params, static_cast<int16_t*>(inputTensor), indexTensor, static_cast<int16_t*>(outputTensor));
        }
        else if (strcmp(dtype, "uint8") == 0) {
            gather_kernel<uint8_t><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(uint8_t), stream>>>(params, static_cast<uint8_t*>(inputTensor), indexTensor, static_cast<uint8_t*>(outputTensor));
        }
        else if (strcmp(dtype, "uint16") == 0) {
            gather_kernel<uint16_t><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(uint16_t), stream>>>(params, static_cast<uint16_t*>(inputTensor), indexTensor, static_cast<uint16_t*>(outputTensor));
        }
        else if (strcmp(dtype, "bool") == 0) {
            gather_kernel<bool><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(bool), stream>>>(params, static_cast<bool*>(inputTensor), indexTensor, static_cast<bool*>(outputTensor));
        }

    
}