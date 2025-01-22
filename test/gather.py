import torch
import ctypes
import numpy as np
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

lib.launchGatherKernel.argtypes = [
    ctypes.c_int64,
    ctypes.c_int64,  # axis
    ctypes.c_void_p,  # inputTensor (device pointer)
    ctypes.c_void_p,  # indexTensor (device pointer)
    ctypes.c_void_p,  # outputTensor (device pointer)
    ctypes.c_int64,  # shape
    ctypes.c_int64,  # strides
    ctypes.c_int64,  # output_size
    ctypes.c_int64,  # num_indices
    ctypes.c_void_p,  # cudaStream_t
    ctypes.c_char_p  # dtype
]
def gather(rank, axis, inputTensor, indexTensor):
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    outTensor = inputTensor[tuple(indices)]
    return outTensor


def custom_gather(rank, axis, input_ptr, index_ptr, output_ptr, shape, strides, output_size, num_indices, stream, dtype):
    lib.launchGatherKernel(
        ctypes.c_int64(rank),
        ctypes.c_int64(axis),
        ctypes.c_void_p(input_ptr),
        ctypes.c_void_p(index_ptr),
        ctypes.c_void_p(output_ptr),
        ctypes.c_int64(shape),
        ctypes.c_int64(strides),
        ctypes.c_int64(output_size),
        ctypes.c_int64(num_indices),
        stream,
        dtype.encode('utf-8')  # Convert Python string to C-style string
    )

def test(inputShape, indexShape, axis, test_dtype, device):
    print(
        f"Testing Gather on {device} with x_shape:{inputShape} , indice_shape:{indexShape}, axis:{axis} ,dtype:{test_dtype}"
    )
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)

    index = np.random.randint(0, inputShape[axis], indexShape).astype(np.int32)
    indexTensor = torch.from_numpy(index).to(torch.int64).to(device)

    rank = len(inputShape)
    outTensor = gather(rank, axis, inputTensor, indexTensor)#

    Q_output = torch.zeros(outTensor.shape, device=device, dtype=test_dtype)

    shape = inputTensor.shape[axis]
    strides = inputTensor.stride()[axis]


    input_ptr = inputTensor.data_ptr()
    index_ptr = indexTensor.data_ptr()
    output_ptr = Q_output.data_ptr()

    output_size = Q_output.numel()
    num_indices = indexTensor.numel()

    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    if test_dtype == torch.float32:
        dtype_str = "float"
        if device == "cuda":
            torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))
            custom_gather_time = performance.CudaProfile((custom_gather, (rank, axis, input_ptr, index_ptr, output_ptr, shape, strides, output_size, num_indices, stream, dtype_str)))
    if test_dtype == torch.float16:
        if device == "cuda":
            dtype_str = "half"
            torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))
            custom_gather_time = performance.CudaProfile((custom_gather, (rank, axis, input_ptr, index_ptr, output_ptr, shape, strides, output_size, num_indices, stream, dtype_str)))
    performance.logBenchmark(torch_gather_time, custom_gather_time)
    tmpa = outTensor.to('cpu').numpy().flatten()
    tmpb = Q_output.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
parser = argparse.ArgumentParser(description="Test softmax on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        ((3, 2), (2, 2), 0, torch.float32, "cuda"),
        ((3, 2), (1, 2), 1, torch.float32, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float32, "cuda"),
        ((2, 3, 4, 5), (2, 4), 2, torch.float32, "cuda"),

        ((3, 2), (2, 2), 0, torch.float16, "cuda"),
        ((3, 2), (1, 2), 1, torch.float16, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float16, "cuda"),
        ((2, 3, 4, 5), (2, 4), 2, torch.float16, "cuda"),
]
filtered_test_cases = [
    (inputShape , indexShape, axis, test_dtype, device)
    for inputShape , indexShape, axis, test_dtype, device in test_cases
    if device == args.device
]

if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for inputShape , indexShape, axis, test_dtype, device in filtered_test_cases:
    test(inputShape , indexShape, axis, test_dtype, device)