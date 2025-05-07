#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <chrono>

namespace Util {

    __host__   void CudaCheckError(cudaError_t error_code, const char* file, int line);
}

namespace EdgePadding {

    __device__ bool IsZeroPixel(uchar4 p);

    __global__ void MakeMaskPixelKernel(const uchar4* input, uint8_t* mask, int width, int height);

    __global__ void FillZeroPixelKernel(const uchar4* input, const uint8_t* input_mask, uchar4* output, uint8_t* output_mask, int width, int height, int* stillHasZero);

    __host__   int  FillZeroPixels(const uchar4* input, uchar4* output, int wiidth, int height, const uint8_t* mask = nullptr);
}

#define CUDACHECK(err) Util::CudaCheckError((err), __FILE__, __LINE__);
