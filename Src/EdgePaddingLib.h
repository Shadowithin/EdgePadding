#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <chrono>

namespace EdgePadding {

    __device__ bool isZeroPixel(uchar4 p);

    __global__ void fillZeroPixelsKernel(const uchar4* input, uchar4* output, int width, int height, int* stillHasZero);

    __host__   int fillZeroPixels(const uchar4* input,  uchar4* output, int wiidth, int height);
}
