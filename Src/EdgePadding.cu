#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <chrono>

__device__ bool isZeroPixel(uchar4 p) {
    return (p.x == 0 && p.y == 0 && p.z == 0 && p.w == 0);
}

__global__ void fillZeroPixelsKernel(const uchar4* input, uchar4* output, int width, int height, int* stillHasZero) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uchar4 p = input[idx];

    if (!isZeroPixel(p)) {
        output[idx] = p;
        return;
    }

    int sumX = 0, sumY = 0, sumZ = 0, sumW = 0;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (dx != 0 && dy != 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;

            uchar4 neighbor = input[ny * width + nx];
            if (!isZeroPixel(neighbor)) {
                sumX += neighbor.x;
                sumY += neighbor.y;
                sumZ += neighbor.z;
                sumW += neighbor.w;
                count++;
            }
        }
    }

    if (count > 0) {
        output[idx].x = sumX / count;
        output[idx].y = sumY / count;
        output[idx].z = sumZ / count;
        output[idx].w = sumW / count;
    }
    else {
        output[idx] = p;
        atomicAdd(stillHasZero, 1);
    }
}

int fillZeroPixels(cv::Mat img) {

    int width = img.cols;
    int height = img.rows;

    int zeroCount = INT_MAX;
    int iter = 0;

    size_t imageSize = width * height * sizeof(uchar4);

    uchar4* devImgA;
    uchar4* devImgB;
    int* devZeroCount;

    cudaMalloc(&devImgA, imageSize);
    cudaMalloc(&devImgB, imageSize);
    cudaMalloc(&devZeroCount, sizeof(int));

    // 上传图像到 CUDA
    cudaMemcpy(devImgA, img.ptr(), imageSize, cudaMemcpyHostToDevice);


    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 4096 && zeroCount > 0; i++)
    {
        zeroCount = 0;

        cudaMemcpy(devZeroCount, &zeroCount, sizeof(int), cudaMemcpyHostToDevice);

        fillZeroPixelsKernel << <grid, block >> > (devImgA, devImgB, width, height, devZeroCount);
        cudaDeviceSynchronize();

        cudaMemcpy(&zeroCount, devZeroCount, sizeof(int), cudaMemcpyDeviceToHost);
        //std::cout << "Iteration " << ++iter << ": remaining zero pixels = " << zeroCount << std::endl;

        std::swap(devImgA, devImgB);

        iter = i + 1;
    }

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << iter << " , " << duration.count() << std::endl;

    // 拷回主机查看结果（可选）
    cudaMemcpy(img.ptr(), devImgA, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(devImgA);
    cudaFree(devImgB);
    cudaFree(devZeroCount);

    return 0;
}
