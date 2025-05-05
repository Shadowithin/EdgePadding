#include "EdgePaddingLib.h"

namespace EdgePadding {

    __device__ bool isZeroPixel(uchar4 p) {
        return (p.x == 0 && p.y == 0 && p.z == 0 && p.w == 0);
    }

    __global__ void fillZeroPixelsKernel(const uchar4* input, const uint8_t* input_mask, uchar4* output, uint8_t* output_mask, int width, int height, int* stillHasZero) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        int idx = y * width + x;
        uchar4 p = input[idx];
        uint8_t m = input_mask[idx];

        if (m > 0) {
            output[idx] = p;
            output_mask[idx] = m;
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
                uint8_t neighbor_mask = input_mask[ny * width + nx];
                if (neighbor_mask > 0) {
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
            output_mask[idx] = 255;
        }
        else {
            output[idx] = p;
            output_mask[idx] = m;
            atomicAdd(stillHasZero, 1);
        }
    }

    __host__ int fillZeroPixels(const uchar4* input, const uint8_t* input_mask, uchar4* output, int width, int height) {

        int zeroCount = INT_MAX;
        int iter = 0;
        int maxIter = width + height;

        size_t imageSize = width * height * sizeof(uchar4);
        size_t maskSize = width * height * sizeof(uint8_t);

        uchar4* devImgA;
        uchar4* devImgB;
        uint8_t* devMaskA;
        uint8_t* devMaskB;
        int* devZeroCount;

        cudaMalloc(&devImgA, imageSize);
        cudaMalloc(&devImgB, imageSize);
        cudaMalloc(&devMaskA, maskSize);
        cudaMalloc(&devMaskB, maskSize);
        cudaMalloc(&devZeroCount, sizeof(int));

        // 上传图像到 CUDA
        cudaMemcpy(devImgA, input, imageSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devMaskA, input_mask, maskSize, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < maxIter && zeroCount > 0; i++)
        {
            zeroCount = 0;

            cudaMemcpy(devZeroCount, &zeroCount, sizeof(int), cudaMemcpyHostToDevice);

            EdgePadding::fillZeroPixelsKernel << <grid, block >> > (devImgA, devMaskA, devImgB, devMaskB, width, height, devZeroCount);
            cudaDeviceSynchronize();

            cudaMemcpy(&zeroCount, devZeroCount, sizeof(int), cudaMemcpyDeviceToHost);
            //std::cout << "Iteration " << ++iter << ": remaining zero pixels = " << zeroCount << std::endl;

            std::swap(devImgA, devImgB);
            std::swap(devMaskA, devMaskB);

            iter = i + 1;
        }

        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << iter << " , " << duration.count() << std::endl;

        // 拷回主机查看结果
        cudaMemcpy(output, devImgA, imageSize, cudaMemcpyDeviceToHost);

        cudaFree(devImgA);
        cudaFree(devImgB);
        cudaFree(devMaskA);
        cudaFree(devMaskB);
        cudaFree(devZeroCount);

        return 0;
    }
}

