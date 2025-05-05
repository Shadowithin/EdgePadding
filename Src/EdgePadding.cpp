#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <chrono>

const int WIDTH = 2048;
const int HEIGHT = 2048;

extern int fillZeroPixels(cv::Mat img);

int main() {

    std::string filename = R"(D:\Assets\EdgePadding\pve01_xht_chexiang01_01_d_2.png)";  // 支持 PNG、JPG 等
    cv::Mat img = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return -1;
    }

    cv::resize(img, img, cv::Size(WIDTH, HEIGHT));

    // 确保为 4 通道图像
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);
    }
    // 1. 创建2048x2048的mask图像（单通道）
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);

    // 2. 定义三角形的三个点
    std::vector<cv::Point> triangle = {
        cv::Point(512, 512),
        cv::Point(1536, 512),
        cv::Point(1024, 1536)
    };

    // 3. 用fillConvexPoly填充mask上的三角形
    cv::fillConvexPoly(mask, triangle, cv::Scalar(255));

    //mask.at<uchar>(728, 0) = uchar(255);

    // 5. 使用mask对image做掩码操作，将三角形外区域置零
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (mask.at<uchar>(y, x) == uchar(0)) {
                img.at<cv::Vec4b>(y, x) = cv::Vec4b(0, 0, 0, 0); // 全部通道置零
            }
        }
    }

    fillZeroPixels(img);

    cv::Mat showImg;
    cv::resize(img, showImg, cv::Size(1024, 1024));
    cv::imshow("Fixed Image", showImg);
    //cv::imwrite("fixed_output.png", img);
    cv::waitKey(0);

    return 0;
}
