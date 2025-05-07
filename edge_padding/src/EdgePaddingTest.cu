#include <EdgePaddingLib.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <chrono>

int test_make_mask(std::string input_filename, std::string save_filename) {
        
    cv::Mat img = cv::imread(input_filename, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_filename << std::endl;
        return -1;
    }

    // 确保为 4 通道图像
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);
    }

    cv::resize(img, img, cv::Size(2048, 2048));
    std::cout << img.size() << std::endl;

    // 创建2048x2048的mask图像（单通道）
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);

    // 定义三角形的三个点
    std::vector<cv::Point> triangle = {
        cv::Point(512, 512),
        cv::Point(1536, 512),
        cv::Point(1024, 1536)
    };

    // 用fillConvexPoly填充mask上的三角形
    cv::fillConvexPoly(mask, triangle, cv::Scalar(255));

    //mask.at<uchar>(728, 0) = uchar(255);

    //// 使用mask对image做掩码操作，将三角形外区域置零
    //for (int y = 0; y < img.rows; y++) {
    //    for (int x = 0; x < img.cols; x++) {
    //        if (mask.at<uchar>(y, x) == uchar(0)) {
    //            img.at<cv::Vec4b>(y, x) = cv::Vec4b(0, 0, 0, 0); // 全部通道置零
    //        }
    //    }
    //}

    EdgePadding::FillZeroPixels(img.ptr<uchar4>(), img.ptr<uchar4>(), img.cols, img.rows, mask.ptr<uint8_t>());

    cv::Mat showImg;
    cv::resize(img, showImg, cv::Size(1024, 1024));
    cv::imshow("Fixed Image", showImg);
    cv::waitKey(0);

    cv::imwrite(save_filename, img);

    return 0;
}

int main() {

    std::string input_filename = R"(D:\Assets\EdgePadding\pve01_xht_chexiang01_01_d_2.png)";  // 支持 PNG、JPG 等
    std::string output_filename = R"(D:\Assets\EdgePadding\pve01_xht_chexiang01_01_d_3.png)";

    test_make_mask(input_filename, output_filename);

    return 0;
}
