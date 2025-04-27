#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int applyKernelAtPixel(const Mat& paddedImage, const vector<vector<int>>& kernel, int x, int y, int padding);
vector<vector<int>> generateKernel(int k);
void printKernel(const vector<vector<int>>& kernel);

int applyKernelAtPixelRGB(const Mat& paddedImage, const vector<vector<int>>& kernel,
    int centerX, int centerY, int padding, int channel);
