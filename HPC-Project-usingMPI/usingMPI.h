#pragma once
#include <mpi.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat MPIHighPassFilterRGB(const Mat& inputImage, int kernelSize);
