#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

Mat RGB2YCbCr(Mat src);
Mat YCbCr2RGB(Mat src);