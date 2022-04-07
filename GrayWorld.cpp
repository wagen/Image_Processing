#include "GrayWorld.h"

Mat grayWorld(Mat src)
{
	int rows = src.rows;
	int cols = src.cols;
	double blue = 0;
	double green = 0;
	double red = 0;
	Mat dst(rows, cols, CV_8UC3);

#pragma omp parallel for num_threads(4)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++){
			blue += src.at<Vec3b>(i, j)[0];
			green += src.at<Vec3b>(i, j)[1];
			red += src.at<Vec3b>(i, j)[2];
		}
	}

	blue /= (rows * cols);
	green /= (rows * cols);
	red /= (rows * cols);

	double gray = (blue + green + red) / 3;
	double kb = gray / blue;
	double kg = gray / green;
	double kr = gray / red;

#pragma omp parallel for num_threads(4)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			dst.at<Vec3b>(i, j)[0] = (int)(src.at<Vec3b>(i, j)[0] * kb);
			dst.at<Vec3b>(i, j)[1] = (int)(src.at<Vec3b>(i, j)[1] * kg);
			dst.at<Vec3b>(i, j)[2] = (int)(src.at<Vec3b>(i, j)[2] * kr);
		}
	}

	return dst;
}
