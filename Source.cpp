#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include "RGBYCbCr.h"
#include "GrayWorld.h"
#include "Source.h"

using namespace cv;
using namespace std;

Mat rgb2gray(Mat src) {
	Mat dst(src.rows, src.cols, CV_8U);

#pragma omp parallel for num_threads(4)
	/*
	//faster version
	for (int i = 0; i < src.rows; i++) {
		Vec3b* Mi = src.ptr<Vec3b>(i);
		uchar* Di = dst.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++) {
			Vec3b& Mij = Mi[j];
			Di[j] = ((Mij[0] << 2) + (Mij[0]) +
				(Mij[1] << 1) +
				(Mij[2])) >> 3;
		}
	}
	*/
	
	//at<T> does a range check at every call, thus making it slower than ptr<T>, but safer
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<uchar>(i, j) = ((src.at<Vec3b>(i, j)[0] << 2) + (src.at<Vec3b>(i, j)[0]) +
				(src.at<Vec3b>(i, j)[1] << 1) +
				(src.at<Vec3b>(i, j)[2])) >> 3;
		}
	}
	
	return dst;
}

Mat historgramEqualization(Mat src) {
	int intensity[256] = { 0 };

	int rows = src.rows;
	int cols = src.cols;

#pragma omp parallel for num_threads(4)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			intensity[src.at<uchar>(i, j)]++;
		}
	}

	double cum = 0;
	for (int i = 0; i < 256; i++) {
		cum += intensity[i];
		intensity[i] = cum / (rows* cols) * 255;
	}

	Mat dst(rows, cols, CV_8U);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			dst.at<uchar>(i, j) = intensity[src.at<uchar>(i, j)];
		}
	}
	return dst;
}

int main() {
	/*Mat src = imread("vlcsnap-2018-10-22-02h02m26s122.png");
	imshow("Gray", rgb2gray(src));
	imshow("HistogramEqualize", historgramEqualization(rgb2gray(src)));
	imshow("YCbCr", RGB2YCbCr(src));
	namedWindow("GrayWorld", WINDOW_NORMAL);
	imshow("GrayWorld", grayWorld(src));
	waitKey(0);*/

	Mat frame;
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "Cannot open the web cam" << endl;
		return 1;
	}
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);

	for (;;) {
		cap >> frame;
		if (frame.empty()) {
			cout << "Can't receive frame..." << endl;
			break;
		}
		//imshow("Camera", frame);
		imshow("GrayWorld", rgb2gray(frame));

		if (waitKey(1) == 'q') {
			break;
		}
	}

	return 0;
 }