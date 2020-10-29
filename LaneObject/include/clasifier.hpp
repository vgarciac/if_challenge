#pragma once 
#include <iostream>
#include <string>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>

#define HOG_SZ 16

using namespace cv;
using namespace std;

enum LABELS
{
    CARS,
    NO_CARS,
    ALL
};

void show(Mat _img, int ms = 0);

Mat GetHOGFeatures(Mat _img);

Mat GetColorHistogramFeatures(Mat _img);

Mat GetBinnedFeatures(Mat _img);

int GetFeatureVector(Mat _img, Mat &_vector);

Mat GenerateLabel(Mat &_vector, bool _label);
