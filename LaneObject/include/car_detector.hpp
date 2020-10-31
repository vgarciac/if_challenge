#pragma once 

#include <iostream>
#include <string>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "./clasifier.hpp"

const int BOARD_W = 9;
const int BOARD_H = 6;
const int WINDOW_SIZE = 100;
const float WINDOW_OVERLAP = 0.2;
const float SQUARE_SIZE = 0.1;
const float ROI_TOP = 0.55;
const float ROI_BOT = 0.30;
const int N_MASKS = 7;
const float MAS_SUM = 0.05;
const float MASK_THRESH = 255.0*0.3;
const int MORPH_K_CAR = 3;

using namespace cv;
using namespace std;
using namespace cv::ml;

class CarsDetector
{
    public:
    vector<vector<Mat>> labeled_imgs;
    vector<vector<Point>> cars_contours;
    vector<Vec4i> hierarchy;
    vector<Mat> masks;
    Mat cumul_masks;
    Mat current_frame;
    Mat camera_matrix;
    Mat dist_coeff;
    Ptr<SVM> svm;

    CarsDetector(){};

    void FeedImage(Mat image_);

    void Compute();

    bool LoadCameraMatrix(String file_);

    bool LoadSVMModel(String file_);

    void UndistortImage();

    void SlideWindowsSearch();

    void UpdateMAsks();

    void DrawCars(Mat &_img);
};