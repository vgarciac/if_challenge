#pragma once 

#include <iostream>
#include <string>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "./clasifier.hpp"

#define BOARD_W  9
#define BOARD_H  6
#define WINDOW_SIZE 64
#define WINDOW_OVERLAP 0.8
#define SQUARE_SIZE 0.1
#define ROI_TOP 0.55
#define ROI_BOT 0.30

using namespace cv;
using namespace std;
using namespace cv::ml;

class CarsDetector
{
    public:
    vector<vector<Mat>> labeled_imgs;
    vector<Rect> detected_rects;
    Mat current_frame;
    Mat mask;
    Mat camera_matrix;
    Mat dist_coeff;
    Ptr<SVM> svm;

    // Initialize first_time as false
    CarsDetector(){};

    bool LoadCameraMatrix(String file_);

    bool LoadSVMModel(String file_);

    void FeedImage(Mat image_);

    void UndistortImage();
    
    void LoadLabeledImages();

    void SlideWindowsSearch();
};