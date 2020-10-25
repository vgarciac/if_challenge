#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define GAUSSIAN_KERNEL 15
#define BOARD_W 9
#define BOARD_H 6
#define SQUARE_SIZE 0.1
#define CANNY_SIGMA 0.5
#define HSV_TH 150
#define LAB_TH 150

#define N_WINDOWS 10
#define WINDOW_H 100

using namespace cv;
using namespace std;

class LaneDetector
{
    public:
    Mat current_frame;
    Mat mask;
    Mat lower_mask;
    Mat T;
    Mat camera_matrix;
    Mat dist_coeff;

    bool LoadCameraMatrix(String file_);

    void FeedImage(Mat image_);

    void UndistortImage();

    void GetImageROI();

    void GetBinaryEdges();

    void PerspectiveTransformation();

    void DetectLine();

    private:

    vector<int> ComputePositionHistogram();

    void printHistogram(vector<int> histogram, std::string filename, cv::Scalar color);
};