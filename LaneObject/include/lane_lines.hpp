#include <iostream>
#include <string>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define GAUSSIAN_KERNEL 15
#define BOARD_W 9
#define BOARD_H 6
#define SQUARE_SIZE 0.1
#define CANNY_SIGMA 0.5
#define HSV_TH 150
#define LAB_TH 150
#define M_2PIX 3.7/700

#define N_WINDOWS 8
#define WINDOW_W 150

using namespace cv;
using namespace std;

void PolyFit(vector<Point> data_pts_, int order_, vector<float>& coef_);
Mat CustomPolyfit(vector<Point>& in_point, int n);
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);

class LaneDetector
{
    public:
    Mat current_frame;
    Mat mask;
    Mat lane_mask;
    Mat T;
    Mat camera_matrix;
    Mat dist_coeff;
    bool first_time;
    double curvature_ref;
    double dist_2center;

    int left_lane_center;
    int right_lane_center;

    vector<Point> left_line_pts;
    vector<Point> right_line_pts;

    // Initialize first_time as false
    LaneDetector(bool _first = false) : first_time(_first) {};

    bool LoadCameraMatrix(String file_);

    void FeedImage(Mat image_);

    void UndistortImage();

    void GetImageROI();

    void GetBinaryEdges();

    void PerspectiveTransformation();

    void DetectLine();

    void DrawLanes();

    void ComputeCurvatureDistance();

    private:

    void FitLines(vector<Point>& left_pts_, vector<Point>& right_pts_);

    void UpdateWindows(int it_);

    vector<int> ComputePositionHistogram(Mat _img);

    void printHistogram(vector<int> histogram, std::string filename, cv::Scalar color);
};