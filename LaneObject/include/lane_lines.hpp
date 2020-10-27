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
#define HSV_TH 170
#define LAB_TH 200
#define M_2PIX 3.7/700
#define MIN_2UPDATE 50
#define MORPH_K 3

#define N_WINDOWS_INIT 7
#define N_WINDOWS_NEXT 20
#define WINDOW_W 250
#define ZONE_SIZE 100


enum ZONES 
{
    UP,
    DO//DOWN
};

using namespace cv;
using namespace std;

void PolyFit(vector<Point> data_pts_, int order_, vector<float>& coef_);
Mat CustomPolyfit(vector<Point>& in_point, int n);
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);

void show(Mat _img, bool rot_ = false);

class LaneDetector
{
    public:
    Mat to_plot;
    Mat current_frame;
    Mat mask;
    Mat lane_mask;
    Mat T;
    Mat camera_matrix;
    Mat dist_coeff;
    bool first_time;
    double curvature_ref;
    double dist_2center;
    int window_step;

    int left_lane_center;
    int right_lane_center;

    vector<Point> left_line_pts;
    vector<Point> right_line_pts;

    // Initialize first_time as false
    LaneDetector(bool _first = true) : first_time(_first) {};

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

    void ApplyZoneMask();

    double GetArea(Point p1_, Point p2_, Point p3_);

    double GetDistance(Point p1_, Point p2_);

    vector<int> ComputePositionHistogram(Mat _img);

    void printHistogram(vector<int> histogram, std::string filename, cv::Scalar color);
};