#pragma once

#include <iostream>
#include <string>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

const int GAUSSIAN_KERNEL = 5;
const float CANNY_SIGMA = 0.5;
const int HSV_TH = 170;
const int LAB_TH = 200;
const float M_2PIX_X = 3.7/700.0;
const float M_2PIX_Y = 30.0/700.0;
const int MIN_2UPDATE = 50;
const int MORPH_K = 3;

const int N_WINDOWS_INIT = 7;
const int N_WINDOWS_NEXT = 20;
const int WINDOW_W = 250;
const int ZONE_SIZE = 150;
const int AVG_LINES = 5;


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

void show(Mat _img, bool rot_ = false, String _name = "DEBUG");

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

    vector<vector<Point>> avg_left;
    vector<vector<Point>> avg_right;

    // Initialize first_time as false
    LaneDetector(bool _first = true) : first_time(_first) {};

    bool LoadCameraMatrix(String file_);

    void FeedImage(Mat image_);

    void UndistortImage();

    void GetImageROI();

    void GetBinaryEdges();

    void PerspectiveTransformation();

    void DetectLine();

    void DrawLanes(Mat &_img);

    void ComputeCurvatureDistance();

    void Compute();

    private:

    void FitLines(vector<Point>& left_pts_, vector<Point>& right_pts_);

    void UpdateWindows(int it_);

    void ApplyZoneMask();

    void AvgPoints();

    double GetArea(Point2f p1_, Point2f p2_, Point2f p3_);

    double GetDistance(Point2f p1_, Point2f p2_);

    vector<int> ComputePositionHistogram(Mat _img);

    void printHistogram(vector<int> histogram, std::string filename, cv::Scalar color);
};