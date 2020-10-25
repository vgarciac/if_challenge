// OpenCV 4.0.1.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define DEBUG
#define BOARD_W 9
#define BOARD_H 6
#define SQUARE_SIZE 0.1


using namespace std;
using namespace cv;

Size pattern_size(BOARD_W, BOARD_H);
Size win_size(11, 11);
Size zero_zone(-1, -1);
bool corners_found;

TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, DBL_EPSILON);

int main()
{
    FileStorage fs;
    Mat input_img;

    fs.open("../calibration_file.xml", FileStorage::READ);
    input_img =  imread("../camera_cal/calibration1.jpg");
    if( input_img.empty())
    {
        cerr << "Failed to open test image" << endl;
        return -1;
    }

    if (!fs.isOpened() || input_img.empty())
    {
        cerr << "Failed to open calibration_file" << endl;
        return -1;
    }

    Mat camera_matrix, new_camera_mtx, dist_coeff;
    Mat undistorted;

    fs["camera_matrix"] >> camera_matrix;
    fs["dist_coeff"] >> dist_coeff;

    #ifdef DEBUG
    cout << "camera_matrix: " << camera_matrix << endl;
    cout << "dist_coeff: " << dist_coeff << endl;
    #endif

    int h = input_img.rows;
    int w = input_img.cols;

    new_camera_mtx = getOptimalNewCameraMatrix(camera_matrix, dist_coeff, input_img.size(), 1);
    undistort(input_img, undistorted, camera_matrix, dist_coeff);

    #ifdef DEBUG
    imshow("original image: ", input_img);
    imshow("undistorted image: ", undistorted);
    waitKey(0);
    #endif//DEBUG


    return 0;
}