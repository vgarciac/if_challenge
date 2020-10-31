#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

//#define DEBUG
const int BOARD_W = 9;
const int BOARD_H = 6;
const float SQUARE_SIZE = 0.24;

using namespace std;
using namespace cv;

Size pattern_size(BOARD_W, BOARD_H);
Size win_size(11, 11);
Size zero_zone(-1, -1);
bool corners_found;

TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, DBL_EPSILON);

int main()
{
    vector<vector<Point3f>> object_points;
    vector<vector<Point2f>> image_points;
    vector<Point2f> img_corners;
    vector<Point3f> obj_corners;
    vector<cv::String> fn;

    for (int i = 0; i < BOARD_H; i++)
      for (int j = 0; j < BOARD_W; j++)
        obj_corners.push_back(Point3f( (float)j * SQUARE_SIZE, (float)i * SQUARE_SIZE, 0) );

    glob("../camera_cal/*.jpg", fn, false);

    size_t count = fn.size();
    if(count == 0)
    {
        cout << "Could not open calibration images" << endl;
        return -1;
    }
    else
    {
        cout << "Found: " << count << " images" << endl;
    }

    Mat input_img;
    for (size_t i = 0; i < count; i++)
    {
        corners_found = false;

        input_img = imread(fn[i]);
        cvtColor(input_img, input_img, COLOR_BGR2GRAY);

        corners_found = findChessboardCorners(input_img, pattern_size, img_corners);

        if (corners_found)
        {
            cornerSubPix(input_img, img_corners, win_size, zero_zone, criteria);
            image_points.push_back(img_corners);
            object_points.push_back(obj_corners);
        }


        #ifdef DEBUG
        cout << " image N:  " << i << endl;
        cout << " image_points.size() " << image_points.size() << endl;
        drawChessboardCorners(input_img, pattern_size, img_corners, corners_found);
        if(corners_found)
        imshow("img with corners", input_img);
        waitKey(0);
        #endif//DEBUG

    }

    Mat camera_matrix, dist_coeff;
    vector<Mat> rvecs, tvecs;
    calibrateCamera(object_points, image_points, input_img.size(), camera_matrix , dist_coeff, rvecs , tvecs, 0);

    FileStorage fs("../calibration_file.xml", FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "dist_coeff" << dist_coeff;
    fs << "board_width" << BOARD_W;
    fs << "board_height" << BOARD_H;
    fs << "square_size" << SQUARE_SIZE;
    printf("Done Calibration\n");

    destroyAllWindows();
    return 0;
}