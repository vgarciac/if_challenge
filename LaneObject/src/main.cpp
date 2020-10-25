#include "../include/lane_lines.hpp"

#define DEBUG

using namespace std;
using namespace cv;

Size pattern_size(BOARD_W, BOARD_H);
Size win_size(11, 11);
Size zero_zone(-1, -1);
bool corners_found;
TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, DBL_EPSILON);

int main()
{
    bool f_ret;
    Mat input_img;
    Mat camera_matrix, dist_coeff;

    LaneDetector lane_detector;

    // Load Camera matrix and distortion coefficients
    f_ret = lane_detector.LoadCameraMatrix("../calibration_file.xml");
    if (!f_ret)
    {
        cerr << "Failed to open calibration_file" << endl;
        return -1;
    }

    // Load new image
    input_img =  imread("../test_images/straight_lines2.jpg");
    if( input_img.empty())
    {
        cerr << "Failed to open test image" << endl;
        return -1;
    }

    lane_detector.FeedImage(input_img);

    // Apply distortion correction to image
    lane_detector.UndistortImage();

    // Get Region Of Interest (ROI)
    lane_detector.GetImageROI();

    // Detect edges and apply color thresholds
    lane_detector.GetBinaryEdges();

    // Appply a perspective tranformation to get the birds-eye-view
    lane_detector.PerspectiveTransformation();

    // 
    lane_detector.DetectLine();



    
    #ifdef DEBUG
    imshow("original image: ", lane_detector.mask);
    //imshow("undistorted image: ", input_img);
    waitKey(0);
    #endif//DEBUG


    return 0;
}