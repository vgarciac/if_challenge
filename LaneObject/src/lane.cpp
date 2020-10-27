#include "../include/lane_lines.hpp"

#define DEBUG

using namespace std;
using namespace cv;

int main()
{
    Mat input_img;

    LaneDetector lane_detector;
    VideoCapture capture("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/videos/project_video.mp4");

    char keyboard = 0;
    if (!capture.isOpened()) {
        //error while opening the video input
        cerr << "Unable to open video file" << endl;
        return -1;
    }

    // Load Camera matrix and distortion coefficients
    if (!lane_detector.LoadCameraMatrix("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/calibration_file.xml"))
    {
        cerr << "Failed to open calibration_file" << endl;
        return -1;
    }

    // Load new image
    input_img =  imread("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/test_images/straight_lines1.jpg");
    if( input_img.empty())
    {
        cerr << "Failed to open test image" << endl;
        return -1;
    }

    while(keyboard != 'q' && keyboard != 27)
    {
        if (!capture.read(input_img)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            return 0;
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

        // Detect both lines in the road
        lane_detector.DetectLine();

        // Compute courvature and distance of the vehicle to the center of the road
        lane_detector.ComputeCurvatureDistance();

        // Draw the detected lane with the original image
        lane_detector.DrawLanes();
        // keyboard = (char) waitKey(0);
    }


    
    #ifdef DEBUG
    //imshow("original image: ", lane_detector.current_frame);
    //imshow("undistorted image: ", input_img);
    //waitKey(0);
    #endif//DEBUG


    return 0;
}