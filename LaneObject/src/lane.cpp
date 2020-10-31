#include "../include/lane_lines.hpp"

#define DEBUG

using namespace std;
using namespace cv;

int main()
{
    Mat input_img;

    LaneDetector lane_detector;
    VideoCapture capture("../videos/project_video.mp4");

    char keyboard = 0;
    if (!capture.isOpened()) {
        //error while opening the video input
        cerr << "Unable to open video file" << endl;
        return -1;
    }

    // Load Camera matrix and distortion coefficients
    if (!lane_detector.LoadCameraMatrix("../calibration_file.xml"))
    {
        cerr << "Failed to open calibration_file" << endl;
        return -1;
    }

    // Load new image
    input_img =  imread("../test_images/straight_lines1.jpg");
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
        
        // Feed the algorithm with new video frame
        lane_detector.FeedImage(input_img);

        // Perform advance lane detection algorithm
        lane_detector.Compute();

        // Draw the detected lane with the original image
        lane_detector.DrawLanes(input_img);

        show(input_img);
    }
    return 0;
}