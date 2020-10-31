#include "../include/car_detector.hpp"

#define DEBUG

using namespace std;
using namespace cv;

int main()
{
    Mat input_img;

    CarsDetector car_detector;
    VideoCapture capture("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/videos/project_video.mp4");
    
    car_detector.LoadLabeledImages();
    if(!car_detector.LoadCameraMatrix("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/calibration_file.xml"))
    {
        cerr << "(!) Failed to open calibration_file. Exiting" << endl;
        return -1;
    }

    if(!car_detector.LoadSVMModel("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/svm_big.xml"))
    {
        cerr << "(!) Failed to open SVM_file. Exiting." << endl;
        return -1;
    }
    // Load new image
    input_img =  imread("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/test_images/straight_lines1.jpg");

    if( input_img.empty())
    {
        cerr << "Failed to open test image" << endl;
        return -1;
    }

    while(true)
    {
        if (!capture.read(input_img)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            return 0;
        }

        // Feed pipeline with new input image
        car_detector.FeedImage(input_img);
        
        // Apply distortion correction to image
        car_detector.UndistortImage();

        Rect roi(0, input_img.rows*ROI_TOP, input_img.cols, input_img.rows*ROI_BOT);
        rectangle(car_detector.current_frame, roi, Scalar(0,255,0), 3, 8, 0);

        car_detector.SlideWindowsSearch();    
        show(car_detector.current_frame, 1);
    }


    return 0;
}