#include "../include/car_detector.hpp"

using namespace std;
using namespace cv;

int main()
{
    Mat input_img;

    CarsDetector car_detector;
    VideoCapture capture("../videos/project_video.mp4");
    
    if(!car_detector.LoadCameraMatrix("../calibration_file.xml"))
    {
        cerr << "(!) Failed to open calibration_file. Exiting" << endl;
        return -1;
    }

    if(!car_detector.LoadSVMModel("../trained_svm.xml"))
    {
        cerr << "(!) Failed to open SVM_file. Exiting." << endl;
        return -1;
    }
    // Load new image
    input_img =  imread("../test_images/straight_lines1.jpg");

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

        car_detector.Compute();

        car_detector.DrawCars(input_img);

        show(input_img, 1);
        
    }


    return 0;
}