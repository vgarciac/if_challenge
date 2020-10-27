#include "../include/car_detector.hpp"

#define DEBUG

using namespace std;
using namespace cv;

int main()
{
    Mat input_img;

    CarsDetector car_detector;
    
    car_detector.LoadLabeledImages();

    // Load new image
    input_img =  imread("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/test_images/straight_lines1.jpg");
    if( input_img.empty())
    {
        cerr << "Failed to open test image" << endl;
        return -1;
    }

    // Feed pipeline with new input image
    car_detector.FeedImage(input_img);

    // Apply distortion correction to image
    car_detector.UndistortImage();


    return 0;
}