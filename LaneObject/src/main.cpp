#include "../include/car_detector.hpp"
#include "../include/lane_lines.hpp"

using namespace std;
using namespace cv;

int main()
{
    Mat input_img;

    CarsDetector car_detector;
    LaneDetector lane_detector;
    String calibration_file = "../calibration_file.xml";
    VideoCapture capture("../videos/project_video.mp4");
    
    if(!car_detector.LoadCameraMatrix(calibration_file))
    {
        cerr << "(!) Failed to open calibration_file. Exiting" << endl;
        return -1;
    }

    if(!lane_detector.LoadCameraMatrix(calibration_file))
    {
        cerr << "(!) Failed to open calibration_file. Exiting" << endl;
        return -1;
    }

    if(!car_detector.LoadSVMModel("../trained_svm.xml"))
    {
        cerr << "(!) Failed to open SVM_file. Exiting." << endl;
        return -1;
    }

    if (!capture.read(input_img)) {
        cerr << "Unable to read next frame." << endl;
        cerr << "Exiting..." << endl;
        return 0;
    }
    VideoWriter video("../pipeline.avi",CV_FOURCC('M','J','P','G'), 30, input_img.size());
    while(true)
    {
        if (!capture.read(input_img)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            return 0;
        }

        // Feed pipeline with new input image
        car_detector.FeedImage(input_img);
        lane_detector.FeedImage(input_img);

        car_detector.Compute();
        lane_detector.Compute();

        car_detector.DrawCars(input_img);
        lane_detector.DrawLanes(input_img);

        // video.write(input_img);
        show(input_img, 1);
    }

    // When everything done, release the video capture object
	capture.release();
    destroyAllWindows();

    return 0;
}