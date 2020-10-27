#include <iostream>
#include <string>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define BOARD_W 9
#define BOARD_H 6
#define SQUARE_SIZE 0.1

using namespace cv;
using namespace std;

void show(Mat _img);

class CarsDetector
{
    public:
    vector<vector<Mat>> labeled_imgs;
    Mat current_frame;
    Mat mask;
    Mat camera_matrix;
    Mat dist_coeff;

    // Initialize first_time as false
    CarsDetector(){};

    bool LoadCameraMatrix(String file_);

    void FeedImage(Mat image_);

    void UndistortImage();
    
    void LoadLabeledImages();
};