#include "../include/car_detector.hpp"

using namespace cv;
using namespace std;

enum LABELS
{
    CARS,
    NON_CARS
};


void show(Mat _img){

    imshow("Debug image", _img);
    waitKey(1);
}

void CarsDetector::FeedImage(Mat image_)
{
    this->mask.release();

    this->current_frame = image_.clone();
    return;
}

bool CarsDetector::LoadCameraMatrix(String file_)
{
    FileStorage fs;
    // Open calibration file
    fs.open(file_, FileStorage::READ);

    if (!fs.isOpened())
    {
        cerr << "Failed to open calibration_file" << endl;
        return false;
    }

    // save opencv matrices
    fs["camera_matrix"] >> this->camera_matrix;
    fs["dist_coeff"] >> this->dist_coeff;

    return true;
}

void CarsDetector::UndistortImage()
{
    Mat undistorted;
    // Get new camera matrix based on new image size
    Mat new_camera_matrix = getOptimalNewCameraMatrix(this->camera_matrix, this->dist_coeff, this->current_frame.size(), 1);

    // Apply distortion correction
    undistort(this->current_frame.clone(), this->current_frame, this->camera_matrix, this->dist_coeff);

    return;
}

void CarsDetector::LoadLabeledImages()
{
    this->labeled_imgs = vector<vector<Mat>>(2);
    this->labeled_imgs[CARS] = vector<Mat>();
    this->labeled_imgs[NON_CARS] = vector<Mat>();

    vector<cv::String> file_cars, file_non_cars;
    glob("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/data/vehicles/vehicles/*.png", file_cars, true);
    glob("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/data/non-vehicles/non-vehicles/*.png", file_non_cars, true);

    for(String path: file_cars)
    {
        this->labeled_imgs[CARS].push_back(imread(path));
    }

    for(String path: file_non_cars)
    {
        this->labeled_imgs[NON_CARS].push_back(imread(path));
    }

    cout <<"this->labeled_imgs[CARS]: " << this->labeled_imgs[CARS].size() << endl;
    cout <<"this->labeled_imgs[NON_CARS]: " << this->labeled_imgs[NON_CARS].size() << endl;

    return;
}