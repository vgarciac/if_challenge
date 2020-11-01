#include "../include/car_detector.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;

Mat to_plot;

void show(Mat _img, int _ms, String _name, bool _write){
    imshow(_name, _img);
    if(_write)
    {
        imwrite("../results/"+_name+".jpg", _img);
    }
    waitKey(_ms);
}

void CarsDetector::FeedImage(Mat image_)
{
    // Feed algoriyhm with new image
    this->current_frame = image_.clone();
    // And create a new mask (previous masks are used to consider passed frames)
    this->masks.push_back(Mat(image_.size(), CV_8UC1, Scalar(0)));
    return;
}

void CarsDetector::Compute()
{
    
    // Apply distortion correction to image
    this->UndistortImage();

    this->SlideWindowsSearch();    

    this->UpdateMAsks();

    return;
}

bool CarsDetector::LoadSVMModel(String file_)
{
    // Create SVM object and load trained model
    this->svm = Algorithm::load<SVM>(file_);
    if(this->svm == NULL)
    {
        // cerr << "(!) Error loading SVM Model. Exiting." << endl;
        return false;
    }

    return true;
}

bool CarsDetector::LoadCameraMatrix(String file_)
{
    FileStorage fs;
    // Open calibration file
    fs.open(file_, FileStorage::READ);

    if (!fs.isOpened())
    {
        //cerr << "Failed to open calibration_file" << endl;
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


void CarsDetector::SlideWindowsSearch()
{
    // Update search region (only consider a portion of the image)
    int w = this->current_frame.cols;
    int h = this->current_frame.rows;
    int h_start = this->current_frame.rows * ROI_TOP;
    int h_end = h_start + this->current_frame.rows * ROI_BOT;

    // Create ROI object (Region of Interest)
    Rect temp_roi(0, 0, WINDOW_SIZE, WINDOW_SIZE);
    Mat candidate;
    int prediction;

    // Start Sliding windows search
    for(int i = h_start; i+WINDOW_SIZE < h_end; i+= WINDOW_SIZE*WINDOW_OVERLAP)
    {
        temp_roi.y = min(i, h-WINDOW_SIZE);
        for(int j = 0; j+WINDOW_SIZE < w; j+=WINDOW_SIZE*WINDOW_OVERLAP)
        {
            temp_roi.x =  min(j, w-WINDOW_SIZE);
            candidate.release();
            Mat resized;
            // Resize to fir 64x64 (trained images size)
            resize(this->current_frame(temp_roi),resized,Size(64, 64));
            // Compute image features (same as training: HOG, Hist, Binning)
            GetFeatureVector(resized, candidate);
            // Ask the SVM if the current ROI is a vehicle or not
            prediction = svm->predict(candidate);
            if(prediction > 0)
            {
                // Perform a weighted sum in the detected square (overlapping frames become whiter)
                this->masks[this->masks.size()-1](temp_roi) = this->masks[this->masks.size()-1](temp_roi)+Scalar(255)*MAS_SUM;
            }
        }
    }
    return;
}

void CarsDetector::UpdateMAsks()
{
    // rectangle(to_plot, temp_roi, Scalar(i,255,0), 1, 8, 0);

    // Delete oldest mask
    if(this->masks.size() >= N_MASKS)
    {
        this->masks.erase(this->masks.begin());
    }

    // Perform a weighted sum of previous masks (to consider previous frames)
    // Consecutive detected frames becomes whiter
    this->cumul_masks = Mat(this->current_frame.size(), CV_8UC1, Scalar(0));
    for(size_t i = 0; i < this->masks.size()-1; i++)
    {
        cumul_masks = cumul_masks + this->masks[i].clone();
    }
    // to_plot = cumul_masks;
    // show(to_plot, 0, String("heat_map"), true);
    cumul_masks = cumul_masks > MASK_THRESH;

    // Apply morpholical operations to delete small rectangles (lines superposed) 
    Mat m_kernel = getStructuringElement(MORPH_RECT, Size(MORPH_K_CAR, MORPH_K_CAR));
    morphologyEx(this->cumul_masks.clone(), this->cumul_masks, MORPH_CLOSE, m_kernel);
    morphologyEx(this->cumul_masks.clone(), this->cumul_masks, MORPH_OPEN, m_kernel);

    // Gaussian Blur to soften the detected shape
    GaussianBlur(cumul_masks, cumul_masks, Size(15, 15), 0);
}

void CarsDetector::DrawCars(Mat &_img)
{
    // Detect contours in the final mask, contours are drawn in the final image
    findContours( this->cumul_masks, this->cars_contours, this->hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    for( size_t i = 0; i< this->cars_contours.size(); i++ )
    {
        drawContours( _img, this->cars_contours, (int)i,  Scalar( 255, 255, 0 ), 3, LINE_8, hierarchy, 0 );
    }
    // to_plot = _img;
    // show(to_plot, 0, String("heat_map_contours"), true);
    cumul_masks = cumul_masks > MASK_THRESH;
}