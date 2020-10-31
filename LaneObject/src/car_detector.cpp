#include "../include/car_detector.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;

void CarsDetector::FeedImage(Mat image_)
{
    this->mask.release();

    this->current_frame = image_.clone();
    return;
}

bool CarsDetector::LoadSVMModel(String file_)
{
    this->svm = Algorithm::load<SVM>(file_);
    if(this->svm == NULL)
    {
        // cerr << "(!) Error loading SVM Model. Exiting." << endl;
        return false;
    }
    svm->setType(SVM::C_SVC);
    svm->setC(0.1);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));

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

void CarsDetector::LoadLabeledImages()
{
    /*
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
*/
    return;
}

void CarsDetector::SlideWindowsSearch()
{
    this->detected_rects = vector<Rect>();
    int w = this->current_frame.cols;
    int h = this->current_frame.rows;
    int h_start = this->current_frame.rows * ROI_TOP;
    int h_end = h_start + this->current_frame.rows * ROI_BOT;

    Rect temp_roi(0, 0, WINDOW_SIZE, WINDOW_SIZE);
    Mat candidate;
    int prediction;

    FileStorage fs;
    fs.open("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/normalisation.xml", FileStorage::READ);
    Mat mean, stdev;
    Mat meansigma;
    fs["meansigma"] >> meansigma;

    Mat means = meansigma.col(0).clone();
    Mat sigmas = meansigma.col(1).clone();

    for(int i = h_start; i < h_end; i+= WINDOW_SIZE*WINDOW_OVERLAP)
    {
        temp_roi.y = min(i, h-WINDOW_SIZE);
        for(int j = 0; j <  w; j+=WINDOW_SIZE*WINDOW_OVERLAP)
        {
            temp_roi.x =  min(j, w-WINDOW_SIZE);
            candidate.release();
            Mat resized;
            resize(this->current_frame(temp_roi),resized,Size(64, 64));
            GetFeatureVector(resized, candidate);

            // for(size_t k = 0; k < mean.cols; k++){
            //     float mean = means.at<float>(k);
            //     float sigma = sigmas.at<float>(k);
            //     candidate.at<float>(0,k) = (candidate.at<float>(0,k) - mean) / sigma;
            //     // candidate.at<float>(0,k) = candidate.at<float>(0,k) - mean.at<float>(0,k);
            //     // candidate.at<float>(0,k) = candidate.at<float>(0,k)/stdev.at<float>(0,k);
            // }

            prediction = svm->predict(candidate);
            if(prediction > 0)
            {
                //show(this->current_frame(temp_roi));
                rectangle(this->current_frame, temp_roi, Scalar(0,255,0), 3, 8, 0);
            }
            //rectangle(this->current_frame, temp_roi, Scalar(0,255,0), 3, 8, 0);
            // imwrite("../data/test_data/"+to_string(i*j)+".png", this->current_frame(temp_roi));
        }
    }

    return;
}