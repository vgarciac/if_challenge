#include "../include/lane_lines.hpp"

using namespace cv;
using namespace std;


void LaneDetector::FeedImage(Mat image_)
{
    this->current_frame = image_.clone();
    return;
}

bool LaneDetector::LoadCameraMatrix(String file_)
{
    FileStorage fs;
    fs.open(file_, FileStorage::READ);

    if (!fs.isOpened())
    {
        cerr << "Failed to open calibration_file" << endl;
        return false;
    }

    fs["camera_matrix"] >> this->camera_matrix;
    fs["dist_coeff"] >> this->dist_coeff;

    return true;
}

void LaneDetector::UndistortImage()
{
    Mat undistorted;
    Mat new_camera_matrix = getOptimalNewCameraMatrix(this->camera_matrix, this->dist_coeff, this->current_frame.size(), 1);

    undistort(this->current_frame, undistorted, this->camera_matrix, this->dist_coeff);

    this->current_frame = undistorted;
    return;
}

void LaneDetector::GetImageROI()
{
    this->mask = Mat(this->current_frame.size(), CV_8UC1, Scalar::all(0));

    // Create triangle vertices
    vector<Point> ptmask3;
    ptmask3.push_back(Point(0, this->mask.rows));
    ptmask3.push_back(Point(this->mask.cols/2, this->mask.rows/1.9));
    ptmask3.push_back(Point(this->mask.cols, this->mask.rows));

    // Create polygon from points
    vector<Point> pt;
    approxPolyDP(ptmask3, pt, 1.0, true);

    // Fill polygon white
    fillConvexPoly(this->mask, &pt[0], pt.size(), 255, 8, 0);

    // Apply mask to input image
    Mat new_image;
    this->current_frame.copyTo(new_image, this->mask);
    this->current_frame = new_image;

    return;
}

void LaneDetector::GetBinaryEdges()
{
    cv::Mat gray;
    cvtColor(this->current_frame, gray, cv::COLOR_RGB2GRAY);

    // Remove noise by apllying gaussian filter
    GaussianBlur(gray, gray, Size(GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0);

    // Find best parameters for canny
    Scalar mean_value = mean(gray);
    double low_th = max(0., (1.0 - CANNY_SIGMA) * double(mean_value[0]));
    double up_th = min(255., (1.0 + CANNY_SIGMA) * double(mean_value[0]));

    // Canny Algorithm
    Mat binary_edges;
    cv::Canny(gray, binary_edges, low_th, up_th);

    // Hsc Colos space threshold
    Mat hsv_image;
    Mat hsv_channels[3];
    cvtColor(this->current_frame, hsv_image, cv::COLOR_RGB2HSV);
    split(hsv_image,hsv_channels);
    Mat binary_s = hsv_channels[2] > HSV_TH;

    // Lab Colos space threshold
    Mat lab_image;
    Mat lab_channels[3];
    cvtColor(this->current_frame, lab_image, cv::COLOR_RGB2Lab);
    split(lab_image, lab_channels);
    Mat binary_l = lab_channels[0] > LAB_TH;

    this->mask = binary_s | binary_l | binary_edges;

    imshow("binary_edges", this->current_frame);
    waitKey(0);
    return;
}


void LaneDetector::PerspectiveTransformation()
{   
    // 
    vector<Point2f> obj_pts;
    obj_pts.push_back(Point2f(252, 686));
    obj_pts.push_back(Point2f(596, 450));
    obj_pts.push_back(Point2f(684, 450));
    obj_pts.push_back(Point2f(1047, 686));

    // 
    vector<Point2f> scene_pts;
    scene_pts.push_back(Point2f(252, 720));
    scene_pts.push_back(Point2f(235, -30));
    scene_pts.push_back(Point2f(1030, -30));
    scene_pts.push_back(Point2f(1047, 720));

    // Transformation from object points to scene points
    Mat T;
    T = getPerspectiveTransform( obj_pts, scene_pts);

    warpPerspective(this->mask.clone(), this->mask, T, this->mask.size(), INTER_LINEAR, BORDER_CONSTANT);
    
    return;
}

void LaneDetector::DetectLine()
{
    // Crop image, we focus in the lower section
    cv::Rect section(0, this->mask.rows/2, this->mask.cols, this->mask.rows/2);
    this->mask = this->mask(section);
    this->mask = this->mask > 0;



    vector<int> hist = this->ComputePositionHistogram();

    return;
}

vector<int> LaneDetector::ComputePositionHistogram()
{
    vector<int> hist(this->current_frame.cols);
    Mat locations;

    findNonZero(this->mask, locations);

    for(size_t i = 0 ; i < locations.rows ; i++)
    {
        hist[locations.at<Point>(i).x]++;
    }

    int first_idx = std::max_element(hist.begin(), hist.begin() + hist.size()/2) - hist.begin();
    int second_idx = std::max_element(hist.begin() + hist.size()/2, hist.end()) - hist.begin();

    this->printHistogram(hist, "../output/hist_test.jpg", Scalar(255,0,0));

    return hist;
}

void LaneDetector::printHistogram(vector<int> histogram, std::string filename, cv::Scalar color)
{
	// Finding the maximum value of the histogram. It will be used to scale the
	// histogram to fit the image.

	int max = *max_element(histogram.begin(), histogram.end());

	// Creating an image from the histogram.
	cv::Mat imgHist(1480, 1580, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Point pt1, pt2;
	pt1.y = 1380;
	for (int i = 0; i < histogram.size(); i++)
	{
		pt1.x = 150 +  i + 1;
		pt2.x = 150 +  i + 3;
		pt2.y = 1380 - 1280 * histogram[i] / max;
		cv::rectangle(imgHist, pt1, pt2, color, cv::FILLED);
	}
	// y-axis labels
	cv::rectangle(imgHist, cv::Point(130, 1400), cv::Point(1450, 80), cv::Scalar(0, 0, 0), 1);
	cv::putText(imgHist, std::to_string(max), cv::Point(10, 100), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(max * 3 / 4), cv::Point(10, 420), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(max / 2), cv::Point(10, 740), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(max / 4), cv::Point(10, 1060), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(0), cv::Point(10, 1380), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
	// x-axis labels
	cv::putText(imgHist, std::to_string(0), cv::Point(152 - 7 * 1, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(int(histogram.size()*0.25)), cv::Point(467 - 7 * 2, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(int(histogram.size()*0.5)), cv::Point(787 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(int(histogram.size()*0.75)), cv::Point(1107 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(int(histogram.size())), cv::Point(1427 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 0), 2.0);

	// Saving the image
	cv::imwrite(filename, imgHist);
}