#include "../include/lane_lines.hpp"

using namespace cv;
using namespace std;

void show(Mat _img, bool rot_){
    if(rot_)
    {
        rotate(_img.clone(), _img, ROTATE_90_COUNTERCLOCKWISE);
    }

    imshow("Debug image", _img);
    waitKey(1);
}


void LaneDetector::FeedImage(Mat image_)
{
    this->mask.release();

    this->current_frame = image_.clone();
    this->to_plot = image_.clone();
    return;
}

bool LaneDetector::LoadCameraMatrix(String file_)
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
        
    // These points are defined by hand using the "straight_lines" images
    vector<Point2f> obj_pts;
    obj_pts.push_back(Point2f(252, 686));
    obj_pts.push_back(Point2f(596, 450));
    obj_pts.push_back(Point2f(684, 450));
    obj_pts.push_back(Point2f(1047, 686));

    // Create the corresponding points in a square form
    vector<Point2f> scene_pts;
    scene_pts.push_back(Point2f(252, 720));
    scene_pts.push_back(Point2f(235, 30));
    scene_pts.push_back(Point2f(1030, 30));
    scene_pts.push_back(Point2f(1047, 720));

    // Transformation from object points to scene points
    this->T = getPerspectiveTransform( obj_pts, scene_pts);

    this->window_step = N_WINDOWS_INIT;
    this->left_line_pts = vector<Point>();
    this->right_line_pts = vector<Point>();

    return true;
}

void LaneDetector::UndistortImage()
{
    Mat undistorted;
    // Get new camera matrix based on new image size
    Mat new_camera_matrix = getOptimalNewCameraMatrix(this->camera_matrix, this->dist_coeff, this->current_frame.size(), 1);

    // Apply distortion correction
    undistort(this->current_frame.clone(), this->current_frame, this->camera_matrix, this->dist_coeff);

    return;
}

void LaneDetector::GetImageROI()
{
    Mat new_roi = Mat(this->current_frame.size(), CV_8UC1, Scalar::all(0));

    // Create triangle vertices
    vector<Point> ptmask3;
    ptmask3.push_back(Point(0, new_roi.rows));
    ptmask3.push_back(Point(new_roi.cols/2, new_roi.rows/1.8));
    ptmask3.push_back(Point(new_roi.cols, new_roi.rows));

    // Create polygon from points
    vector<Point> pt;
    approxPolyDP(ptmask3, pt, 1.0, true);

    // Fill polygon white
    fillConvexPoly(new_roi, &pt[0], pt.size(), 255, 8, 0);

    // Apply mask to input image
    Mat new_image;
    this->current_frame.copyTo(this->mask, new_roi);
    //this->current_frame = new_image;
    return;
}

void LaneDetector::GetBinaryEdges()
{
    cv::Mat gray;
    cvtColor(this->mask, gray, cv::COLOR_RGB2GRAY);

    // Remove noise by apllying gaussian filter
    GaussianBlur(gray, gray, Size(GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0);

    // Find best parameters for canny
    Scalar mean_value = mean(gray);
    double low_th = max(0., (1.0 - CANNY_SIGMA) * double(mean_value[0]));
    double up_th = min(255., (1.0 + CANNY_SIGMA) * double(mean_value[0]));

    // Apply Canny Algorithm to detect edges
    Mat binary_edges;
    cv::Canny(gray, binary_edges, low_th, up_th);
    //show(binary_edges);  

    // Hsv Colos space threshold
    Mat hsv_image;
    Mat hsv_channels[3];
    cvtColor(this->mask, hsv_image, cv::COLOR_RGB2HSV);
    split(hsv_image,hsv_channels);
    Mat binary_sat = hsv_channels[2] > HSV_TH;

    // Mat binary_yellow;
    // inRange(hsv_image, Scalar(10,50,0), Scalar(90,255,255), binary_yellow);
    // show(binary_yellow);

    // Lab Colos space threshold
    Mat lab_image;
    Mat lab_channels[3];
    cvtColor(this->mask, lab_image, cv::COLOR_RGB2Lab);
    split(lab_image, lab_channels);
    Mat binary_l = lab_channels[0] > LAB_TH;
    // show(binary_l); 

    // Combine all masks
    this->mask = binary_sat | binary_l | binary_edges;

    Mat m_kernel = getStructuringElement(MORPH_RECT, Size(MORPH_K, MORPH_K));
    morphologyEx(this->mask.clone(), this->mask, MORPH_CLOSE, m_kernel);
    //show(this->mask);;
    morphologyEx(this->mask.clone(), this->mask, MORPH_OPEN, m_kernel);
    //show(this->mask);   
    return;
}


void LaneDetector::PerspectiveTransformation()
{   
    warpPerspective(this->mask.clone(), this->mask, this->T, this->mask.size(), INTER_LINEAR, BORDER_CONSTANT);
    this->mask = this->mask > 0;

    //show(this->mask);

    return;
}

void LaneDetector::DetectLine()
{
    if(this->first_time)
    {
        // Crop image, we focus in the lower section
        cv::Rect section(0, this->mask.rows/2, this->mask.cols, this->mask.rows/2);
        vector<int> hist = this->ComputePositionHistogram(this->mask(section));

        // Easier to work with the rotated image (for polinomial fitting)
        rotate(this->mask.clone(), this->mask, ROTATE_90_CLOCKWISE);
    
        // Search for two peak values (before and after the middle point)
        this->left_lane_center = std::max_element(hist.begin(), hist.begin() + hist.size()/2) - hist.begin();
        this->right_lane_center = std::max_element(hist.begin() + hist.size()/2, hist.end()) - hist.begin();
        this->first_time = false;
    }
    else
    {
        // Easier to work with the rotated image (for polinomial fitting)
        rotate(this->mask.clone(), this->mask, ROTATE_90_CLOCKWISE);
    
        this->ApplyZoneMask();
        this->left_lane_center = this->left_line_pts[0].y;
        this->right_lane_center = this->right_line_pts[this->right_line_pts.size()-1].y;
        this->window_step = N_WINDOWS_NEXT;
    }

    for(size_t i = 0; i < this->window_step; i++)
    {
        // Update search window for line search
        this->UpdateWindows(i);
    }

    // Once we have some points of the line, we fit it to a 2nd degree polinomial curve
    this->FitLines(this->left_line_pts, this->right_line_pts);


    return;
}

void LaneDetector::FitLines(vector<Point>& left_pts_, vector<Point>& right_pts_)
{
    cv::Mat debug_img(this->mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat imgHist;

    //Draw the fitting points onto the blank graph  
    for (int i = 0; i < left_pts_.size(); i++)
    {
        cv::circle(imgHist, left_pts_[i], 5, cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    // Compute 2 order polynomial which better fits the points
    cv::Mat poly_coeff_l, poly_coeff_r;
    polynomial_curve_fit(left_pts_, 2, poly_coeff_l);
    polynomial_curve_fit(right_pts_, 2, poly_coeff_r);
    
    std::vector<cv::Point> left_points_fit, right_points_fit;
    // Evaluate X points to get the line
    for (int x = 0; x <= 600; x+=20)
    {
        double y_l = poly_coeff_l.at<double>(0, 0) + poly_coeff_l.at<double>(1, 0) * x +
            poly_coeff_l.at<double>(2, 0)*std::pow(x, 2) + poly_coeff_l.at<double>(3, 0)*std::pow(x, 3);

        double y_r = poly_coeff_r.at<double>(0, 0) + poly_coeff_r.at<double>(1, 0) * x +
            poly_coeff_r.at<double>(2, 0)*std::pow(x, 2) + poly_coeff_r.at<double>(3, 0)*std::pow(x, 3);
    
        left_points_fit.push_back(cv::Point(x, y_l));
        // We fill this one inversely, so the draw functions can detect the two lines as a closed contour
        right_points_fit.insert(right_points_fit.begin(), cv::Point(x, y_r));
    }
    
    left_pts_ = left_points_fit;
    right_pts_ = right_points_fit;


    vector<Mat> channels;
    channels.push_back(this->mask);
    channels.push_back(this->mask);
    channels.push_back(this->mask);
    //this->mask.convertTo(imgHist, CV_8UC3);
    merge(channels, imgHist);
    cv::polylines(imgHist, left_pts_, false, cv::Scalar(0, 255, 255), 15, 8, 0);
    cv::polylines(imgHist, right_pts_, false, cv::Scalar(0, 255, 255), 15, 8, 0);
 
    //show(imgHist, true);

    return;
}

vector<int> LaneDetector::ComputePositionHistogram(Mat img_)
{
    vector<int> hist(this->current_frame.cols);
    Mat locations;
    // Get location of Non Zero pixels
    findNonZero(img_, locations);
    // Fill histogram vector
    for(size_t i = 0 ; i < locations.rows ; i++)
    {
        hist[locations.at<Point>(i).x]++;
    }

    // Save histogram image
    // this->printHistogram(hist, "../output/hist_test.jpg", Scalar(255,0,0));

    return hist;
}

void LaneDetector::UpdateWindows(int it_)
{
    // Define slide windows size and position
    int left_center_y = max(this->left_lane_center - WINDOW_W/2, 0);
    int left_center_x = it_*this->mask.cols/this->window_step ;
    int left_width = this->mask.cols/this->window_step ;
    int left_height = WINDOW_W;
    Rect left_roi(left_center_x, left_center_y, left_width, left_height);
    // same for right line
    int right_center_y = min(this->right_lane_center - WINDOW_W/2, this->mask.rows - WINDOW_W);
    int right_center_x = it_*this->mask.cols/this->window_step ;
    int right_width = this->mask.cols/this->window_step;
    int right_height = WINDOW_W;
    Rect right_roi(right_center_x, right_center_y, right_width, right_height);
    
    // Find pixel position for line points
    vector<Point> left_aux_pts, right_aux_pts;
    findNonZero(this->mask(left_roi), left_aux_pts);
    findNonZero(this->mask(right_roi), right_aux_pts);
    
    // If not enough poits detected in window, do not update windows position
    cv::Point2f sum;
    if(left_aux_pts.size() >= MIN_2UPDATE)
    {
        // Get the average from all the points in the window
        sum  = std::accumulate(left_aux_pts.begin(), left_aux_pts.end(), Point(0.0,0.0));
        left_center_y = left_center_y + sum.y/left_aux_pts.size();
        this->left_line_pts.push_back(Point(left_center_x+this->mask.cols/(this->window_step*2), left_center_y));
        // Update centers for next window
        this->left_lane_center = left_center_y;
    }

    if(right_aux_pts.size() >= MIN_2UPDATE)
    {
        // same for right line
        sum  = std::accumulate(right_aux_pts.begin(), right_aux_pts.end(), Point(0.0,0.0));
        right_center_y = right_center_y + sum.y/right_aux_pts.size();
        this->right_line_pts.push_back(Point(right_center_x+this->mask.cols/(this->window_step*2), right_center_y));
        // Update centers for next window
        this->right_lane_center = right_center_y;
    }

    //cv::Mat imgHist(this->mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat imgHist;
    vector<Mat> channels;
    channels.push_back(this->mask);
    channels.push_back(this->mask);
    channels.push_back(this->mask);
    //this->mask.convertTo(imgHist, CV_8UC3);
    merge(channels, imgHist);

    rectangle(imgHist, left_roi, Scalar(0,255,0), 3, 8, 0);
    rectangle(imgHist, right_roi, Scalar(0,255,0), 3, 8, 0);
    circle(imgHist, this->left_line_pts[this->left_line_pts.size()-1], 10, (0,0,255), -1);
    circle(imgHist, this->right_line_pts[this->right_line_pts.size()-1], 10, (0,0,255), -1);
    //show(imgHist, true);

    return;
}

void LaneDetector::ApplyZoneMask()
{
    vector<vector<Point>> left_zone(2), right_zone(2);

    for(size_t i = 0; i < this->left_line_pts.size(); i++)
    {
        left_zone[UP].push_back(Point(this->left_line_pts[i].x, this->left_line_pts[i].y - ZONE_SIZE));
        left_zone[DO].insert(left_zone[DO].begin(), Point(this->left_line_pts[i].x, this->left_line_pts[i].y + ZONE_SIZE));

        right_zone[UP].push_back(Point(this->right_line_pts[i].x, this->right_line_pts[i].y - ZONE_SIZE));
        right_zone[DO].insert(right_zone[DO].begin(), Point(this->right_line_pts[i].x, this->right_line_pts[i].y + ZONE_SIZE));
    }

    cv::Mat mask_zone(this->mask.size(), CV_8UC1, cv::Scalar(0));
    // vector<Mat> channels;
    // channels.push_back(this->mask);
    // channels.push_back(this->mask);
    // channels.push_back(this->mask);
    // merge(channels, imgHist);
    // cv::polylines(imgHist, left_zone[UP], false, cv::Scalar(0, 255, 255), 5, 8, 0);
    // cv::polylines(imgHist, left_zone[DO], false, cv::Scalar(0, 255, 255), 5, 8, 0);
    // cv::polylines(imgHist, right_zone[UP], false, cv::Scalar(0, 255, 255), 5, 8, 0);
    // cv::polylines(imgHist, right_zone[DO], false, cv::Scalar(0, 255, 255), 5, 8, 0);

    vector<Point> print_points;
    print_points.insert(print_points.begin(), left_zone[UP].begin(), left_zone[UP].end());
    print_points.insert(print_points.begin(), left_zone[DO].begin(), left_zone[DO].end());
    fillConvexPoly(mask_zone, &print_points[0], print_points.size(), Scalar(255), 8, 0);

    print_points = vector<Point>();
    print_points.insert(print_points.begin(), right_zone[UP].begin(), right_zone[UP].end());
    print_points.insert(print_points.begin(), right_zone[DO].begin(), right_zone[DO].end());
    fillConvexPoly(mask_zone, &print_points[0], print_points.size(), Scalar(255), 8, 0);

    this->mask = this->mask & mask_zone;

    //show(this->mask, true);
}

double LaneDetector::GetDistance(Point p1_, Point p2_) {
    double dx = p1_.x - p2_.x;
    double dy = p1_.y - p2_.y;
    return abs(sqrt(dx*dx + dy*dy));
}

double LaneDetector::GetArea(Point p1_, Point p2_, Point p3_) {
    return (p1_.x * (p2_.y - p3_.y) + p2_.x * (p3_.y - p1_.y) + p3_.x * (p1_.y - p2_.y)) / 2;
}

void LaneDetector::ComputeCurvatureDistance()
{
    vector<Point> mid_points;
    vector<double> curvature;

    for(size_t i = 0; i < this->left_line_pts.size(); i++)
    {
        mid_points.push_back( Point(this->left_line_pts[i].x, (this->left_line_pts[i].y + this->right_line_pts[i].y)/2));
        // Compute Curvure using 3 points (4*triangleArea/(sideLength1*sideLength2*sideLength3))
        if(i > 2 && i < this->left_line_pts.size()-3)
        {
            Point p1, p2, p3;
            p1 = mid_points[i-2];
            p2 = mid_points[i-1];
            p3 = mid_points[i];

            double d12 = this->GetDistance(p1, p2);
            double d23 = this->GetDistance(p2, p3);
            double d31 = this->GetDistance(p3, p1);

            double area = this->GetArea(p1, p2, p3);

            curvature.push_back(4*area/(d12*d23*d31));
        }
    }

    this->dist_2center = mid_points[0].y - this->current_frame.cols/2;
    this->dist_2center *= M_2PIX;

    double sum  = accumulate(curvature.begin(), curvature.end(), 0.0);
    this->curvature_ref = sum/curvature.size();
    //cout << this->dist_2center << endl;

    cv::Mat debug_img(this->mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::polylines(debug_img, mid_points, false, cv::Scalar(0, 255, 255), 8, 8, 0);


    // rotate(debug_img.clone(), debug_img, ROTATE_90_COUNTERCLOCKWISE );
    //debug_img = this->current_frame + debug_img*0.5;
    //imshow("debug_img", debug_img);
    //waitKey(0);
    return;
}

void LaneDetector::DrawLanes()
{
    cv::Mat debug_img(this->mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    vector<Point> print_points;
    print_points.insert(print_points.begin(), this->left_line_pts.begin(), this->left_line_pts.end());
    print_points.insert(print_points.begin(), this->right_line_pts.begin(), this->right_line_pts.end());
    fillConvexPoly(debug_img, &print_points[0], print_points.size(), Scalar(200,0,0), 8, 0);

    cv::polylines(debug_img, this->left_line_pts, false, cv::Scalar(0, 255, 255), 15, 8, 0);
    cv::polylines(debug_img, this->right_line_pts, false, cv::Scalar(0, 255, 255), 15, 8, 0);

    rotate(debug_img.clone(), debug_img, ROTATE_90_COUNTERCLOCKWISE );

    warpPerspective(debug_img.clone(), debug_img, this->T.inv(), debug_img.size(), INTER_LINEAR, BORDER_CONSTANT);

    debug_img = this->current_frame + debug_img*0.5;

    this->left_lane_center = this->left_line_pts[0].y;
    this->right_lane_center = this->right_line_pts[this->right_line_pts.size()-1].y;


    circle(debug_img, Point(this->left_lane_center,700), 10, (0,0,255), -1);
    circle(debug_img, Point(this->right_lane_center,700), 10, (0,0,255), -1);

    show(debug_img);

    return;
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