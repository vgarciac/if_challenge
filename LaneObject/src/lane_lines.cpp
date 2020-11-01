#include "../include/lane_lines.hpp"

using namespace cv;
using namespace std;

// Auxiliar function
void show(Mat _img, bool rot_, String _name){
    if(rot_)
    {
        rotate(_img.clone(), _img, ROTATE_90_COUNTERCLOCKWISE);
    }
    imshow(_name, _img);
    waitKey(1);
}

// Auxiliar function
struct AddPoint
{
    Point pt;
    AddPoint(Point _pt) : pt(_pt){};

    void operator()(Point &in_pt) const
    {
        in_pt.x += pt.x;
        in_pt.y += pt.y;
    }
};

void LaneDetector::FeedImage(Mat image_)
{
    // Save input image. mask image will be used as binary mask
    this->mask =  Mat(image_.size(), CV_8UC1, Scalar::all(0));
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

    // Initialize parameters
    this->window_step = N_WINDOWS_INIT;
    this->left_line_pts = vector<Point>();
    this->right_line_pts = vector<Point>();

    return true;
}

void LaneDetector::Compute()
{
    // Apply distortion correction to image
    this->UndistortImage();

    // Detect edges and apply color thresholds
    this->GetBinaryEdges();

    // Get Region Of Interest (ROI)
    this->GetImageROI();

    // Appply a perspective tranformation to get the birds-eye-view
    this->PerspectiveTransformation();

    // Detect both lines in the road
    this->DetectLine();

    // Compute courvature and distance of the vehicle to the center of the road
    this->ComputeCurvatureDistance();

    return;
}

void LaneDetector::UndistortImage()
{

    // Get new camera matrix based on new image size
    Mat new_camera_matrix = getOptimalNewCameraMatrix(this->camera_matrix, this->dist_coeff, this->current_frame.size(), 1);
    // Apply distortion correction
    undistort(this->current_frame.clone(), this->current_frame, this->camera_matrix, this->dist_coeff);

    return;
}

void LaneDetector::GetImageROI()
{
    Mat new_roi = Mat(this->mask.size(), CV_8UC1, Scalar::all(0));

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
    Mat aux_mat;
    this->mask.copyTo(aux_mat, new_roi);
    this->mask = aux_mat;
    // show(this->mask);
    return;
}

void LaneDetector::GetBinaryEdges()
{
    cv::Mat gray;
    cvtColor(this->current_frame, gray, cv::COLOR_RGB2GRAY);

    // Find best parameters for canny
    Scalar mean_value = mean(gray);
    double low_th = max(0., (1.0 - CANNY_SIGMA) * double(mean_value[0]));
    double up_th = min(255., (1.0 + CANNY_SIGMA) * double(mean_value[0]));

    Mat binary_edges;
    // Replaced by gradient images
    // Apply Canny Algorithm to detect edges
    // cv::Canny(gray, binary_edges, low_th, up_th);

    // Remove noise by apllying gaussian filter
    GaussianBlur(gray, gray, Size(GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0);
    // Compute gradient 
    Mat abs_grad_x, abs_grad_y;
    Mat dx, dy;
    Sobel(gray, dx, CV_32F, 1, 0);
    Sobel(gray, dy, CV_32F, 0, 1);
    convertScaleAbs( dx, abs_grad_x );
    convertScaleAbs( dy, abs_grad_y );
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, binary_edges );
    // show(binary_edges);

    // Hsv Colos space threshold
    Mat hsv_image;
    Mat hsv_channels[3];
    cvtColor(this->current_frame, hsv_image, cv::COLOR_RGB2HSV);
    split(hsv_image,hsv_channels);
    Mat binary_val = hsv_channels[2] > HSV_TH;

    // Lab Colos space threshold
    Mat lab_image;
    Mat lab_channels[3];
    cvtColor(this->current_frame, lab_image, cv::COLOR_RGB2Lab);
    split(lab_image, lab_channels);
    Mat binary_l = lab_channels[0] > LAB_TH;

    // Combine all masks
    this->mask = binary_val | binary_l | binary_edges;
    //show(mask, 0);

    // Apply morphological transformations to delete small particles, and fill small holes
    Mat m_kernel = getStructuringElement(MORPH_RECT, Size(MORPH_K, MORPH_K));
    morphologyEx(this->mask.clone(), this->mask, MORPH_CLOSE, m_kernel);
    morphologyEx(this->mask.clone(), this->mask, MORPH_OPEN, m_kernel); 
    //show(this->mask);
    return;
}


void LaneDetector::PerspectiveTransformation()
{   
    // Apply perspective transformation to mask (bird-eye)
    warpPerspective(this->mask.clone(), this->mask, this->T, this->mask.size(), INTER_LINEAR, BORDER_CONSTANT);
    this->mask = this->mask > 50;

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

        // Apply mask near the position of the previous detected line
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

    // Average last N lines (slow implementation)
    // this->AvgPoints();

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
    for (int x = 0; x <= 720; x+=20)
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

    // Mat debug_img;
    // cv::polylines(debug_img, left_pts_, false, cv::Scalar(0, 0, 255), 15, 8, 0);
    // cv::polylines(debug_img, right_pts_, false, cv::Scalar(0, 0, 255), 15, 8, 0);
 
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
        std::for_each(left_aux_pts.begin(), left_aux_pts.end(), AddPoint(Point(left_center_x, left_center_y)));
        this->left_line_pts.insert(this->left_line_pts.begin(), left_aux_pts.begin(), left_aux_pts.end());
        sum  = std::accumulate(left_aux_pts.begin(), left_aux_pts.end(), Point(0.0,0.0));
        left_center_y = sum.y/left_aux_pts.size();
        // this->left_line_pts.push_back(Point(left_center_x+this->mask.cols/(this->window_step*2), left_center_y));
        // Update centers for next window
        this->left_lane_center = left_center_y;
    }

    if(right_aux_pts.size() >= MIN_2UPDATE)
    {
        // same for right line
        std::for_each(right_aux_pts.begin(), right_aux_pts.end(), AddPoint(Point(right_center_x, right_center_y)));
        this->right_line_pts.insert(this->right_line_pts.begin(), right_aux_pts.begin(), right_aux_pts.end());
        sum  = std::accumulate(right_aux_pts.begin(), right_aux_pts.end(), Point(0.0,0.0));
        right_center_y = sum.y/right_aux_pts.size();
        // this->right_line_pts.push_back(Point(right_center_x+this->mask.cols/(this->window_step*2), right_center_y));
        // Update centers for next window
        this->right_lane_center = right_center_y;
    }

    //cv::Mat imgHist(this->mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat imgHist;
    vector<Mat> channels;
    channels.push_back(this->mask);
    channels.push_back(this->mask);
    channels.push_back(this->mask);
    // this->mask.convertTo(imgHist, CV_8UC3);
    merge(channels, imgHist);

    rectangle(imgHist, left_roi, Scalar(0,255,0), 3, 8, 0);
    rectangle(imgHist, right_roi, Scalar(0,255,0), 3, 8, 0);
    //circle(to_plot, this->left_line_pts[this->left_line_pts.size()-1], 10, (0,0,255), -1);
    //circle(to_plot, this->right_line_pts[this->right_line_pts.size()-1], 10, (0,0,255), -1);

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

    // Draw function
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

    // show(to_plot, 0, "mask_zone");
    // imwrite("../results/mask_zone.jpg", to_plot );
}

double LaneDetector::GetDistance(Point2f p1_, Point2f p2_) {
    // Return the euclidean distance between two points
    double dx = p1_.x - p2_.x;
    double dy = p1_.y - p2_.y;
    return abs(sqrt(dx*dx + dy*dy));
}

double LaneDetector::GetArea(Point2f p1_, Point2f p2_, Point2f p3_) {
    // Return the area formed by three points
    return (p1_.x * (p2_.y - p3_.y) + p2_.x * (p3_.y - p1_.y) + p3_.x * (p1_.y - p2_.y)) / 2;
}

void LaneDetector::ComputeCurvatureDistance()
{
    vector<Point> mid_points;
    vector<double> curvature;
    // Loop over all the points, we have to detect the curvure locally for each three points
    for(size_t i = 0; i < this->left_line_pts.size(); i++)
    {
        mid_points.push_back( Point(this->left_line_pts[i].x, (this->left_line_pts[i].y + this->right_line_pts[i].y)/2));
        // Compute Curvure using 3 points (4*triangleArea/(sideLength1*sideLength2*sideLength3))
        if(i > 2 && i < this->left_line_pts.size()-3)
        {
            Point2f p1, p2, p3;
            p1 = mid_points[i-2];
            p2 = mid_points[i-1];
            p3 = mid_points[i];

            // Convert units to 'm'
            // Note: at this point the image is still rotated, so X -> Y
            p1.x = p1.x * M_2PIX_Y;
            p1.y = p1.y * M_2PIX_X;
            p2.x = p2.x * M_2PIX_Y;
            p2.y = p2.y * M_2PIX_X;
            p3.x = p3.x * M_2PIX_Y;
            p3.y = p3.y * M_2PIX_X;

            // Compute distance between all points
            double d12 = this->GetDistance(p1, p2);
            double d23 = this->GetDistance(p2, p3);
            double d31 = this->GetDistance(p3, p1);
            // Compute the area formed by all points
            double area = this->GetArea(p1, p2, p3);
            // Get tue curvure given three points
            curvature.push_back(4*area/(d12*d23*d31));
        }
    }

    // Dist from the center of the vehicle to the center of the lane
    this->dist_2center = this->left_line_pts[0].y + this->right_line_pts[this->right_line_pts.size()-1].y;
    this->dist_2center /= 2;
    this->dist_2center = abs(this->dist_2center - this->current_frame.cols/2);
    this->dist_2center *= M_2PIX_X;

    // Return the average curvure
    double sum  = accumulate(curvature.begin(), curvature.end(), 0.0);
    this->curvature_ref = sum/curvature.size();

    return;
}

void LaneDetector::AvgPoints()
{

    this->avg_left.push_back(this->left_line_pts);
    this->avg_right.push_back(this->right_line_pts);

    if(this->avg_left.size() >= AVG_LINES)
    {
        this->avg_left.erase(this->avg_left.begin());
        this->avg_right.erase(this->avg_right.begin());
    }

    for(size_t i = 0; i < this->avg_left.size()-1; i++ )
    {
        for(size_t j = 0; j < this->left_line_pts.size(); j++)
        {
            this->left_line_pts[j].x += this->avg_left[i][j].x;
            this->left_line_pts[j].y += this->avg_left[i][j].y;
            this->right_line_pts[j].x += this->avg_right[i][j].x;
            this->right_line_pts[j].y += this->avg_right[i][j].y;
        }
    }
    for(size_t j = 0; j < this->left_line_pts.size(); j++)
    {
        this->left_line_pts[j].x /= (this->avg_left.size());
        this->left_line_pts[j].y /= (this->avg_left.size());
        this->right_line_pts[j].x /= (this->avg_right.size());
        this->right_line_pts[j].y /= (this->avg_right.size());
    }

    return;
}

void LaneDetector::DrawLanes(Mat &_img)
{
    cv::Mat debug_img(this->mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    // concatenate all lane points in a single vector
    vector<Point> print_points;
    print_points.insert(print_points.begin(), this->left_line_pts.begin(), this->left_line_pts.end());
    print_points.insert(print_points.begin(), this->right_line_pts.begin(), this->right_line_pts.end());

    // Fill the space between the lines
    fillConvexPoly(debug_img, &print_points[0], print_points.size(), Scalar(200,0,0), 8, 0);

    // Draw lane lines
    cv::polylines(debug_img, this->left_line_pts, false, cv::Scalar(0, 255, 255), 15, 8, 0);
    cv::polylines(debug_img, this->right_line_pts, false, cv::Scalar(0, 255, 255), 15, 8, 0);
    rotate(debug_img.clone(), debug_img, ROTATE_90_COUNTERCLOCKWISE );

    // Apply inverse transformation to drawn image
    warpPerspective(debug_img.clone(), debug_img, this->T.inv(), debug_img.size(), INTER_LINEAR, BORDER_CONSTANT);

    // Draw curve and distance information to image
    putText(debug_img, "Radius of curvature: ", Point(20,50), FONT_HERSHEY_DUPLEX, 1, Scalar(50,0,250), 2);
    putText(debug_img, to_string(int(abs(1/this->curvature_ref)))+" m", Point(400,50), FONT_HERSHEY_DUPLEX, 1, Scalar(50,0,250), 2);
    putText(debug_img, "Distance to center: ", Point(20,100), FONT_HERSHEY_DUPLEX, 1, Scalar(50,0,250), 2);
    putText(debug_img, to_string(float(abs(this->dist_2center)))+" m", Point(340,100), FONT_HERSHEY_DUPLEX, 1, Scalar(50,0,250), 2);

    // Perform a weighted sum between original ans drawn image
    _img = _img + debug_img*0.5;

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