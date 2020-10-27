//Polynomial Fit algorithm
//taked from: https://www.programmersought.com/article/50515052663/
#include "../include/lane_lines.hpp"
#include<iomanip>
#include<cmath>

using namespace cv;
using namespace std;

bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
    //Number of key points
    int N = key_point.size();
    
    //Construct matrix X
    cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            for (int k = 0; k < N; k++)
            {
                X.at<double>(i, j) = X.at<double>(i, j) +
                    std::pow(key_point[k].x, i + j);
            }
        }
    }
    
    //Construct matrix Y
    cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int k = 0; k < N; k++)
        {
            Y.at<double>(i, 0) = Y.at<double>(i, 0) +
                std::pow(key_point[k].x, i) * key_point[k].y;
        }
    }
    
    A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    //Solve matrix A
    cv::solve(X, Y, A, cv::DECOMP_SVD);
    return true;
}