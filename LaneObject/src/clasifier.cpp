#include "../include/clasifier.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;


void show(Mat _img, int ms){
    imshow("Debug image", _img);
    waitKey(ms);
}


Mat get_hogdescriptor_visu(Mat& origImg, vector<float>& descriptorValues);

Mat GetHOGFeatures(Mat _img)
{   
    // Create HOG object and set the parameters
    HOGDescriptor hog;
    hog.winSize = _img.size();
    hog.cellSize = Size(HOG_SZ,HOG_SZ);

    // Convert to another color space (default YcrCb)
    Mat color_c, gray;
    cvtColor(_img, color_c, CV_RGB2YCrCb);
    cvtColor(_img, gray, CV_RGB2GRAY);

    // Split color channels and compute HOG for each one
    vector<Mat> channels;
    split( color_c, channels);
    vector<vector<float>> v_desc(3);
    hog.compute(channels[0], v_desc[0], Size(HOG_SZ, HOG_SZ), Size(0, 0));
    hog.compute(channels[1], v_desc[1], Size(HOG_SZ, HOG_SZ), Size(0, 0));
    hog.compute(channels[2], v_desc[2], Size(HOG_SZ, HOG_SZ), Size(0, 0));
    
    // Convert descriptor vectors to cv::Mat objects
    channels[0] = Mat(v_desc[0]).clone();
    channels[1] = Mat(v_desc[1]).clone();
    channels[2] = Mat(v_desc[2]).clone();

    // Concatenate HOG vectors in one single array (Mat nx1)
    Mat mat_descriptor;
    vconcat(channels[0], channels[1], mat_descriptor);
    vconcat(mat_descriptor, channels[2], mat_descriptor);
    
    //Mat test = get_hogdescriptor_visu(gray, v_desc[0]);
    // Convert to 1xn matrix
    mat_descriptor = mat_descriptor.reshape(1, 1);

    return mat_descriptor;
}

Mat GetColorHistogramFeatures(Mat _img)
{   
    Mat descriptor;
    Mat yrcb;

    // Convert to another color space (default YcrCb)
    cvtColor(_img, yrcb, CV_RGB2YCrCb);
    vector<Mat> channels;
    split( yrcb, channels);

    // define histogram parameters
    int hist_size = yrcb.cols;
    float range[] = { 0, 256 };
    const float* hist_Range = { range };

    // Commpute histogram for each color channel
    vector<Mat> hist_channel(3);
    calcHist( &channels[0], 1, 0, Mat(), hist_channel[0], 1, &hist_size, &hist_Range, true, false );
    calcHist( &channels[1], 1, 0, Mat(), hist_channel[1], 1, &hist_size, &hist_Range, true, false );
    calcHist( &channels[2], 1, 0, Mat(), hist_channel[2], 1, &hist_size, &hist_Range, true, false );

    // Variables for histogram visualisation
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/hist_size );
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    // Normalize values of histogram (between 0-1)
    // normalize(hist_channel[0], hist_channel[0], 1, 0, NORM_L2 , -1, Mat() );
    // normalize(hist_channel[1], hist_channel[1], 1, 0, NORM_L2 , -1, Mat() );
    // normalize(hist_channel[2], hist_channel[2], 1, 0, NORM_L2 , -1, Mat() );

    // Concatenate histogram vectors in one single array (Mat nx1)
    vconcat(hist_channel[0], hist_channel[1], descriptor);
    vconcat(descriptor, hist_channel[2], descriptor);
    // Convert to 1xn matrix
    descriptor = descriptor.reshape(1, 1);

    normalize(descriptor, descriptor, 1, 0, NORM_L2, -1, Mat());

    // cout << hist_channel[0];
    // show(_img);

    return descriptor;
}

Mat GetBinnedFeatures(Mat _img)
{   
    Mat descriptor;
    Mat res_img;

    // Split color channels
    vector<Mat> channels;
    resize(_img, res_img, cv::Size(), 0.5, 0.5);
    res_img.convertTo(res_img, CV_32FC1);
    split( res_img, channels);

    // Convert to 1xn matrix
    channels[0] = channels[0].reshape(1, 1);
    channels[1] = channels[1].reshape(1, 1);
    channels[2] = channels[2].reshape(1, 1);

    // Concatenate histogram vectors in one single array
    hconcat(channels[0], channels[1], descriptor);
    hconcat(descriptor, channels[2], descriptor);

    //Normalize values of histogram (between 0-1)
    // normalize(descriptor, descriptor, 1, 0, NORM_L2 , -1, Mat() );

    descriptor = (descriptor - 125) / 125;

    return descriptor;
}

int GetFeatureVector(Mat _img, Mat &_vector)
{
    Mat feature_vector;

    // Compute HOG features vector
    Mat hog_f = GetHOGFeatures(_img);

    // Compute Color Histogram vector
    Mat hist_f = GetColorHistogramFeatures(_img);

    // Compute Spatial Binning vector
    Mat bin_f = GetBinnedFeatures(_img);

    // Concatenate all three vectors in a single one
    hconcat(hog_f, hist_f, feature_vector);
    int non_normalized_sz = hog_f.cols;
    hconcat(hog_f, bin_f, feature_vector);

    // Check to assign or concatenated the computed vector with the input vector
    if(_vector.empty())
    {
        _vector = feature_vector;
    }
    else
    {
        vconcat(_vector, feature_vector, _vector);
    }

    return non_normalized_sz;
}

Mat GenerateLabel(Mat &_vector, bool _label)
{
    Mat new_label(1,1, CV_32S, Scalar(_label));

    if(_vector.empty())
    {
        _vector = new_label;
    }
    else
    {
         vconcat(_vector, Mat(1,1, CV_32S, Scalar(_label)), _vector);
    }

    return _vector;
}

// Taked from: https://github.com/blacksoil/HOGVisualizer/blob/master/testCV/main.cpp
// Help function to visualize HOG vector as image
Mat get_hogdescriptor_visu(Mat& origImg, vector<float>& descriptorValues)
{
    Mat color_origImg;
    cvtColor(origImg, color_origImg, CV_GRAY2RGB);
    
    float zoomFac = 3;
    Mat visu;
    resize(color_origImg, visu, Size(color_origImg.cols*zoomFac, color_origImg.rows*zoomFac));
    
    int blockSize       = 16;
    int cellSize        = 16;
    int gradientBinSize = 9;
    float radRangeForOneBin = M_PI/(float)gradientBinSize; // dividing 180° into 9 bins, how large (in rad) is one bin?
    
    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = 64 / cellSize;
    int cells_in_y_dir = 64 / cellSize;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
            
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
    
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
    
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
    
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
                
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
                    
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
                    
                } // for (all bins)
                
                
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;
                
            } // for (all cells)
            
            
        } // for (all block x pos)
    } // for (all block y pos)
    
    
    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
            
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
    
    
    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
    
    // draw cells
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize;
            int drawY = celly * cellSize;
            
            int mx = drawX + cellSize/2;
            int my = drawY + cellSize/2;
            
            rectangle(visu, Point(drawX*zoomFac,drawY*zoomFac), Point((drawX+cellSize)*zoomFac,(drawY+cellSize)*zoomFac), CV_RGB(100,100,100), 1);
            
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
                
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
                
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
                
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = cellSize/2;
                float scale = 2.5; // just a visualization scale, to see the lines better
                
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
                
                // draw gradient visualization
                line(visu, Point(x1*zoomFac,y1*zoomFac), Point(x2*zoomFac,y2*zoomFac), CV_RGB(180,180,180), 1);
                
            } // for (all bins)
            
        } // for (cellx)
    } // for (celly)
    
    
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
        for (int x=0; x<cells_in_x_dir; x++)
        {
            delete[] gradientStrengths[y][x];            
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
    
    return visu;
    
} // get_hogdescriptor_visu

// Code for histogram visualisation
//cout << hist_2c << endl;
//show(_img);

// Normalize histograms
// hist_0c = hist_0c * 1/4096;
// hist_1c = hist_1c * 1/4096;
// hist_2c = hist_2c * 1/4096;

// for( int i = 1; i < hist_size; i++ )
// {
//     line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist_0c.at<float>(i-1)) ),
//           Point( bin_w*(i), hist_h - cvRound(hist_0c.at<float>(i)) ),
//           Scalar( 255, 0, 0), 2, 8, 0  );
//     line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist_1c.at<float>(i-1)) ),
//           Point( bin_w*(i), hist_h - cvRound(hist_1c.at<float>(i)) ),
//           Scalar( 0, 255, 0), 2, 8, 0  );
//     line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist_2c.at<float>(i-1)) ),
//           Point( bin_w*(i), hist_h - cvRound(hist_2c.at<float>(i)) ),
//           Scalar( 0, 0, 255), 2, 8, 0  );
// }
// imshow("Source image", _img );
// imshow("calcHist Demo", histImage );
// waitKey();
