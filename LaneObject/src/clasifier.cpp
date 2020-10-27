#include "../include/clasifier.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;

enum LABELS
{
    CARS,
    NO_CARS,
    ALL
};

void show(Mat _img){
    imshow("Debug image", _img);
    waitKey(0);
}

Mat GetHOGFeatures(Mat _img);

Mat GetColorHistogramFeatures(Mat _img);

Mat GetBinnedFeatures(Mat _img);

Mat GetFeatureVector(Mat _img);

int main()
{

    Ptr<SVM> loaded_svm = Algorithm::load<SVM>("../svm_data.xml");

    if(loaded_svm != NULL)
    {
        cout << "LOADED TRAINED CLASIFIER" << endl;

        // vector<cv::String> file_cars;
        // glob("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/data/vehicles/vehicles/*.png", file_cars, true);
        // int count = 0;
        // for(String path: file_cars)
        // {
        //     Mat test_case = GetFeatureVector(imread(path));
        //     Mat float_mat;

        //     // cout << test_case.type() << " " << test_case.cols << endl;
        //     // flip(test_case.clone(),test_case,1);
        //     // cout << test_case.type() << " " << test_case.cols << endl;

        //     cout << "svm->predic[" << count << "]: " << loaded_svm->predict(test_case);
        //     if (++count >= 1000) break;
        // }
    }
    else
    {
        cout << "NEW CLASIFIER" << endl;
    

        vector<vector<Mat>> feature_vectors(3);
        feature_vectors[CARS] = vector<Mat>();
        feature_vectors[NO_CARS] = vector<Mat>();

        vector<cv::String> file_cars, file_non_cars;
        glob("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/data/vehicles/vehicles/*.png", file_cars, true);
        glob("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/data/non-vehicles/non-vehicles/*.png", file_non_cars, true);

        int counr = 0;
        for(String path: file_cars)
        {
            feature_vectors[CARS].push_back(GetFeatureVector(imread(path)));
            if (++counr >= 1000) break;
        }
        counr = 0;
        for(String path: file_non_cars)
        {
            feature_vectors[NO_CARS].push_back(GetFeatureVector(imread(path)));
            if (++counr >= 1000) break;
        }

        int size = feature_vectors[CARS].size();
        int cols = feature_vectors[CARS][0].cols;
        int rows = feature_vectors[CARS][0].rows;

        Mat training_samples;//(rows, size, CV_32F);
        Mat labels;
        hconcat(feature_vectors[CARS][0], feature_vectors[CARS][1], training_samples);
        hconcat(Mat(1,1, CV_32S, Scalar(1)), Mat(1,1, CV_32S, Scalar(1)), labels);
        for(size_t i = 2; i < size; i++)
        {
            hconcat(training_samples, feature_vectors[CARS][i], training_samples);
            hconcat(labels, Mat(1,1, CV_32S, Scalar(1)), labels);
        }

        for(size_t i = 0; i < size; i++)
        {
            hconcat(training_samples, feature_vectors[NO_CARS][i], training_samples);
            hconcat(labels, Mat(1,1, CV_32S, Scalar(0)), labels);
        }

        cols = labels.cols;
        rows = labels.rows;
        
        cout << "training_samples size: " << training_samples.size()  << " cols: " << training_samples.cols << " rows: "  << training_samples.rows << endl;
        cout << "labels size: " << size  << " cols: " << cols << " rows: "  << rows << endl;

        Ptr<SVM> svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setC(0.1);
        svm->setKernel(SVM::LINEAR);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));

        cout << "Starting training process" << endl;
        svm->train(training_samples, COL_SAMPLE, labels);
        cout << "Finished training process" << endl;

        svm->save("../svm_data.xml");

        // int vector_size = feature_vectors[CARS][0].rows + feature_vectors[NO_CARS][0].rows;
        // int n_trainging_vectors = feature_vectors[CARS].size() + feature_vectors[NO_CARS].size();

        // vconcat(feature_vectors[CARS], feature_vectors[NO_CARS], feature_vectors[ALL]);


        // const int NTRAINING_SAMPLES = 100;         // Number of training samples per class
        // const float FRAC_LINEAR_SEP = 0.9f;        // Fraction of samples which compose the linear separable part
        //--------------------- 1. Set up training data randomly ---------------------------------------
        // Mat trainData(2*NTRAINING_SAMPLES, 2, CV_32F);
        // Mat labels   (2*NTRAINING_SAMPLES, 1, CV_32S);
        //------------------------- Set up the labels for the classes ---------------------------------
        // labels.rowRange(                0,   NTRAINING_SAMPLES).setTo(1);  // Class 1
        // labels.rowRange(NTRAINING_SAMPLES, 2*NTRAINING_SAMPLES).setTo(2);  // Class 2
        //------------------------ 2. Set up the support vector machines parameters --------------------
        // cout << "Starting training process" << endl;
        // Ptr<SVM> svm = SVM::create();
        // svm->setType(SVM::C_SVC);
        // svm->setC(0.1);
        // svm->setKernel(SVM::LINEAR);
        // svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
        //------------------------ 3. Train the svm ----------------------------------------------------
        //svm->train(trainData, ROW_SAMPLE, labels);
        // cout << "Finished training process" << endl;

        //Ptr<SVM> svm2 = Algorithm::load<SVM>("../svm_data.xml");
        // Mat sampleMat = (Mat_<float>(1,2) << 500, 500);
        //cout << svm->predict(sampleMat) << endl;

        //svm->save("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/svm_data.xml");
        
        
    }

    return 0;
}
    

Mat GetHOGFeatures(Mat _img)
{   
    Mat mat_descriptor;
    HOGDescriptor hog;
    vector<float> v_desc;
    hog.winSize = _img.size();

    cvtColor(_img.clone(), _img, CV_RGB2GRAY);

    hog.compute(_img, v_desc, Size(4, 4), Size(0, 0));

    mat_descriptor = Mat(v_desc).clone();

    return mat_descriptor;
}

Mat GetColorHistogramFeatures(Mat _img)
{   
    Mat descriptor;

    cvtColor(_img.clone(), _img, CV_RGB2YCrCb);
    vector<Mat> channels;
    split( _img, channels);

    int hist_size = _img.cols;
    float range[] = { 0, 256 };
    const float* hist_Range = { range };

    Mat hist_0c, hist_1c, hist_2c;
    calcHist( &channels[0], 1, 0, Mat(), hist_0c, 1, &hist_size, &hist_Range, true, false );
    calcHist( &channels[1], 1, 0, Mat(), hist_1c, 1, &hist_size, &hist_Range, true, false );
    calcHist( &channels[2], 1, 0, Mat(), hist_2c, 1, &hist_size, &hist_Range, true, false );

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/hist_size );
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    // normalize(hist_0c, hist_0c, 1, 0, NORM_L2, -1, Mat() );
    // normalize(hist_1c, hist_1c, 1, 0, NORM_L2, -1, Mat() );
    // normalize(hist_2c, hist_2c, 1, 0, NORM_L2, -1, Mat() );

    // Normalize histograms
    hist_0c = hist_0c * 1/4096;
    hist_1c = hist_1c * 1/4096;
    hist_2c = hist_2c * 1/4096;

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

    vconcat(hist_0c, hist_1c, descriptor);
    vconcat(descriptor, hist_2c, descriptor);

    return descriptor;
}

Mat GetBinnedFeatures(Mat _img)
{   
    Mat descriptor;
    Mat res_img;

    _img.convertTo(_img, CV_32FC1);

    vector<Mat> channels;
    resize(_img.clone(), _img, cv::Size(), 0.5, 0.5);
    split( _img, channels);

    channels[0] = channels[0].reshape(1, channels[0].cols * channels[0].rows);
    channels[1] = channels[1].reshape(1, channels[1].cols * channels[1].rows);
    channels[2] = channels[2].reshape(1, channels[2].cols * channels[2].rows);

    vconcat(channels[0], channels[1], descriptor);
    vconcat(descriptor, channels[2], descriptor);

    descriptor = descriptor / 255;

    return descriptor;
}

Mat GetFeatureVector(Mat _img)
{
    Mat feature_vector;

    Mat hog_f = GetHOGFeatures(_img);

    Mat hist_f = GetColorHistogramFeatures(_img);

    Mat bin_f = GetBinnedFeatures(_img);

    vconcat(hog_f, hist_f, feature_vector);
    vconcat(feature_vector, bin_f, feature_vector);

    return feature_vector;
}