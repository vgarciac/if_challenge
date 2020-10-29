#include "../include/clasifier.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;


int main()
{

    Ptr<SVM> loaded_svm = Algorithm::load<SVM>("../svm_trained_test.xml");

    if(loaded_svm != NULL)
    {
        cout << "LOADED TRAINED CLASIFIER" << endl;

        vector<cv::String> file_cars, file_non_cars;
        glob("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/data/test_data/positive/*.png", file_cars, false);
        glob("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/data/test_data/negative/*.png", file_non_cars, false);

        cout << file_cars.size() << endl;
        cout << file_non_cars.size() << endl;

        int ok_count = 0;
        int nok_count = 0;
        
        for(String path: file_cars)
        {
            Mat test_sample;
            GetFeatureVector(imread(path), test_sample);
            //cout << "svm->predic[" << count << "]: " << loaded_svm->predict(test_sample) << endl;
            if (loaded_svm->predict(test_sample))
            {
                ok_count++;
            }
        }

        for(String path: file_non_cars)
        {
            Mat test_sample;
            GetFeatureVector(imread(path), test_sample);
            if (!loaded_svm->predict(test_sample))
            {
                nok_count++;
            }
        }

        cout << "ACCURACY POS: " <<  float(ok_count)/float(file_cars.size()) << endl;
        cout << "ACCURACY NEG: " <<  float(nok_count)/float(file_non_cars.size()) << endl;
    }
    else
    {
        cout << "NEW CLASIFIER" << endl;

        vector<vector<Mat>> feature_vectors(3);
        feature_vectors[CARS] = vector<Mat>();
        feature_vectors[NO_CARS] = vector<Mat>();

        vector<cv::String> file_cars, file_non_cars;
        glob("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/data/train_data/positive/*.png", file_cars, true);
        glob("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/data/train_data/negative/*.png", file_non_cars, true);

        int counr = 0;
        Mat training_samples;
        Mat labels;
        int no_normalized;
        for(String path: file_cars)
        {
            no_normalized = GetFeatureVector(imread(path), training_samples);
            GenerateLabel(labels, true);
            cout << "Count vehicle: " << ++counr << endl;
            //if (++counr >= 10) break;
        }
        counr = 0;
        Mat training_samples_no;
        for(String path: file_non_cars)
        {
            GetFeatureVector(imread(path), training_samples_no);
            GenerateLabel(labels, false);
            cout << "Count no-vehicle: " << ++counr << endl;
            //if (++counr >= 10) break;
        }

        vconcat(training_samples, training_samples_no, training_samples);
        Mat mean_mtx(1, no_normalized, CV_32F);
        Mat std_dev_mtx(1, no_normalized, CV_32F);

        Mat means, sigmas;
        for (size_t i = 0; i < no_normalized; i++)
        {
            Mat mean, sigma;
            //cout << training_samples << endl;
            meanStdDev(training_samples.col(i), mean, sigma);
            //cout << training_samples << endl;
            //training_samples.col(i) = (test_ - mean[0])/stddev[0];
            means.push_back(mean);
            sigmas.push_back(sigma);

            //cout << "Normalisation: " << i << endl;
            training_samples.col(i) = (training_samples.col(i) - mean) / sigma;
        }
        Mat meansigma;
        hconcat(means, sigmas, meansigma);  //both params in same matrix

        //cout << mean_mtx << endl;
        cout << "mean_mtx rows: " << mean_mtx.rows << endl;
        FileStorage fs;
        fs.open("../normalisation.xml", FileStorage::WRITE);

        fs << "meansigma" << meansigma;
        
        // cv::Scalar mean, stddev;
        // cv::meanStdDev(hist, mean, stddev);

        //cout << labels << endl;
        cout << "training_samples Type: " << training_samples.type() << endl;
        cout << "labels Type: " << labels.type() << endl;
        cout << "training_samples size: " << training_samples.size()  << " cols: " << training_samples.cols << " rows: "  << training_samples.rows << endl;
        cout << "labels size: " << labels.size()  << " cols: " << labels.cols << " rows: "  << labels.rows << endl;

        Ptr<SVM> svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setC(0.1);
        svm->setKernel(SVM::LINEAR);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));

        cout << "Starting training process" << endl;
        svm->trainAuto(TrainData::create(training_samples, ml::ROW_SAMPLE, labels));
        cout << "Finished training process" << endl;

        // const int NTRAINING_SAMPLES = 100;         // Number of training samples per class
        // const float FRAC_LINEAR_SEP = 0.9f;        // Fraction of samples which compose the linear separable part
        //--------------------- 1. Set up training data randomly ---------------------------------------
        // Mat trainData(2*NTRAINING_SAMPLES, 2, CV_32F);
        // Mat labels   (2*NTRAINING_SAMPLES, 1, CV_32S);
        //------------------------- Set up the labels for the classes ---------------------------------
        // labels.rowRange(                0,   NTRAINING_SAMPLES).setTo(1);  // Class 1
        // labels.rowRange(NTRAINING_SAMPLES, 2*NTRAINING_SAMPLES).setTo(2);  // Class 2
        //------------------------ 2. Set up the support vector machines parameters --------------------
        //svm->train(trainData, ROW_SAMPLE, labels);
        // cout << "Finished training process" << endl;
        //Ptr<SVM> svm2 = Algorithm::load<SVM>("../svm_data.xml");
        // Mat sampleMat = (Mat_<float>(1,2) << 500, 500);
        //cout << svm->predict(sampleMat) << endl;

        svm->save("/home/blanco-deniz.julio-cesar/if_challenge/LaneObject/svm_trained_test.xml");        
    }

    return 0;
}
    