#include "../include/clasifier.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;


int main()
{

    Ptr<SVM> loaded_svm = Algorithm::load<SVM>("../trained_svm_2.xml");

    if(loaded_svm != NULL)
    {
        cout << "LOADED TRAINED CLASIFIER" << endl;
        // Load all images filenames
        vector<cv::String> file_cars, file_non_cars;
        glob("../data/test_data/positive/*.png", file_cars, false);
        glob("../data/test_data/negative/*.png", file_non_cars, false);

        cout << "Positive images: " << file_cars.size() << endl;
        cout << "Negative images:" << file_non_cars.size() << endl;

        int ok_count = 0;
        int nok_count = 0;
        
        for(String path: file_cars)
        {
            Mat test_sample;
            // Get feature vector for each image
            GetFeatureVector(imread(path), test_sample);
            // Predict between: vehicle and not vehicle
            if (loaded_svm->predict(test_sample))
            {
                ok_count++;
            }
        }

        for(String path: file_non_cars)
        {
            Mat test_sample;
            // Get feature vector for each image
            GetFeatureVector(imread(path), test_sample);
            // Predict between: vehicle and not vehicle
            if (!loaded_svm->predict(test_sample))
            {
                nok_count++;
            }
        }

        cout << "ACCURACY POS: " <<  (float(ok_count) - float(file_cars.size()))/float(file_cars.size()) << endl;
        cout << "ACCURACY NEG: " <<  (float(nok_count) - float(file_non_cars.size()))/float(file_non_cars.size()) << endl;
    }
    else
    {
        cout << "NEW CLASIFIER" << endl;
        // Load all images filenames
        vector<cv::String> file_cars, file_non_cars;
        glob("../data/train_data/positive/*.png", file_cars, true);
        glob("../data/train_data/negative/*.png", file_non_cars, true);

        int count = 0;
        Mat training_samples, labels;
        int no_normalized;
        // Get feature vector for each image ans concatenate it into a Matrix
        for(String path: file_cars)
        {
            GetFeatureVector(imread(path), training_samples);
            GenerateLabel(labels, true);
            cout << "Count vehicle: " << ++count << endl;
            //if (++count >= 1000) break;
        }
        count = 0;
        Mat training_samples_no;
        // Get feature vector for each image ans concatenate it into a Matrix
        for(String path: file_non_cars)
        {
            GetFeatureVector(imread(path), training_samples_no);
            GenerateLabel(labels, false);
            cout << "Count no-vehicle: " << ++count << endl;
            //if (++count >= 1000) break;
        }

        // Concat vehicles and non-vehicles vectors
        vconcat(training_samples, training_samples_no, training_samples);

        // Verbose: How many training examples
        cout << "training_samples size: " << training_samples.size()  << " cols: " << training_samples.cols << " rows: "  << training_samples.rows << endl;
        cout << "labels size: " << labels.size()  << " cols: " << labels.cols << " rows: "  << labels.rows << endl;

        // Create and configure SVM
        Ptr<SVM> svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::LINEAR);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));

        // Training process
        cout << "Starting training process" << endl;
        svm->train(TrainData::create(training_samples, ml::ROW_SAMPLE, labels));
        cout << "Finished training process" << endl;

        // Save the trained model into a XML file
        svm->save("../trained_svm_2.xml");        
    }

    return 0;
}
    