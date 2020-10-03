
// SVM : https://stackoverflow.com/questions/14694810/using-opencv-and-svm-with-images 

#include "opencv2/imgcodecs.hpp"
#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <utility>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <stdio.h>
#include <bitset>
#include <cstdio>
#include <ctime>
#include <tuple>


using namespace cv;
using namespace ml;
using namespace std;

vector<string> folders;
int IMAGE_WIDTH = 32;
int IMAGE_HEIGHT = 94; //Get these from the function?


/*
*   An object of class Picture contains an image (Mat), the label of this image and its path.
*/
class Picture {
public:

    Mat image;          // Image of a character
    Mat outline;        // Outline of the image (answer)
    string filename;    // Path to the image


    Picture(Mat& image, Mat& outline, string filename) : image(image), outline(outline), filename(filename) {};
};

/*
*   Read and prepare the image for training.
*   That includes resizing the matrix and converting it to a float 1-channel.
*/
bool loadImage(string imagePath, Mat& outputImage) {

    Mat image = imread(imagePath, IMREAD_GRAYSCALE);

    // Check for invalid input
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return false;
    }

    //resize(image, image, Size(IMAGE_WIDTH, IMAGE_HEIGHT));
    padImage(image, IMAGE_WIDTH, IMAGE_HEIGHT);
    // Convert to float 1-channel
    image.convertTo(outputImage, CV_32FC1, 1.0 / 255.0);

    return true;
}

/*
* Add to the top and right side pixels
*/
void padImage(Mat& padMe, int newWidth, int newHeight) {

    Scalar value(0, 0, 0); //Can I put another 0 for transparency? Who knows :) Test this in the main and save to see please!

    int widthDifference = padMe.size().width - newWidth;
    if (widthDifference <= 0) {
        copyMakeBorder(padMe, padMe, 0, 0, 0, widthDifference, BORDER_CONSTANT, value);
    }
    else {
        cout << "Image width too large already" << endl;
    }

    int heightDifference = padMe.size().width - newHeight;
        if (heightDifference <= 0) {
            copyMakeBorder(padMe, padMe, newHeight, 0, 0, 0, BORDER_CONSTANT, value);
        }
        else {
            cout << "Image height too large already" << endl;
        }
}
/*
*   Create a Picture object for each image from the given folder and adds it to a vector.
*/
vector<Picture> loadImages(vector<string> folderNames) {

    vector<Picture> images;

    for (string folder : folderNames) {
        vector<String> imagePaths;
        glob(folder, imagePaths, false);                  // list of all files path in folder

        for (int i = 0; i < imagePaths.size(); i += 2) {
            Mat image; //0
            Mat outline; //1
            loadImage(imagePaths[i], image);
            loadImage(imagePaths[i + 1], outline);
            images.emplace_back(image, outline, imagePaths[i]);
        }
    }

    return images;
}


/*
*   Divide the dataset into two vectors : training data and testing data.
*/
void getTrainingAndTestingDataVectors(vector<Picture>& data,
    vector<Picture>& trainingData, vector<Picture>& testingData) {

    int ratio = 3;

    for (int i = 0; i < data.size(); ++i) {
        if (i % ratio == 0) {
            testingData.push_back(data[i]);
        }
        else trainingData.push_back(data[i]);
    }

    cout << trainingData.size() << " images ready for training." << endl;
    cout << testingData.size() << " images ready for testing.\n" << endl;
}

/*
*   Get matrix containing all the images from the given vector of Picture objects.
*/
Mat getInputData(vector<Picture>& images) {

    Mat imagesData;

    for (Picture img : images) {
        Mat imgDataInOneRow = img.image.reshape(0, 1);
        imagesData.push_back(imgDataInOneRow);
    }

    return imagesData;
}

/*
*   Get matrix of outlines corresponding to each image in the given Picture vector.
*/
Mat getMLPOutputOutlinesData(vector<Picture>& images) {

    Mat imagesData;

    for (Picture pic : images) {

        //Mat imgDataInOneRow(pic.outline, false);

        imagesData.push_back(pic.outline.reshape(0, 1)); //Used to be imgDataInOneRow

    }
    return imagesData;
}



/*
*   Create a MLP model with the given hidden layer size and training datta.
*/
Ptr<ANN_MLP> createMLP(const int HIDDEN_LAYER_SIZE, Mat& inputTrainingData, Mat& outputTrainingData) {

    // Create and set up MLP architecture
    Ptr<ANN_MLP> mlp = ANN_MLP::create();

    Mat layersSize = Mat(3, 1, CV_16U);
    layersSize.row(0) = Scalar(inputTrainingData.cols);
    layersSize.row(1) = Scalar(HIDDEN_LAYER_SIZE);
    layersSize.row(2) = Scalar(outputTrainingData.cols);

    mlp->setLayerSizes(layersSize);
    mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM, 1.0, 1.0);
    mlp->setTrainMethod(ANN_MLP::BACKPROP, 0.0001);

    TermCriteria termCrit = TermCriteria(TermCriteria::MAX_ITER, 100, 0.000001);
    mlp->setTermCriteria(termCrit);

    return mlp;
}


/*
*   Train a model with the given input data.
*   Return the trained model.
*/
Ptr<StatModel>& trainModel(Ptr<StatModel>& model, Mat& inputTrainingData, Mat& outputTrainingData) {

    cout << "Starting training ..." << endl;

    Ptr<TrainData> trainingData = TrainData::create(inputTrainingData, SampleTypes::ROW_SAMPLE, outputTrainingData);
    model->train(trainingData);

    return model;
}

/*
* Compare two images by their difference in pixels by ratio e.g. 0.5 = 50% the same
*/
bool compareMatrices(Mat& image1, Mat& image2, float confidence) {
    //Count and return the number of white pixels / black pixels / transparent pixels in a % confidence

    double width = image1.size().width;
    double height = image1.size().height;

    double width2 = image2.size().width;
    double height2 = image2.size().height;

    if (width != width2 || height != height2) return false;

    int errors = 0;

        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < width; ++y) {
                auto image1colour = image1.at<Vec3b>(x, y);
                auto image2colour = image2.at<Vec3b>(x, y);
                if (image1colour != image2colour) { //Only black and white so no need for absolute +/- of RGB values
                    ++errors;
                }
            }
        }
    if (errors / width * height <= confidence) return true;

    return false;
    
}
/*
*   Compare each prediction with its corresponding answer.
*   Save all the mistakes to a vector.
*/
void interpretMLPResults(string& path, Mat& results, Mat& answer, vector<tuple<Mat, Mat, string>>& predictionErrors) {


    if (!compareMatrices(results, answer, 0.05)) {
        predictionErrors.emplace_back(results, answer, path);
    }

}
/**
* Input a folder name/path, return the largest width/height of images in that folder
*/
tuple<int, int> getMaxWidthHeight(string folderName) {

    int maxWidth;
    int maxHeight;

    vector<String> imageNames;
    glob(folderName, imageNames, false);

    for (int i = 0; i < imageNames.size(); ++i) {

        Mat currImage = imread(imageNames[i], IMREAD_GRAYSCALE);

        if (currImage.rows > maxHeight) maxHeight = currImage.rows;
            
        if (currImage.cols > maxWidth) maxWidth = currImage.cols;
        
    }

    tuple<int, int> maxTuple(maxWidth, maxHeight);

    return maxTuple;
}

/**
*   Test the input model with the given vector of Picture objects.
*   Update the out vector with the failed predictions made by the model.
*
*/
void testModel(Ptr<StatModel>& model, vector<Picture>& testImages, vector<tuple<Mat, Mat, string>>& out) {

    cout << "Starting testing ..." << endl;

    Mat inputTestingData = getInputData(testImages);

    for (int i = 0; i < inputTestingData.rows; ++i) {
        Mat result;
        Picture& pic = testImages[i];

        model->predict(pic.image.reshape(0, 1), result);
        interpretMLPResults(pic.filename, result, pic.outline, out); //I changed here
    }

}


/*
*   Write the statistics of the current run to a text file.
*   It includes the number of images for training and testing, the number of wrong predictions and their details.
*/
void writeStats(string modelName, int trainImages, int testImages, int duration, vector<tuple<Mat, Mat, string>>& predictionErrors) {

    ofstream outfile;
    outfile.open(modelName + "_Stats.txt", ios_base::app);

    outfile << ">> RESULTS FOR " + modelName << endl;
    outfile << "\nDataset : " << endl;
    outfile << "\nTime taken : " << to_string(duration) << " seconds." << endl;

    for (auto f : folders) outfile << f << endl;

    outfile << "\nThe model was trained on " << to_string(trainImages) << " files." << endl;
    outfile << "\nThe model was tested on " << to_string(testImages) << " files." << endl;
    outfile << "\nThe model made " << predictionErrors.size() << " mistakes out of " << to_string(testImages) << " testing files." << endl;


}

/**
*   Run the MLP training and testing.
*   Return the vector containing all prediction errors.
*
*/
vector<tuple<Mat, Mat, string>> runMLP(string modelName, vector<Picture>& trainImages, vector<Picture>& testImages) {

    Mat inputTrainingData = getInputData(trainImages);          // Mat of all training images
    Mat outputTrainingData = getMLPOutputOutlinesData(trainImages);     // Mat of all training image labels

    //Ptr<StatModel> model = ANN_MLP::load("model.xml");                              // Uncomment to load a MLP
    Ptr<StatModel> model = createMLP(75, inputTrainingData, outputTrainingData);      // Uncomment to create a new MLP and indicate number of neurons

    trainModel(model, inputTrainingData, outputTrainingData);             // TRAINING
    model->save(modelName + ".xml");                                     // Uncomment to save the MLP as an xml

    vector<tuple<Mat, Mat, string>> predictionErrors;
    testModel(model, testImages, predictionErrors);

    return predictionErrors;

}



int main() {

    // Start a clock
    clock_t start;
    double duration;
    start = clock();

    // Dataset
    folders = { "datasets/highAltitudeDust/*.tiff" };

    string modelName = "MLP_FULL";      // Name of the model 

    /** Load training and testing images */
    vector<Picture> data = loadImages(folders);
    vector<Picture> trainImages;
    vector<Picture> testImages;
    getTrainingAndTestingDataVectors(data, trainImages, testImages);

    /** Run the model and save predictions */
    vector<tuple<Mat, Mat, string>> predictionErrors = runMLP(modelName, trainImages, testImages);

    /** Save statistics */
    duration = (clock() - start) / (double)CLOCKS_PER_SEC;
    writeStats(modelName, trainImages.size(), testImages.size(), duration, predictionErrors);

    return 0;

}
