#include <iostream>
#include "BackPropagationMethod.h"
#include "NeuralNetwork.h"
#include "MNISTReader.h"

typedef BackPropagationMethod::TrainingSamplesType SamplesType;
typedef BackPropagationMethod::TrainingOuputsType  OutputsType;

int main (int argc, char** argv) {
    if (argc < 3) {
        std::cout << "ERROR input: " << std::endl;
        std::cout << "1: path to MNIST train-data " << std::endl;
        std::cout << "2: path to MNIST test-data " << std::endl;
        std::cout << "3: number of epochs (30 by default) " << std::endl;
        std::cout << "4: cross error (0.005 by default) " << std::endl;
        std::cout << "5: learn rate (0.01 by default) " << std::endl;
        std::cout << "6: number of neunrons in hidden layer (300 by default) " << std::endl;
        return 0;
    }

    std::string aPathToTrainSet = argv[1];
    std::string aPathToTestSet = argv[2];

    std::string aTrainImage = aPathToTrainSet + "train-images.idx3-ubyte";
    std::string aTrainLabel = aPathToTrainSet + "train-labels.idx1-ubyte";
    std::string aTestImage  = aPathToTestSet  + "t10k-images.idx3-ubyte";
    std::string aTestLabel  = aPathToTestSet  + "t10k-labels.idx1-ubyte";

    size_t aTrainImageCount = 60000;
    size_t aTestImageCount  = 10000;

    size_t aHiddenNeuronsNum = 300;
    size_t anEpochsNum = 30;
    double aLearnRate  = 0.01;
    double aCrossError = 0.005;
 
    switch (argc) {
        case 4: anEpochsNum       = std::atoi (argv[3]); break;
        case 5: anEpochsNum       = std::atoi (argv[3]); 
                aCrossError       = std::atof (argv[4]); break;
        case 6: anEpochsNum       = std::atoi (argv[3]); 
                aCrossError       = std::atof (argv[4]);
                aLearnRate        = std::atof (argv[5]); break;
        case 7: anEpochsNum       = std::atoi (argv[3]); 
                aCrossError       = std::atof (argv[4]);
                aLearnRate        = std::atof (argv[5]); 
                aHiddenNeuronsNum = std::atoi (argv[6]); break;
    }

    auto ReadMnist = [&] (const std::string& theDataPath, const std::string& theLabelPath, SamplesType& theSamples, OutputsType& theOutputs, size_t theCount) {
        std::vector<double> aLabels (theCount);

        ReadMNIST (theDataPath, theSamples);
        ReadMNISTLabel (theLabelPath, aLabels);

        theOutputs.resize (theCount);

        for (size_t i = 0; i < theCount; ++i) {
            theOutputs[i].resize (10, 0);
            theOutputs[i][static_cast <size_t> (aLabels[i])] = 1.0;
        }
    };

    size_t numInput = 28 * 28;
    size_t numOutput = 10;

    SamplesType aTrainSamples, aTestSamples;
    OutputsType aTrainOutputs, aTestOutputs;

    ReadMnist (aTrainImage, aTrainLabel, aTrainSamples, aTrainOutputs, aTrainImageCount);
    ReadMnist (aTestImage, aTestLabel, aTestSamples, aTestOutputs, aTestImageCount);

    BackPropagationMethod aTeacher (anEpochsNum, aLearnRate, aCrossError);
    NeuralNetwork aNetwork (numInput, aHiddenNeuronsNum, numOutput);
    aTeacher.Teach (aTrainSamples, aTrainOutputs, aNetwork);

    std::cout << "Train result: " << aTeacher.Accuracy (aTrainSamples, aTrainOutputs, aNetwork) << std::endl;
    std::cout << "Test result:" << aTeacher.Accuracy (aTestSamples, aTestOutputs, aNetwork) << std::endl;

    system ("PAUSE");
    return 0;
}
