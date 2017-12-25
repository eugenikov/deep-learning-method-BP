#include <iostream>

#include "BackPropagationMethod.h"
#include "NeuralNetwork.h"
#include "Functions.h"


namespace {
size_t MaxIndex (const std::vector<double>& theInput) {
    size_t aMaxIndex = 0;
    double aMaxValue = theInput[0];
    
    for (size_t i = 0; i < theInput.size(); ++i) {
        if (theInput[i] > aMaxValue) {
            aMaxValue = theInput[i];
            aMaxIndex = i;
        }
    }
    return aMaxIndex;
}
}
double BackPropagationMethod::Accuracy (const TrainingSamplesType& testData, const TrainingOuputsType& theOutputs, NeuralNetwork& theNetwork) {
    size_t aCorrectNum = 0;
    size_t aWrongNum = 0;

    for (size_t i = 0; i < testData.size(); ++i) {
        const auto& anOutput = theNetwork.Compute (testData[i]);
        size_t anIdx = MaxIndex (anOutput); 

        if (std::abs (theOutputs[i][anIdx] - 1.0) < 1e-3) {
           aCorrectNum++;
        } else {
           aWrongNum++;
        }
    }
    return static_cast <double> (aCorrectNum) / (aCorrectNum + aWrongNum);
}

void BackPropagationMethod::Init (size_t theHiddenOuputNum, size_t theOuputNum) {
    myCurrentHiddenGradient.resize (theHiddenOuputNum, 0);
    myCurrentOutputFromHiddenLayer.resize (theHiddenOuputNum, 0);
    myCurrentLastGradient.resize (theOuputNum, 0);
}


void BackPropagationMethod::UpdateSynapses (const InputType& theInputs, NeuralNetwork& theNetwork, double aLearnRate) {
    for (size_t i = 0; i < theNetwork.myLayers[1].size(); ++i) {
        for (size_t j = 0; j < theNetwork.myLayers[1][i].mySynapses.size(); ++j) {
            theNetwork.myLayers[1][i].mySynapses[j] += aLearnRate * myCurrentLastGradient[i] * myCurrentOutputFromHiddenLayer[j];
        }
    }

    for (size_t i = 0; i < theNetwork.myLayers[0].size(); ++i) {
        for (size_t j = 0; j < theNetwork.myLayers[0][i].mySynapses.size(); ++j) {
            theNetwork.myLayers[0][i].mySynapses[j] += aLearnRate * myCurrentHiddenGradient[i] * theInputs[j];
        }
    }
}

void BackPropagationMethod::UpdateBiases (NeuralNetwork& theNetwork, double aLearnRate) {
    for (size_t i = 0; i < theNetwork.myLayers[1].size(); i ++) {
        theNetwork.myLayers[1][i].myBias += aLearnRate * myCurrentLastGradient[i];
    }

    for (size_t i = 0; i < theNetwork.myLayers[0].size(); i++) {
        theNetwork.myLayers[0][i].myBias += aLearnRate * myCurrentHiddenGradient[i];
    }
}

void BackPropagationMethod::ComputeGradient (const OutputType& theOutput, const OutputType& theTargetOutput, NeuralNetwork& theNetwork) {

    for (size_t i = 0; i < theOutput.size(); ++i) {
        myCurrentLastGradient[i] = (theTargetOutput[i] - theOutput[i]);
    }

    for (size_t i = 0; i < myCurrentOutputFromHiddenLayer.size(); ++i) {
        double aDerivative = (1 - myCurrentOutputFromHiddenLayer[i]) * (1 + myCurrentOutputFromHiddenLayer[i]);
        double aSum = 0.0;
        for (size_t j = 0; j < myCurrentLastGradient.size(); ++j) {
            aSum += myCurrentLastGradient[j] * theNetwork.myLayers[1][j].mySynapses[i];
        }
        myCurrentHiddenGradient[i] = aDerivative * aSum;
    }
}

void BackPropagationMethod::Teach (const TrainingSamplesType& theSamples, const TrainingOuputsType& theOutputs, NeuralNetwork& theNetwork) {
    std::cout << "training..." << std::endl;

    Init (theNetwork.myLayers[0].size(), theNetwork.myLayers[1].size());

    std::cout << "Count of samples: " << theSamples.size() << std::endl;

    auto ComputeError = [&] () {
        double aSumError = 0;
        for (size_t i = 0; i < theSamples.size(); ++i) {
            const auto& anOutput = theNetwork.Compute (theSamples[i]);
            for (size_t j = 0; j < anOutput.size(); ++j) {
                aSumError += std::log (anOutput[j]) * theOutputs[i][j];
            }
        }
        return -aSumError / theSamples.size();
    };

    for (size_t j = 0; j < myEpochsNum; ++j) {
        for (size_t i = 0; i < theSamples.size(); ++i) {
            std::cout << i << " of " << theSamples.size() << "\r";
            myCurrentOutputFromHiddenLayer = HyperTan::GetValue (theNetwork.ComputeOutput (theSamples[i], 0));
            const auto& anOuput = SoftMax::GetValue (theNetwork.ComputeOutput (myCurrentOutputFromHiddenLayer, 1));
            ComputeGradient (anOuput, theOutputs[i], theNetwork);
            UpdateSynapses (theSamples[i], theNetwork, myLearnRate);
            UpdateBiases (theNetwork, myLearnRate);
        }

        double anError = ComputeError();
        std::cout << "Error: " << anError << std::endl;
        if (anError < myCrossError) { 
            break;
        }
        std::cout << "Continue training.." << std::endl;
    }
    std::cout << std::endl;
}
