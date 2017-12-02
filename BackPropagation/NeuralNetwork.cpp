#include "NeuralNetwork.h"
#include "Functions.h"

NeuralNetwork::NeuralNetwork (size_t theInputSize, size_t theHiddenLayerSize, size_t theOutputSize) {
    myLayers.reserve (2);

    auto CreateLayer = [&] (size_t theIS, size_t theSize) {
        std::vector<Neuron> aNeurons;
        aNeurons.reserve (theSize);

        for (size_t i = 0; i < theSize; ++i) {
            aNeurons.push_back (Neuron (theIS));
        }

        myLayers.push_back (std::move (aNeurons));
    };
    CreateLayer (theInputSize, theHiddenLayerSize);
    CreateLayer (theHiddenLayerSize, theOutputSize);
}

std::vector <double> NeuralNetwork::ComputeOutput (const std::vector <double>& theInputs, size_t theLayerNum) {
    std::vector <double> anOutput;
    anOutput.reserve (myLayers[theLayerNum].size());

    for (size_t i = 0; i < myLayers[theLayerNum].size(); ++i) {
        anOutput.push_back (myLayers[theLayerNum][i].GetOutput (theInputs));
    }
    return anOutput;
}

std::vector <double> NeuralNetwork::Compute (const std::vector<double>& theInputs) {
    const auto& aFirstOutput = HyperTan::GetValue (ComputeOutput (theInputs, 0));
    return SoftMax::GetValue (ComputeOutput (aFirstOutput, 1));
}
