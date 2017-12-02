#ifndef NeuralNetwork_HeaderFile
#define NeuralNetwork_HeaderFile

#include "Neuron.h"

class BackPropagationMethod;

class NeuralNetwork {
public:
    NeuralNetwork (size_t theInputSize, size_t theHiddenLayerSize, size_t theOutputSize);
    std::vector <double> Compute (const std::vector <double>& theInput);

private:
    std::vector <double> ComputeOutput (const std::vector <double>& theInput, size_t theLayerNum);

    typedef std::vector<Neuron>    LayerType;

    std::vector<LayerType> myLayers;

    friend BackPropagationMethod;
};


#endif
