#ifndef Neuron_HeaderFile
#define Neuron_HeaderFile

#include <vector>
#include <memory>
#include "Functions.h"

class BackPropagationMethod;

class Neuron {
public:

    Neuron (size_t theInputNum);
    double GetOutput (const std::vector<double>& theInput) const;

private:
    friend BackPropagationMethod;

    std::vector <double> mySynapses;
    double myBias;

};

#endif
