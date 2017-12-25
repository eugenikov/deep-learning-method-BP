#include "Neuron.h"
#include <cassert>

Neuron::Neuron (size_t theInputNum) {
    auto Random = [] () {
        return (rand() / double(RAND_MAX)) / 100.0;
    };

    mySynapses.resize (theInputNum);

    for (size_t i = 0; i < mySynapses.size(); ++i) {
        mySynapses[i] = Random();
    }

    myBias = Random();
}

double Neuron::GetOutput (const std::vector<double>& theInput) const {
    assert (theInput.size() == mySynapses.size());
    assert (theInput.size() != 0);

    double aSum = 0;
    for (size_t i = 0; i < theInput.size(); ++i) {
        aSum += theInput[i] * mySynapses[i];
    }

   return aSum + myBias;
}
