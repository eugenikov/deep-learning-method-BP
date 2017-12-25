#include "Functions.h"
#include <cmath>

std::vector<double>  SoftMax::GetValue (const std::vector<double>& theInput) {
    double aMax = theInput[0];

    for (size_t i = 1; i < theInput.size(); ++i) {
        if (theInput[i] > aMax) {
            aMax = theInput[i];
        }
    }

    double aScale = 0.0;
    for (size_t i = 0; i < theInput.size(); ++i) {
        aScale += std::exp (theInput[i] - aMax);
    }

    std::vector <double> anOutput (theInput.size());

    for (size_t i = 0; i < theInput.size(); ++i) {
        anOutput[i] = std::exp (theInput[i] - aMax) / aScale;
    }
    return anOutput;
}


std::vector<double> HyperTan::GetValue (const std::vector<double>& theInput) {
    std::vector<double> anOutput (theInput.size());

    for (size_t i = 0; i < theInput.size(); ++i) {
        double aValue;
        if (theInput[i] < -20.0) {
            aValue =  -1.0;
        } else if (theInput[i] > 20.0) {
            aValue =  1.0;
        } else {
            aValue = std::tanh(theInput[i]);
        }
        anOutput[i] = aValue;
    }
    return anOutput;
}

