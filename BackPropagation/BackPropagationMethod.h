#ifndef BackPropagationMethod_HeaderFile
#define BackPropagationMethod_HeaderFile

#include <vector>
#include "NeuralNetwork.h"
class NeuralNetwork;

class BackPropagationMethod {
public:
    typedef std::vector <double>     InputType;
    typedef std::vector <InputType>  TrainingSamplesType;
    typedef std::vector <double>     OutputType;
    typedef std::vector <OutputType> TrainingOuputsType;
    
    BackPropagationMethod (size_t theEpochsNum = 30, double theLearnRate = 0.01, double theCrossError = 0.005) :
        myEpochsNum (theEpochsNum), myLearnRate (theLearnRate), myCrossError (theCrossError) {}

    void Teach (const TrainingSamplesType& theSamples, const TrainingOuputsType& theOutputs, NeuralNetwork& theNetwork);

    double Accuracy (const TrainingSamplesType& testData, const TrainingOuputsType& theOutputs, NeuralNetwork& theNetwork);
private:

    void ComputeGradient (const OutputType& theOutput, const OutputType& theTargetOutput, NeuralNetwork& theNetwork);
    void UpdateSynapses (const InputType& theInputs, NeuralNetwork& theNetwork, double aLearnRate);
    void UpdateBiases (NeuralNetwork& theNetwork, double aLearnRate);

    void Init (size_t theOuputNum, size_t theHiddenOuput);

    size_t myEpochsNum;
    double myLearnRate;
    double myCrossError;
    std::vector <double> myCurrentLastGradient;
    std::vector <double> myCurrentHiddenGradient;
    std::vector <double> myCurrentOutputFromHiddenLayer;
};
#endif
