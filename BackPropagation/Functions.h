#ifndef Function_HeaderFile
#define Function_HeaderFile

#include <vector>

//class Function {
//public:
//    virtual std::vector<double> GetValue (const std::vector<double>& theInput) const = 0;
//};


class SoftMax {
public:
    static std::vector<double> GetValue (const std::vector<double>& theInput);
};

class HyperTan {
public:
    static std::vector<double> GetValue (const std::vector<double>& theInput);
};

#endif
