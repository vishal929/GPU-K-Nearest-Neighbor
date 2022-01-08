#pragma once
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
// header for data utility functions

// access enums for the iris elements
enum iris {sepalLength, sepalWidth, petalLength, petalWidth, classification};

// grabs iris dataset as an array of pointers to pointers to void*
vector<void**> grabIrisData();

// freeing iris data, given a c++ style vector implementation
void freeIrisData(vector<void**> dataVector);

// freeing iris data, given a c style array implementation (used for the gpu kernels)
void freeIrisData(void*** dataArray, int numElements);

// defined distance function for iris data
double irisDistanceFunction(void** elementOne, void** elementTwo);
