#include "dataUtil.h"
using namespace std;
// file hosts logic for defining distance functions and elements of different datasets used for testing
// the datasets are NOT included in my github repo, but below are the links to download them yourself
// after download, you can just drop the datasets into the main folder and run the test to see results


// first dataset is the iris dataset, available at the following link:
// https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
// this is a relatively small dataset, where we will probably see the gpu implementation being close to the cpu one or differences being unsubstantial


// function to read lines from the iris dataset and grab the data (dataset is small, so we can grab all the data in one go)
// returns a pointer to an array of void pointers, which can be casted to the specific types
// iris data types go like:
// 1) sepal length in cm -> double
// 2) sepal width in cm -> double
// 3) petal length in cm -> double
// 4) petal width in cm -> double
// 5) string of classification of the iris flower -> string
vector<void**> grabIrisData() {
	// the dataset is newline delimited	
	// we will just use streams to get the data
	vector<void**> dataVector;
	ifstream irisFile("iris.data");
	double sepalLengthVal;
	double sepalWidthVal;
	double petalLengthVal;
	double petalWidthVal;
	string classificationVal;
	string buffer;
	while (getline(irisFile, buffer)) {
		istringstream stringParse(buffer);
		if (!(stringParse >> sepalLengthVal >> "," >> sepalWidthVal >> "," >> petalLengthVal >> "," >> petalWidthVal >> "," >> classificationVal)) break;
		// mallocing data and pushing to our vector
		double* sepalLengthPointer = (double*)malloc(sizeof(double));
		*sepalLengthPointer = sepalLengthVal;
		double* sepalWidthPointer = (double*)malloc(sizeof(double));
		*sepalWidthPointer = sepalLengthVal;
		double* petalLengthPointer = (double*)malloc(sizeof(double));
		*petalLengthPointer = sepalLengthVal;
		double* petalWidthPointer = (double*)malloc(sizeof(double));
		*petalWidthPointer = sepalLengthVal;
		// extra char for null terminator
		char* classificationString = (char*)malloc(sizeof(char) * (buffer.size() + 1));
		strcpy(classificationString, classificationVal.c_str());
		// allocating outer array pointers
		void** element = (void**)malloc(sizeof(void*) * 5);
		element[sepalLength] = (void*) sepalLengthPointer;
		element[sepalWidth] = (void*) sepalWidthPointer;
		element[petalLength] = (void*) petalLengthPointer;
		element[petalWidth] = (void*) petalWidthPointer;
		element[classification] = (void*) classificationString;
		// pushing to vector
		dataVector.push_back(element);
	}
	// returning the vector
	return dataVector;
}

void freeIrisData(vector<void**> dataVector) {
	for (int i = 0;i < dataVector.size();i++) {
		void** element = dataVector[i];
		//freeing inner elements
		free(((double*)element[sepalLength]));
		free(((double*)element[sepalWidth]));
		free(((double*)element[petalLength]));
		free(((double*)element[petalWidth]));
		free(((char*)element[classification]));
		//freeing outer pointer
		free(element);
	}
}

void freeIrisData(void*** dataArray, int numElements) {
	for (int i = 0;i < numElements;i++) {
		// freeing inner elements first, then outer element pointer
		void** element = dataArray[i];
		//freeing inner elements
		free(((double*)element[sepalLength]));
		free(((double*)element[sepalWidth]));
		free(((double*)element[petalLength]));
		free(((double*)element[petalWidth]));
		free(((char*)element[classification]));
		//freeing outer pointer
		free(element);
	}
	// freeing entire array pointer
	free(dataArray);
}

double irisDistanceFunction(void** elementOne, void** elementTwo) {
	// we will just use euclidean distance here
	double distance = 0;
	distance += pow(*((double*)(elementOne[sepalLength])), 2);
	distance += pow(*((double*)(elementOne[sepalWidth])), 2);
	distance += pow(*((double*)(elementOne[petalLength])), 2);
	distance += pow(*((double*)(elementOne[petalLength])), 2);
	// taking square root
	return sqrt(distance);
}


// second dataset is
// this is much larger dataset, but still able to be completely fit into memory
// we will probably see a much larger divide in cpu vs gpu performance here

// the third dataset is
// the catch with this dataset is that it is not able to be fit into my memory limit of 5GB
// so, we will see interesting results here for batched cpu implementations vs batched gpu implementations
