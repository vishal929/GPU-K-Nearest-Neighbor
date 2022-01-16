// file houses gpu kernels for knn evaluation
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include<stdlib.h>
#include "dataUtil.h"

//device definition for iris dataset distance function (just euclidean distance of numeric elements)
__device__ double irisDistanceFunction(void** elementOne, void** elementTwo) {
	double distance = 0;
	distance += pow(*((double*)(elementOne[sepalLength])), 2);
	distance += pow(*((double*)(elementOne[sepalWidth])), 2);
	distance += pow(*((double*)(elementOne[petalLength])), 2);
	distance += pow(*((double*)(elementOne[petalLength])), 2);
	// taking square root
	return sqrt(distance);
}

// idea here is to have each thread in a warp compute the distance function, and then, do an ordering sort based on that (this implementation uses odd/even sort) 
// elementToCompare is the base element for knn
// dataset is the array of elements in the training dataset for knn
// distanceFunction is the device specified distance  function for the knn implementation
// distances is the array of distances for each element
// numElements is the number of elements in the training dataset (needed for iteration)
// k is the number of nearest neighbors to get
__global__ void computeDistances(void** elementToCompare, void*** dataset, double(*distanceFunction)(void** elementOne, void** elementTwo), double* distances, int numElements, int k) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// idea here is to just test with global memory and then do memory optimization after implementation is solid

	// each thread handles distance computation between 1 element and stores the distance in the distance array

	for (int i = tid;i < numElements;i += gridDim.x * blockDim.x) {
		// computing the distance function with this thread
		double distance = distanceFunction(elementToCompare, dataset[i]);
		//storing distance in the global array
		distances[i] = distance;
	}
	
	// now distances[i] holds the distance from elementToCompare to dataset[i]
}

// even sort as an implementation for knn after computing distances
// distances is the computed distance array
// elementPositions is the indices of each element 
// (this is so we know after sorting which elements in the dataset actually correspond to the k nearest neighbors, rather than just distance values) 
__global__ void evenSort(double* distances, int* elementPositions, int numElements) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// this does the even phase in one kernel
	for (int i = tid;i < (numElements - 1) / 2;i += gridDim.x * blockDim.x) {
		// the mapped even index is 2*tid
		if (distances[2 * tid] > distances[(2 * tid) + 1]) {
			// then we can swap them
		}
	}
}

// even sort as an implementation for knn after computing distances
// distances is the computed distance array
// elementPositions is the indices of each element 
// (this is so we know after sorting which elements in the dataset actually correspond to the k nearest neighbors, rather than just distance values) 
__global__ void oddSort(double* distances, int* elementPositions, int numElements) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// this does the odd phase in one kernel
}

// the idea here is that we launch k blocks, and each block of threads places the best neighbor among the block into global memory
// the resulting array is the k best approximate neighbors
__global__ void approximateKNN(void** elementToCompare, void*** dataset, double(*distanceFunction)(void** elementOne, void** elementTwo), double* distances, int* elementPositions, int numElements, int k) {

}
