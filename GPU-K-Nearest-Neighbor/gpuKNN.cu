// file houses gpu kernels for knn evaluation
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include<stdlib.h>
#include "dataUtil.h"

//device definition for iris dataset distance function
__device__ double irisDistanceFunction(void** elementOne, void** elementTwo) {
	double distance = 0;
	distance += pow(*((double*)(elementOne[sepalLength])), 2);
	distance += pow(*((double*)(elementOne[sepalWidth])), 2);
	distance += pow(*((double*)(elementOne[petalLength])), 2);
	distance += pow(*((double*)(elementOne[petalLength])), 2);
	// taking square root
	return sqrt(distance);
}

// idea here is to have each thread in a warp compute the distance function, and then, do an ordering reduction based on that
// elementToCompare is the base element for knn
// dataset is the array of elements in the training dataset for knn
// distanceFunction is the device specified distance  function for the knn implementation
// distances is the array of distances for each element
// elementPositions so that we keep track of which indices correspond to which distances
// numElements is the number of elements in the training dataset (needed for iteration)
// k is the number of nearest neighbors to get
__global__ void knn(void** elementToCompare, void*** dataset, double(*distanceFunction)(void** elementOne, void** elementTwo), double* distances, int* elementPositions, int numElements, int k) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// idea here is to just test with global memory and then do memory optimization after implementation is solid

	// each thread handles distance computation between 1 element and stores the distance in the distance array

	for (int i = tid;i < numElements;i += gridDim.x * blockDim.x) {
		// computing the distance function with this thread
		double distance = distanceFunction(elementToCompare, dataset[i]);
		//storing distance in the global array
		distances[i] = distance;
	}

	// syncing threads
	__sync_threads();

	// doing distance reduction to get the k nearest neighbors

	
}
