// file houses gpu kernels for knn evaluation
#include <stdio.h>
#include <cuda_runtime_api.h>
#include<stdlib.h>

//device definition for iris dataset distance function
__device__ double irisDistanceFunction(void** elementOne, void** elementTwo) {

}

// idea here is to have each thread in a warp compute the distance function, and then, do an ordering reduction based on that
__global__ void knn(void** elementToCompare, void*** dataset, double(*distanceFunction)(void** elementOne, void** elementTwo)) {

}
