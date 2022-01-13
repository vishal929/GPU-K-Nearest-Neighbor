#include "myPriorityQueue.h"
using namespace std;
// this file will house logic for cpu k-nn search methodologies, namely naive knn, knn with kd trees, and a small worlds approach to k-nn search


// naive KNN compares every element in the dataset to the element for comparison, and gets the k closest neighbors
// for better comparison, we will use a max heap here
// little k is the number of nearest neighbors to return
// K is the type of the key (distance function result type)
// element is a pointer to the element type to compare to the dataset
// distanceFunction is a function pointer to the distance function to use
// dataset contains pointers to the elements in the dataset
// returns a pointer to the max-heap priority queue containing the k closest neighbors from the dataset to the element
template <class K>
PriorityQueue<K>* naiveKNN(int k, void** element, K (*distanceFunction)(void** elementOne, void** elementTwo), vector<void**> &dataset) {
	// creating our priority queue as a smart pointer
	PriorityQueue<K>* maxHeap = new PriorityQueue<K>(k, true);
	// iterating through the dataset and performing comparisons
	for (int i = 0;i < dataset.size();i++) {
		void** elementToCompare = dataset[i];
		K distance = distanceFunction(element, elementToCompare);
		if (maxHeap->getSize() < k) {
			maxHeap->insertElement(make_pair(elementToCompare, distance));
		}
		else {
			K topDistance = maxHeap->peek().second;
			if (topDistance > distance) {
				// then the worst neighbor in the maxHeap is worse than the currently compared neighbor -> we should replace them
				maxHeap->extractTop();
				maxHeap->insertElement(make_pair(elementToCompare, distance));
			}
		}
	}
	// returning the priority queue object with the nearest neighbors
	return maxHeap;
}

// an idea I had here where we can split up the dataset into n subsets and run knn on them, and them compile the results after parallel processing
// hope is that parallelism will benefit runtime, even if we do have extra comparisons that one pass does not have (1 thread naive version)
template<class K>
PriorityQueue<K>* parallelNaiveKNN(int k, void** element, K (*distanceFunction)(void** elementOne, void** elementTwo), vector<void**> &dataset) {

}

// sometimes our dataset will be very extensive, so we might have to batch knn results, below is the naive implementation for batched knn
// this will be generic in the definition, but I will probably assume a maximum memory of 6GB in my tests
template <class K>
PriorityQueue<K>* batchedNaiveKNN(int k, void** element, K(*distanceFunction)(void** elementOne, void** elementTwo), vector<void**>& dataset) {

}

