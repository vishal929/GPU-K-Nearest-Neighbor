#pragma once
#include <stdlib.h>
#include <vector>
#include <any>
using namespace std;

// K is the class for the key
// elements are void**  (these are not relevant to the priority queue until data extraction, so I will keep them as void**, we do not want to copy tuple objects anyway)
template<typename K>
class PriorityQueue {
	public:
		// our heaps will have a maximum size (for bounding knn)
		int maxSize;
		// vector of tuple representing the element and a double key
		*vector< pair<void**, K>> heapArray;
		// need to specify whether this is a maxHeap or a minHeap
		bool comparisonType;
		/*Function declarations*/
		// false for min heap, true for max heap
		//constructor
		PriorityQueue<K> PriorityQueue(int maxSize, bool comparisonType) {
			this->maxSize = 0;
			this->comparisonType = comparisonType;
			//smart pointer for the heap vector
			this->heapArray = ptr(new vector<pair<void**, K>>());
		}
		
		// getting the size of the heapArray currently
		int getSize() {
			return this->heapArray->size();
		}

		// returns 1 if empty, 0 if not empty
		bool isEmpty() {
			return this->getSize() == 0;
		}

		//sift down operation on a specific index
		void siftDown(int position);

		// sift up operation on a specific index
		void siftUp(int position);

		// peeking the top element
		pair<void**, K> peek();

		// extracting the top element
		pair<void**, K> extractTop();

		// inserting elements
		void insertElement(pair<void**, K> element);

		// heapify operation given a vector of tuples representing the data and their key values
		void heapify(vector<pair<void**, K>> elementKeyArray);

};
