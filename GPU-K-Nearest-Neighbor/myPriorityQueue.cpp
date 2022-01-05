// file holds all the logic for data structures to use for k-nn when executing cpu code
#include "myPriorityQueue.h"

template<typename K>
void PriorityQueue<K>::siftDown(int position) {
	// comparisons with children, if appropriate
			int leftChild = 2 * position +1;
			int rightChild = leftChild + 1;
			// best is defined based on our comparison type
			int bestIndex = position;
			if (leftChild <= this->getSize() - 1) {
				// this is a valid comparison index
				if (this->comparisonType) {
					// then children should be larger than the parent
					(this->heapArray[leftChild].second < heapArray[position].second) ? bestIndex = leftChild : ;
				}
				else {
					// then children should be smaller than the parent
					(heapArray[leftChild].second > heapArray[position].second) ? bestIndex = leftChild : ;
				}
			}
			if (rightChild <= this->getSize() - 1) {
				// this is a valid comparison index
				if (this->comparisonType) {
					// then children should be larger than the parent
					(heapArray[rightChild].second< heapArray[position].second) ? bestIndex = rightChild : ;
				}
				else {
					// then children should be smaller than the parent
					(heapArray[rightChild].second> heapArray[position].second) ? bestIndex = rightChild : ;
				}
			}
			if (bestIndex != position) {
				// then we have to continue the siftDown operation after swapping
				pair<void*, K> temp = this->heapArray[position];;
				this->heapArray[position] = this->heapArray[bestIndex];
				this->heapArray[bestIndex] = temp;
				this->siftDown(bestIndex);
			}
}

// sift up operation on a specific index
template<typename K>
void PriorityQueue<K>::siftUp(int position) {
	if (position == 0) return;
	// getting parent index
	int parent = (position+1) / 2;
	if (this->comparisonType) {
		// then the parent should be <= the child
		if (this->heapArray[parent].second > this.heapArray[position].second) {
			// need to swap and continue sift up operation
			pair<void*, K> temp = this->heapArray[position];
			this->heapArray[position] = this->heapArray[parent];
			this->heapArray[parent] = temp;
			this->siftUp(parent);
		}
	}
	else {
		// then the parent should be >= the child
		if (this->heapArray[parent].second < this.heapArray[position].second) {
			// need to swap and continue sift up operation
			pair<void*, K> temp = this->heapArray[position];
			this->heapArray[position] = this->heapArray[parent];
			this->heapArray[parent] = temp;
			this->siftUp(parent);
		}
	}
}

// peeking the top element
template <typename K>
pair<void*, K> PriorityQueue<K>::peek() {
	if (this->getSize() == 0) return NULL;
	return this->heapArray[0];
}

// extracting the top element
template<typename K>
pair<void*, K> PriorityQueue<K>::extractTop() {
	if (this->getSize() == 0) return NULL;
	pair < void*, K> top = this->heapArray[0];
	// swapping top value with a leaf value, if possible
	if (this->getSize() == 1) {
		this->heapArray.clear()
	}
	else {
		this->heapArray[0] = this->heapArray[(this->heapArray).size() - 1];
		// removing last element
		(this->heapArray).delete((this->heapArray).end());
		// performing the sift down operation on the root
		this->siftdown(0);
	}

	return top;
}

// inserting elements
template<typename K>
void PriorityQueue<K>::insertElement(pair<void*,K> element) {
	this->heapArray.push_back(element);
	this->siftUp(this->heapArray.size() - 1);
}

// heapify operation given a vector of tuples representing the data and their key values
template<typename K>
void PriorityQueue<K>::heapify(vector<pair<void*,K>> elementKeyArray) {
	// we keep inserting elements as the last element and sifting down
	for (int i = 0;i < elementKeyArray.size();i++) {
		this->heapArray.push_back(element);
		this->siftdown(this->heapArray.size() - 1);
	}
}

