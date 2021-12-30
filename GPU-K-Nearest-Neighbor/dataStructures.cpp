// file holds all the logic for data structures to use for k-nn (mostly for cpu side)
#include <stdlib.h>
using namespace std;
// need a max heap to find elements of LARGEST distance away, so they can be thrown away, and we need a min heap to find the LOWEST distance best neighbors
// K is the type of the key, ... Ts are the types of the element tuples
template <class K, class ... Ts> 
class PriorityQueue {
	public:
		// heaps have a size
		int size;
		// vector of tuple representing the element and a double key
		vector< pair<tuple<...Ts>, K>> heapArray;
		// need to specify whether this is a maxHeap or a minHeap
		bool comparisonType;
		/*Function declarations*/
		// false for min heap, true for max heap
		PriorityQueue<K,...Ts> PriorityQueue(int maxSize, bool comparisonType) {
			this->size = 0;
			this->comparisonType = comparisonType;
		}
		// returns 1 if empty, 0 if not empty
		bool isEmpty() {
			return this->size == 0;
		}
		//sift down operation on a specific index
		void siftDown(int position) {
			// comparisons with children, if appropriate
			int leftChild = 2 * position +1;
			int rightChild = leftChild + 1;
			// best is defined based on our comparison type
			int bestIndex = position;
			if (leftChild <= this->size - 1) {
				// this is a valid comparison index
				if (this->comparisonType) {
					// then children should be larger than the parent
					(heapArray[leftChild].second.value < heapArray[position].second.value) ? bestIndex = leftChild : ;
				}
				else {
					// then children should be smaller than the parent
					(heapArray[leftChild].second.value > heapArray[position].second.value) ? bestIndex = leftChild : ;
				}
			}
			if (rightChild <= this->size - 1) {
				// this is a valid comparison index
				if (this->comparisonType) {
					// then children should be larger than the parent
					(heapArray[rightChild].second.value < heapArray[position].second.value) ? bestIndex = rightChild : ;
				}
				else {
					// then children should be smaller than the parent
					(heapArray[rightChild].second.value > heapArray[position].second.value) ? bestIndex = rightChild : ;
				}
			}
			if (bestIndex != position) {
				// then we have to continue the siftDown operation after swapping
				pair<tuple<...Ts>, K> temp = this->heapArray[position];;
				this->heapArray[position] = this->heapArray[bestIndex];
				this->heapArray[bestIndex] = temp;
				this->siftDown(bestIndex);
			}
				
		}

		// sift up operation on a specific index
		void siftUp(int position) {
			if (position == 0) return;
			// getting parent index
			int parent = (position+1) / 2;
			if (this->comparisonType) {
				// then the parent should be <= the child
				if (this->heapArray[parent].second.value > this.heapArray[position].second.value) {
					// need to swap and continue sift up operation
					pair<tuple<...Ts>, K> temp = this->heapArray[position];
					this->heapArray[position] = this->heapArray[parent];
					this->heapArray[parent] = temp;
					this->siftUp(parent);
				}
			}
			else {
				// then the parent should be >= the child
				if (this->heapArray[parent].second.value < this.heapArray[position].second.value) {
					// need to swap and continue sift up operation
					pair<tuple<...Ts>, K> temp = this->heapArray[position];
					this->heapArray[position] = this->heapArray[parent];
					this->heapArray[parent] = temp;
					this->siftUp(parent);
				}
			}
		}

		// peeking the top element
		pair<tuple<...Ts>, K> peek() {
			if (this->size == 0) return NULL;
			return this->heapArray[0];
		}

		// extracting the top element
		pair<tuple<...Ts>, K> extractTop() {
			if (this->size == 0) return NULL;
			pair < tuple<...Ts, K> top = this->heapArray[0];
			// swapping top value with a leaf value, if possible
			if (this->size == 1) {
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
		void insertElement(pair<tuple<...Ts>,K> element) {
			this->heapArray.push_back(element);
			this->siftUp(this->heapArray.size() - 1);
		}

		// heapify operation given a vector of tuples representing the data and their key values
		void heapify(vector<pair<tuple<...Ts>,K>> elementKeyArray) {
			// we keep inserting elements as the last element and sifting down
			for (int i = 0;i < elementKeyArray.size();i++) {
				this->heapArray.push_back(element);
				this->siftdown(this->heapArray.size() - 1);
			}
		}

};
