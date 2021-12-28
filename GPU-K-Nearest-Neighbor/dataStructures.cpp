// file holds all the logic for data structures to use for k-nn (mostly for cpu side)
#include <stdlib.h>
using namespace std;
// need a max heap to find elements of LARGEST distance away, so they can be thrown away, and we need a min heap to find the LOWEST distance best neighbors
template <typename ...Ts>
class priorityQueue{
	public:
		// heaps have a size	
		int size;
		// internal implementation will be an array, of fixed size
		// first element will be a pointer to the data tuple
		// second element is the distance or the key for the priority queue
		tuple<tuple<...Ts>*, double>* heapArray;
		// if 0, we have a min heap, if 1, we have a max heap
		int comparator;	
		// constructor for the priority queue
		// we only need a maximum size for the array and a comparison type (min or max heap) to create the priority queue
		void create(int maxSize, int comparisonType, Ts...) {
			this->size = 0;
			heapArray = malloc(sizeof(tuple<tuple<Ts...>*, double>) * maxSize);
			this->comparator = comparisonType;
		}
		// returns 1 if empty, 0 otherwise
		int isEmpty() {
			return this->size == 0;
		}
		// sift down operation
		void siftDown(int index);
		// sift up operation
		void siftUp(int index);
		// insert element into our priority queue
		void insertElement(tuple<Ts>* element) {

		}
		// peek the minimum or maximum element of the priority queue
		// this peeks the TUPLE at the top
		void peekTop() {
			return this.heapArray[0];
		}
		// extract the top element and its distance from the queue, we will replace with a leaf if applicable and sift down
		// if no leaf, the space is kept as NULL and the size is decremented
		tuple<tuple<Ts>*, double>* extractTop() {
			
		}
		
};
