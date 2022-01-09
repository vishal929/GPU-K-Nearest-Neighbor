#pragma once

// class declaration for kdTree object
template <typename K>
class KDTree {
	// KD Trees are just binary trees, but they are split on an axis of data
	public:
		// left child
		KDTree<K>* left;
		// right child
		KDTree<K>* right;
		// data element of the dataset used for the split
		void** dataElement;
		// integer of the feature index that was split
		int axisSplit;

		// constructor
		// input is a dataset, since we construct a kdtree by splitting the dataset by dimension
		// validAxes are valid feature indices to split on, since for example, we cant split a kdtree based on a string classification, or qualitative feature
		KDTree<K> KDTree(vector<void**> dataset, vector<int> validAxes) {

		}
};

// function declarations for kdtree knn implementation


