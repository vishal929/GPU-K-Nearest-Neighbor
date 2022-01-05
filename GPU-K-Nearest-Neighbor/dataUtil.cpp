// file hosts logic for defining distance functions and elements of different datasets used for testing
// the datasets are NOT included in my github repo, but below are the links to download them yourself
// after download, you can just drop the datasets into the main folder and run the test to see results


// first dataset is the iris dataset, available at the following link:
// https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
// this is a relatively small dataset, where we will probably see the gpu implementation being close to the cpu one or differences being unsubstantial

// second dataset is
// this is much larger dataset, but still able to be completely fit into memory
// we will probably see a much larger divide in cpu vs gpu performance here

// the third dataset is
// the catch with this dataset is that it is not able to be fit into my memory limit of 5GB
// so, we will see interesting results here for batched cpu implementations vs batched gpu implementations
