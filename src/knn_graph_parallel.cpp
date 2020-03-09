#include <Rcpp.h>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <queue>
#include <limits>
using namespace Rcpp;

#include <omp.h>
// [[Rcpp::plugins(openmp)]]

#define NUM_THREADS(N) ((N) >= 0 ? (N) : omp_get_num_procs() + (N) + 1)

namespace knnparallel {
/*
 *  from
 *  vptree.h
 *  Implementation of a vantage-point tree.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 */

class DataPoint
{
  int _D;
  int _ind;
  double* _x;
  
public:
  DataPoint() {
    _D = 1;
    _ind = -1;
    _x = NULL;
  }
  DataPoint(int D, int ind, double* x) {
    _D = D;
    _ind = ind;
    _x = (double*) malloc(_D * sizeof(double));
    for (int d = 0; d < _D; d++) _x[d] = x[d];
  }
  DataPoint(const DataPoint& other) {                     // this makes a deep copy -- should not free anything
    if (this != &other) {
      _D = other.dimensionality();
      _ind = other.index();
      _x = (double*) malloc(_D * sizeof(double));
      for (int d = 0; d < _D; d++) _x[d] = other.x(d);
    }
  }
  ~DataPoint() { if (_x != NULL) free(_x); }
  DataPoint& operator= (const DataPoint& other) {         // asignment should free old object
    if (this != &other) {
      if (_x != NULL) free(_x);
      _D = other.dimensionality();
      _ind = other.index();
      _x = (double*) malloc(_D * sizeof(double));
      for (int d = 0; d < _D; d++) _x[d] = other.x(d);
    }
    return *this;
  }
  int index() const { return _ind; }
  int dimensionality() const { return _D; }
  double x(int d) const { return _x[d]; }
};


double euclidean_distance(const DataPoint &t1, const DataPoint &t2) {
  double dd = .0;
  for (int d = 0; d < t1.dimensionality(); d++) {
    dd += (t1.x(d) - t2.x(d)) * (t1.x(d) - t2.x(d));
  }
  dd = sqrt(dd);
  return dd;
}

double cosine_distance(const DataPoint &t1, const DataPoint &t2) {
  double dd = .0;
  for(int d = 0; d < t1.dimensionality(); d++) dd += t1.x(d) * t2.x(d) ;
  double norm = .0;
  for(int d = 0; d < t1.dimensionality(); d++) norm += t1.x(d)*t1.x(d) ;
  dd /=sqrt(norm);
  norm = .0;
  for(int d = 0; d < t2.dimensionality(); d++) norm += t2.x(d)*t2.x(d) ;
  dd /=sqrt(norm);
  dd =1-dd;
  dd = fabs(dd);
  return dd;
}

double precomputed_distance(const DataPoint &t1, const DataPoint &t2) {
  double dd = .0;
  dd = t1.x(t2.index());
  return dd;
}

template<typename T>
class VpTree
{
public:
  
  // Default constructor
  VpTree(std::function<double (const T&, const T&)> f) : _root(0), distance(f) {}
  
  // Destructor
  ~VpTree() {
    delete _root;
  }
  
  // Function to create a new VpTree from data
  void create(const std::vector<T>& items) {
    delete _root;
    _items = items;
    _root = buildFromPoints(0, items.size());
  }
  
  // Function that uses the tree to find the k nearest neighbors of target
  void search(const T& target, int k, std::vector<T>* results, std::vector<double>* distances)
  {
    
    // Use a priority queue to store intermediate results on
    std::priority_queue<HeapItem> heap;
    
    // Variable that tracks the distance to the farthest point in our results
    double tau = DBL_MAX;
    
    // Perform the searcg
    search(_root, target, k, heap, tau);
    
    // Gather final results
    results->clear(); distances->clear();
    while (!heap.empty()) {
      results->push_back(_items[heap.top().index]);
      distances->push_back(heap.top().dist);
      heap.pop();
    }
    
    // Results are in reverse order
    std::reverse(results->begin(), results->end());
    std::reverse(distances->begin(), distances->end());
  }
  
private:
  std::vector<T> _items;
  std::function<double (const T&, const T&)> distance;
  // Single node of a VP tree (has a point and radius; left children are closer to point than the radius)
  struct Node
  {
    int index;              // index of point in node
    double threshold;       // radius(?)
    Node* left;             // points closer by than threshold
    Node* right;            // points farther away than threshold
    
    Node() :
      index(0), threshold(0.), left(0), right(0) {}
    
    ~Node() {               // destructor
      delete left;
      delete right;
    }
  }* _root;
  
  
  // An item on the intermediate result queue
  struct HeapItem {
    HeapItem( int index, double dist) :
    index(index), dist(dist) {}
    int index;
    double dist;
    bool operator<(const HeapItem& o) const {
      return dist < o.dist;
    }
  };
  
  // Distance comparator for use in std::nth_element
  struct DistanceComparator
  {
    const T& item;
    std::function<double (const T&, const T&)> distance;
    DistanceComparator(const T& item, std::function<double (const T&, const T&)> f) : item(item), distance(f) {}
    bool operator()(const T& a, const T& b) {
      return distance(item, a) < distance(item, b);
    }
  };
  
  // Function that (recursively) fills the tree
  Node* buildFromPoints( int lower, int upper )
  {
    Rcpp::RNGScope scope;
    if (upper == lower) {     // indicates that we're done here!
      return NULL;
    }
    
    // Lower index is center of current node
    Node* node = new Node();
    node->index = lower;
    
    if (upper - lower > 1) {      // if we did not arrive at leaf yet
      
      // Choose an arbitrary point and move it to the start
      int i = (int) ((double)R::runif(0,1) * (upper - lower - 1)) + lower;
      std::swap(_items[lower], _items[i]);
      
      // Partition around the median distance
      int median = (upper + lower) / 2;
      std::nth_element(_items.begin() + lower + 1,
                       _items.begin() + median,
                       _items.begin() + upper,
                       DistanceComparator(_items[lower],distance));
      
      // Threshold of the new node will be the distance to the median
      node->threshold = distance(_items[lower], _items[median]);
      
      // Recursively build tree
      node->index = lower;
      node->left = buildFromPoints(lower + 1, median);
      node->right = buildFromPoints(median, upper);
    }
    
    // Return result
    return node;
  }
  
  // Helper function that searches the tree
  void search(Node* node, const T& target, int k, std::priority_queue<HeapItem>& heap, double& tau)
  {
    if (node == NULL) return;    // indicates that we're done here
    
    // Compute distance between target and current node
    double dist = distance(_items[node->index], target);
    
    // If current node within radius tau
    if (dist < tau) {
      if (heap.size() == k) heap.pop();                // remove furthest node from result list (if we already have k results)
      heap.push(HeapItem(node->index, dist));           // add current node to result list
      if (heap.size() == k) tau = heap.top().dist;    // update value of tau (farthest point in result list)
    }
    
    // Return if we arrived at a leaf
    if (node->left == NULL && node->right == NULL) {
      return;
    }
    
    // If the target lies within the radius of ball
    if (dist < node->threshold) {
      if (dist - tau <= node->threshold) {        // if there can still be neighbors inside the ball, recursively search left child first
        search(node->left, target, k, heap, tau);
      }
      
      if (dist + tau >= node->threshold) {        // if there can still be neighbors outside the ball, recursively search right child
        search(node->right, target, k, heap, tau);
      }
      
      // If the target lies outsize the radius of the ball
    } else {
      if (dist + tau >= node->threshold) {        // if there can still be neighbors outside the ball, recursively search right child first
        search(node->right, target, k, heap, tau);
      }
      
      if (dist - tau <= node->threshold) {         // if there can still be neighbors inside the ball, recursively search left child
        search(node->left, target, k, heap, tau);
      }
    }
  }
};
}

using namespace knnparallel;

// inspired by Multicore-TSNE by Dmitry Ulynov 
// see https://github.com/DmitryUlyanov/Multicore-TSNE/blob/master/multicore_tsne/tsne.cpp

Rcpp::List openmp_knn_list(
    double* X,                                                      // coordinates
    const int N,                                                    // vertex count (row count)
    int D,                                                          // dimension (column count)
    int**    _ind_arr,                                              // K nearest neighbours indices
    double** _dist_arr,                                             // distances
    int K,                                                          // neighbour count, including origin (!)
    std::function<double (const DataPoint&, const DataPoint&)> dist // distance function
) {
  *_ind_arr  = (int*)    calloc(N * K, sizeof(int));
  *_dist_arr = (double*) calloc(N * K, sizeof(double));
  if (*_ind_arr == NULL || *_dist_arr == NULL)
  {
    Rcpp::Rcout << "Memory allocation failed." << std::endl;
    return Rcpp::List("fail");
  }
  
  int*    ind_arr = *_ind_arr;
  double* dist_arr = *_dist_arr;
  
  // build vantage-point tree on data set
  VpTree<DataPoint> tree(dist);
  
  std::vector<DataPoint> vertices(N, DataPoint(D, -1, X)); // vertices
  for (int n = 0; n < N; n++)
    vertices[n] = DataPoint(D, n, X + n * D);
  tree.create(vertices);
  
  Rcpp::Rcout << "Vantage-point tree built." << std::endl;
  
  // loop over all points to find nearest neighbors
  int steps_completed = 0;
  
  Rcpp::Rcout << "Initiating nearest-neighbour lookup." << std::endl;
  
#pragma omp parallel for
  for (int n = 0; n < N; ++n) // loop over vertices
  {
    std::vector<DataPoint> neighbours;
    std::vector<double>    distances;
    
    // find nearest neighbors
    tree.search(vertices[n], K+1, &neighbours, &distances);
    
    for (int m = 0; m < K; ++m)
    {
      ind_arr[K*n+m]  = neighbours[m].index();
      dist_arr[K*n+m] = distances[m];
    }
    
    // print progress
#pragma omp atomic
    ++steps_completed;
    
    if (steps_completed % (10000) == 0)
    {
#pragma omp critical
      fprintf(stderr, " - point %d of %d\n", steps_completed, N);
    }
  }
  
  // Clean up memory
  vertices.clear();
  
  Rcpp::NumericMatrix _IND(N, K);
  Rcpp::NumericMatrix _DIST(N, K);
  
  for (int n = 0; n < N; ++n)
  {
    for (int m = 0; m < K; ++m)
    {
      _IND(n,m) = ind_arr[K*n + m];
      _DIST(n,m) = dist_arr[K*n + m];
    }
  }
  
  free(ind_arr); ind_arr = NULL;
  free(dist_arr); dist_arr = NULL;
  
  return Rcpp::List::create(Named("IND",  _IND),Named("DIST", _DIST));
}

// [[Rcpp::export]]
Rcpp::List openmp_knn_C( // R-compatible: creates input for openmp_knn_list
    Rcpp::NumericMatrix coordinates,
    int                 K,
    int                 distance_function = 0 // 0 ~ Euclidean, 1 ~ cosine
) {
  std::function<double (const DataPoint&, const DataPoint&)> dist;
  switch(distance_function)
  {
  case 0:
    dist = euclidean_distance;
    Rcpp::Rcout << "Using Euclidean distance." << std::endl;
    break;
  case 1:
    dist = cosine_distance;
    fprintf(stderr, "Using cosine distance is not supported yet, since it leads to unexpected behaviour.");
    return Rcpp::List("fail");
    break;
  default:
    Rcpp::Rcout << "Invalid distance metric. 0 ~ Euclidean, 1 ~ cosine." << std::endl;
  }
  
  int N = coordinates.nrow();
  int D = coordinates.ncol();
  
  if (K >= N)
  {
    Rcpp::Rcout << "K must be less than number of vertices." << std::endl;
    return Rcpp::List("fail");
  }
  
  ++K; // include origin
  
  // transfer data from matrix to calloc array
  double* X = (double*) calloc(N*D, sizeof(double));
  if (!X)
  {
    Rcpp::Rcout << "Could not transfer data from coordinates matrix to calloc array:\nmemory allocation failed." << std::endl;
    return Rcpp::List("fail");
  }
  
  for (int n = 0; n < N; ++n)
    for (int m = 0; m < D; ++m)
      X[n*D + m] = coordinates(n, m);
  
  int* ind_arr;
  double* dist_arr;
  
  Rcpp::List _output = openmp_knn_list(X, N, D, &ind_arr, &dist_arr, K, dist);
  
  free(X);
  return _output;
}

