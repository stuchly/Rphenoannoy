#if defined(__MINGW32__)
#undef Realloc
#undef Free
#endif
// define R’s REprintf as the ’local’ error
// print method for Annoy
#include"Rcpp.h"
#define __ERROR_PRINTER_OVERRIDE__REprintf
#include "annoylib.h"
#include "kissrandom.h"
#include <algorithm>
#include <vector>
#include <omp.h>
// [[Rcpp::plugins(openmp)]]

#define NUM_THREADS(N) ((N) >= 0 ? (N) : omp_get_num_procs() + (N) + 1)

using namespace Rcpp;


typedef float ANNOYTYPE;

typedef AnnoyIndex<int, ANNOYTYPE, Euclidean, Kiss64Random> MyAnnoyIndex;

// [[Rcpp::export]]
SEXP knn_annoy(Rcpp::NumericMatrix mat, const int K=100, const int trees=150) {
  const size_t nsamples=mat.nrow();
  const size_t ndims=mat.ncol();
  std::vector<std::vector<ANNOYTYPE>> dist(nsamples);
  std::vector<std::vector<int>> inds(nsamples);
  for(size_t i=0; i<nsamples; ++i) {
    dist[i].resize(ndims);
    inds[i].resize(ndims);
  }
  MyAnnoyIndex obj(ndims);
  // from <vector>
  std::vector<ANNOYTYPE> tmp(ndims);
  for(size_t i=0; i<nsamples; ++i) {
    Rcpp::NumericMatrix::Row cr=mat(i,_);// from <algorithm>
    std::copy(cr.begin(), cr.end(), tmp.begin());
    obj.add_item(i, tmp.data());
    }
  std::cout<<"Building annoy index\n";
  obj.build(trees);
  
  std::cout<<"index_built\n";
  
  int steps_completed = 0;
  
#pragma omp parallel for
  for (size_t i=0;i<nsamples;++i){
    std::vector<int> neighbor_index;
    std::vector<ANNOYTYPE> neighbor_dist;
    obj.get_nns_by_item(i, K + 1, -1, &neighbor_index,&neighbor_dist);
    inds[i]=neighbor_index;
    dist[i]=neighbor_dist;   
    
    // print progress
#pragma omp atomic
    ++steps_completed;
    
    if (steps_completed % (10000) == 0)
    {
#pragma omp critical
      fprintf(stderr, " - point %d of %zu\n", steps_completed, nsamples);
    }
    //dist.push_back(neighbor_dist);
  }
  return Rcpp::List::create(Rcpp::Named("inds",Rcpp::wrap(inds)),Rcpp::Named("dist",Rcpp::wrap(dist)));
      
}
