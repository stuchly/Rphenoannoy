#include <Rcpp.h>
#include <omp.h>
#include <vector>
// [[Rcpp::plugins(openmp)]]

#define NUM_THREADS(N) ((N) >= 0 ? (N) : omp_get_num_procs() + (N) + 1)

using namespace Rcpp;

// Compute jaccard coefficient between nearest-neighbor sets
//
// Weights of both i->j and j->i are recorded if they have intersection. In this case
// w(i->j) should be equal to w(j->i). In some case i->j has weights while j<-i has no
// intersections, only w(i->j) is recorded. This is determinded in code `if(u>0)`. 
// In this way, the undirected graph is symmetrized by halfing the weight 
// in code `weights(r, 2) = u/(2.0*ncol - u)/2`.
//
// Author: Chen Hao, Date: 25/09/2015


int intersection(int D, int *array1, int *array2)
{       
    int count = 0;
    // std::sort(array1, array1+D, std::greater<int>()); 
    // std::sort(array2, array2+D, std::greater<int>()); 
    // std::cout<<D;
    // for(int i = 0; i < D; i++)
    //     std::cout<<array2[i]<<std::endl;
    for(int i = 0; i < D; i++)
    {
        // std::cout<<i<<std::endl;
        for(int j = 0; j < D; j++)
        {
            if(array1[i]==array2[j])
            {
                count = count + 1;

            }
        }
    }
    return(count);
    
}

// [[Rcpp::export]]
NumericMatrix jaccard_coeff_parallel(NumericMatrix idx) {
    int nrow = idx.nrow(), ncol = idx.ncol();
    int x[ncol];
    int y[ncol];
    NumericMatrix W(nrow*ncol, 3);
    int r = 0;
    int* X = (int*) calloc(nrow*ncol, sizeof(int));
    
    
    
#pragma omp parallel for 
    for (int n = 0; n < nrow; ++n)
        for (int m = 0; m < ncol; ++m)
            X[n*ncol + m] = idx(n, m);


    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            int k = X[i*ncol+j]-1;
            
            
            for (int ii=0; ii<ncol; ii++) {
                x[ii]=X[i*ncol+ii];
                y[ii]=X[k*ncol+ii];
                // std::cout<<y[ii]<<",";
                
            }
            // std::cout<<std::endl;
            int u = intersection(ncol, x, y);  // count intersection number
            if (u>0){
                W(r,0)=i+1;
                W(r,1)=k+1;
                W(r,2)=u/(2.0*ncol - u)/2;
                r++;
            }
            
           
        }
    }
    
    
    return W;
}

// [[Rcpp::export]]
NumericMatrix jaccard_coeff(NumericMatrix idx) {
    int nrow = idx.nrow(), ncol = idx.ncol();
    NumericMatrix weights(nrow*ncol, 3);
    int r = 0;
    
    
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            int k = idx(i,j)-1;
            NumericVector nodei = idx(i,_);
            NumericVector nodej = idx(k,_);
            int u = intersect(nodei, nodej).size();  // count intersection number
            if(u>0){ 
                weights(r, 0) = i+1;
                weights(r, 1) = k+1;
                weights(r, 2) = u/(2.0*ncol - u)/2;  // symmetrize the graph
                r++;
            }
        }
    }
    
    return weights;
}

// [[Rcpp::export]]
NumericMatrix jaccard_coeff_true_parallel(NumericMatrix idx) {
    int nrow = idx.nrow(), ncol = idx.ncol();
    
    NumericMatrix W(nrow*ncol, 3);
    int r = 0;
    int* X = (int*) calloc(nrow*ncol, sizeof(int));
    
    
    

    for (int n = 0; n < nrow; ++n)
        for (int m = 0; m < ncol; ++m)
            X[n*ncol + m] = idx(n, m);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nrow; i++) {

        for (int j = 0; j < ncol; j++) {
            int k = X[i*ncol+j]-1;
            
            int x[ncol];
            int y[ncol];
            for (int ii=0; ii<ncol; ii++) {
                x[ii]=X[i*ncol+ii];
                y[ii]=X[k*ncol+ii];
                // std::cout<<y[ii]<<",";
                
            }
            // std::cout<<std::endl;
            int u = intersection(ncol, x, y);  // count intersection number
            if (u>0){
                W(i*ncol+j,0)=i+1;
                W(i*ncol+j,1)=k+1;
                W(i*ncol+j,2)=u/(2.0*ncol - u)/2;
               
            }
            
            
        }
    }
    
    
    return W;
}
