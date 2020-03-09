#' K Nearest Neighbour Search
#'
#' Uses a paralell version of vantage point tree to find K nearest neighbors in dataset
#' see https://lvdmaaten.github.io/tsne/
#'
#' @param data matrix; input data matrix
#' @param K integer; number of nearest neighbours
#'
#' @details If used as precomputed matrix for Rphenoannoy, discard first column - 1st nearest 
#' neigbor is the point itself
#' 
#' @return a n-by-k matrix of neighbors indices or list(IND,DIST)
#'
#' @examples
#' iris_unique <- unique(iris) # Remove duplicates
#' data <- as.matrix(iris_unique[,1:4])
#' neighbors <- knn.balltree(data, k=10)
#'
#' @export
knn.balltree<-function(X,K=30){
  res<-openmp_knn_C(coordinates = X,K = K)
  return(list(IND=res$IND+1,DIST=res$DIST))
}