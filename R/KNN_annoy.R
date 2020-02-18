#' K Nearest Neighbour Search
#'
#' Uses a annoylib to find K nearest neighbors in dataset
#' see https://github.com/spotify/annoy/
#'
#' @param data matrix; input data matrix
#' @param K integer; number of nearest neighbours
#' @param trees integer; number of trees in annoylib; more trees - more precise result
#' @param return_dist bool; if TRUE a list of indices and distances is returned
#'
#' @return a n-by-k matrix of neighbors indices or list(IND,DIST) see parameter return_dist 
#'
#' @examples
#' iris_unique <- unique(iris) # Remove duplicates
#' data <- as.matrix(iris_unique[,1:4])
#' neighbors <- knn.annoy(data, k=10)
#'
#' @export
knn.annoy<-function(data,K=30,trees=150, return_dist=FALSE){
  res<-knn_annoy(data,K,trees)
  if (return_dist) return(list(IND=do.call("rbind",res[[1]])+1,DIST=do.call("rbind",res[[2]])))
  return(do.call("rbind",res[[1]])+1)
}