#' @export
knn.annoy<-function(data,K=30,trees=150){
  res<-knn_annoy(data,K,trees)
  return(do.call("rbind",res[[1]]))
}