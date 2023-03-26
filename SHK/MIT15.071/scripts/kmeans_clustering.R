# K-Means clustering
library(flexclust) # for kcca

# scree plot to guess appropriate number of clusters, with elbow method
num_clusters = 1:10
tot_withins = sapply(1:10, function(x) kmeans(vec, centers=x, iter.max=1000)$tot.withinss)
plot(num_clusters, tot_withins, type='b', col='darkviolet', pch=18) # pch: shape of points
k = 5

# K-Means with 5 clusters & 1000 max iterations
kmc = kmeans(vec, centers = k, iter.max = 1000)
cluster_labels = kmc$cluster
dim(cluster_labels) = dim(mat)
centers = kmc$centers

# comparing images of data and cluster labels
image(mat, axes=FALSE, col=gray(seq(0, 1, length=256)))
image(cluster_labels, axes=FALSE, col=rainbow(k))

# testing with K-Centroids Cluster Analysis from clustering training
kmc.kcca = as.kcca(kmc, vec)
predict_clusters = predict(kmc.kcca, newdata=test_vec)
dim(predict_clusters) = dim(test_mat)

# visualization comparison
image(test_mat, axes=FALSE, col=rainbow(k))
image(predict_clusters, axes=FALSE, col=rainbow(k))
