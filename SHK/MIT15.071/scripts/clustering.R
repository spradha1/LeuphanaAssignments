# clustering
# NOTE: be sure to include only numeric data columns in df to avoid errors for the functions used below

# get distances between each row
distances = dist(df, method="euclidean")

# hierarchical clustering function
# ward.D (Ward's method) considers both distances between cluster centroids and the variance of clusters
hclusters = hclust(distances, method='ward.D')

# plot dendrogram
plot(clusters)

# get cluster labels for data for desired k = # of clusters
cluster_labels = cutree(hclusters, k=10)

# to find all cluster centroids
# split data by cluster labels
cluster_dfs = split(df, cluster_labels)
# lapply runs second function argument on each element of the first list-type argument
lapply(cluster_dfs, colMeans)
