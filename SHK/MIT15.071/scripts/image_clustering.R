# image clustering

# our data are image pixels, so we use vectors
df = read.csv('daafile.csv', header=FALSE) # header False as no variable row at start
mat = as.matrix(df) # cannot change dataframe directly to vector
vec = as.vector(mat) # isolates each pixel

# distances between each vector or pixel
distances = dist(vec, method='euclidean')

# hierarchical clustering
clusters = hclust(distances, method='ward.D')

# plot dendrogram, a horizontal line that has more space to move vertically is a good cut
plot(clusters)
# draw rectangles around dendrogram splits
rect.hclust(clusters, k=3, border='red')

# view image pixel levels by clusters
cluster_labels = cutree(clusters, k=3)
tapply(vec, cluster_labels, mean)

# give cluster labels original matrix dimensions for visualization
dim(cluster_labels) = dim(mat)
image(cluster_labels, axes=FALSE) # hide axes

# check how well the clustering did by visualizing the  matrix
image(mat, axes=FALSE, col=grey(seq(0, 1, length=256)) )
