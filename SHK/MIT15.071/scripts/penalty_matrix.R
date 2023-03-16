# penalty matrix

# 5x5 penalty matrix penalizing elements to the left of the diagonal more
penalty_matrix = matrix(c(0,1,2,3,4,2,0,1,2,3,4,2,0,1,2,6,4,2,0,1,8,6,4,2,0), byrow=TRUE, nrow=5)

# confusion matrix of outcomes & predictions
conf_mat = as.matrix(table(reals, preds))

# total mean error
sum(conf_mat*penalty_matrix) / nrow(test_df)
