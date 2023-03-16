# ROC curve

library(ROCR)

ROCR_pred = prediction(train_pred, df_train$dep)
ROCR_perf = performance(ROCR_pred, "tpr", "fpr") # y-axis, x-axis

# auc
auc = as.numeric(performance(ROCR_pred, "auc")@y.values) # y-axis

# plot threshold points, and position of threshold text
plot(ROCR_perf, colorize=TRUE, print.cutoffs.at=seq(0, 1, 0.1), text.adj=c(-0.5, 0.5))