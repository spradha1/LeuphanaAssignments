# Random Forest model

library(randomForest)

# nodesize: smallest number of observations allowed in a terminal node
# ntree: # of trees to generate for the random forest algorithm
mod1 = randomForest(dep ~ ind + ... , data=train_df, nodesize=25, ntree=100)

pred1 = predict(mod1, newdata=test_df)
