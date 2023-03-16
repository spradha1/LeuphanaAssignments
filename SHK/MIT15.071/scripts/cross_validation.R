# cross validation

library(caret)
library(e1071)

# number argument = # of folds
# method = cv for cross validation
folds = trainControl(method="cv", number=10)

# complex parameters to test
cps = expand.grid(.cp=seq(0.01, 0.5, 0.01))

# perform cross validation
# method: model type (rpart for CART model)
# trControl: trainControl output
# tuneGrid: expand.grid output for our sequence of parameters to test
mod1 = train(dep ~ ind + ... , method="rpart", data=train_df, trControl=folds, tuneGrid=cps)

# use cp from output in new model

# only for CARTs: print the tree with best cp
prp(mod1$finalModel)
