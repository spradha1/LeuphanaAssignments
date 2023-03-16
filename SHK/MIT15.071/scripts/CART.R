# CART model

library(rpart)
library(rpart.plot)

# method argument says to build classification tree
# minbucket: smallest number of observations allowed in a terminal node
mod1 = rpart(dep ~ indep + ... , data=train_stevens, method="class", minbucket=25)
# or use cp (complex parameter) instead of minbucket in the case of cross-validation
mod1 = rpart(dep ~ indep + ... , data=train_stevens, method="class", cp=0.18)

# plot tree
prp(mod1)
# or
plot(mod1)
text(mod1)

# type argument gives majority class predictions like applying threshold 0.5
# pred1 holds a list of probabilities for each class, so pick the right one for ROCR while evaluating
pred1 = predict(mod1, newdata=test_df, type="class")
