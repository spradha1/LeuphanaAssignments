# logistic regression with glm

mod1 = glm(dep ~ ind + ..., data=train_df, family=binomial)
pred1 = predict(mod1, type='response', newdata=test_df)

# visualize performance on testing set with confusion matrix
table(test_df$dep, pred1 >= 0.5)