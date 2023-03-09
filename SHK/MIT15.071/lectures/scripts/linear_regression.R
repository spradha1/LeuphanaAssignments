# linear regression model

mod1 = lm(dep ~ ind + ..., data=train_df)
pred1 = predict(mod1, newdata=test_df)

# visualize confusion matrix for testing set
table(test_df$dep, pred1 >= 0.5)