# linear regression model

mod1 = lm(dep ~ ind + ..., data=train_df)
pred1 = predict(mod1, newdata=test_df)

# residuals give the differences between the actual values of the training set and the predicted values of the model
residuals = mod1$residuals

# visualize confusion matrix for testing set
table(test_df$dep, pred1 >= 0.5)