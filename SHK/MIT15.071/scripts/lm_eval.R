# linear regression model evaluation

# sum of squared errors wrt to the model's predictions
SSE = sum((pred - test_df$dep)^2)
# sum of squared erros wrt to the baseline model (for instance: mean)
SST = sum((mean(train_df$dep) - test_df$dep)^2)

# R-squared
r2 = 1 - SSE/SST

# Root mean squared error
RMSE = sqrt(mean((pred - test_df$dep)^2))