# data pertains to growth of caterpillars [mg] in relation to tannin concentration [micromole] in their diet
df <- read.table("regression.txt", header=T)

# inspection
names(df)  # column names
str(df)
dim(df)  # dimensions
summary(df)
View(df)

# data examination
plot(df$tannin, df$growth)

# try changing the look of your graph
plot(df$tannin, df$growth, xlab="tannin concentration [micromole]", ylab="caterpillar growth [mg]", main="My funny plot", pch=18, col="darkblue", cex=2)

# clear all plots
dev.off(dev.list()["RStudioGD"]) 

# building linear model
model <- lm(growth ~ tannin, data=df)

# draw the regression line associated with the model
abline(model, col="darkgreen")

# inspect model
summary(model)
summary(model)$coefficients
summary(model)$r.squared

# residuals are normally distributed
# get residuals with either model$residuals OR resid(model)
model$residuals
hist(model$residuals)
plot(model$residuals)
shapiro.test(model$residuals)

# error measures
# sum of squared errors
SSE = sum(model$residuals^2)
# root mean squared error
RMSE = sqrt(mean(model$residuals^2))
# total sum of squares
SST = sum((mean(df$growth) - df$growth)^2)
# R-squared
r2 = 1 - SSE/SST

# try it out with other data sets
