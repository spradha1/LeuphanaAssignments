# plotting instances

# histogram
hist(df$var,col="gold")

# boxplot
boxplot(df$var ~ df$cat_var, col="red")

# scatter plot
plot(df$var1, df$var2, col="blue")

# line plot
# type: 'p' for points, 'b' for both points and lines
# lty: line types: 1=solid, 2=dashed .....
plot(df$var1, df$var2, type="l", lty=1, col="blue")