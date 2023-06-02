guns = read.csv("guns.csv")

boxplot(murder ~ state, data=guns)

which.max(guns$murder)  # the position of the maximum murder rate in the data frame
guns[199,]  # get 199th row, all columns

# correlation between income & robbery rate
cor(guns$income, guns$robbery)

# effects of state on violence rate
model1 = aov(violent ~ state, data=guns)
summary(model1)

# removing the DC state rows
guns_ohne_dc = subset(guns, state != 'District of Columbia')
