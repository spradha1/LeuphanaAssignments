# boxplot

# data
restaurant <- read.table("boxplot.txt", header =T)

# inspect
str(restaurant)
mean(restaurant$miles)
sort(restaurant$miles)
median(restaurant$miles)
length(restaurant$miles)
range(restaurant$miles)
summary(restaurant$miles)

# plot
boxplot(restaurant$miles)
# now give it a label on the y-axis, a title, and some color
# search for help with '?boxplot'

# let's see how this would look like with outliers
# adding an observation to a dataframe
# create a new variable to keep the old dataframe
restaurant1 <- rbind(restaurant, 40)
boxplot(restaurant1$miles)
# add a very large outlier and our boxplot will get squashed


############## ANOVA ###################

# another data set
df <- read.table("yields.txt", header=T)
str(df)
summary(df)

# check normality
boxplot(yield ~ soil, data=df)

# R sorts alphabetical, if you want a custom order, change order by defining soil to be an ordered factor
df$soil <- factor(df$soil, levels=c("sand", "clay", "loam"))

# checking sample size across our categorical variable
table(df$soil)

# Fligner-Killeen Test
# testing for homogeneity of variances (homoscedasticity) 
# null hypotheses: all samples have equal variances
fligner.test(yield ~ soil, data=df)


# two ways to calculate ANOVA

# 1: linear model
model1 <- lm(yield ~ soil, data=df)
anova(model1)
summary(model1)
plot(model1)

# residuals are also supposed to be normally distributed
hist(model1$residuals)
shapiro.test(model1$residuals)

# 2: aov
model2 <- aov(yield ~ soil, data=df)
summary(model2)
plot(model2)

# Post-hoc-Test = to test every factor level against each other
# Tukey's test
TukeyHSD(model2)


# now try it with fertilizer
table(df$fertilizer)
model3 <-aov(yield ~ fertilizer, data=df)
summary(model3)

# t-test also an option as we have only 2 levels
?t.test
t.test(subset(df, fertilizer=="giant")$yield, subset(df, fertilizer=="lushly")$yield)

# assignment: use our survey data
# find a categorical variable with 3 or more levels and a continuous variable
# inspect your data for normality, homoscedasticity
# perform one-way ANOVA
