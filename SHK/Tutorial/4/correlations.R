# Correlations

# lcrating some data
v1 <- c(94,73,59,80,93,85,66,79,77,91)
v2 <- c(17,13,12,15,16,14,16,16,18,19)

# Pearson's correlation

# assumptions: both variables are normally distributed
# visualize data with a histogram to check the normality or use the Shapiro-Wilk test
hist(v1)
hist(v2)
shapiro.test(v1)
shapiro.test(v2)

# Pearson's correlation coefficient
cor(v1, v2)
# Pearson's correlation test statistic
cor.test(v1, v2)
# plot
plot(v1, v2)


# load clean survey data
survey <- read.csv("survey23_clean.csv", header=T, sep=",")
str(survey)
View(survey)

# check normality for Pearson's correlation
hist(survey$siblings_num)
hist(survey$pets)

# before trying log for normalization, check if there are zeros
summary(survey$siblings_num)
table(survey$siblings_num)
summary(survey$pets)
table(survey$pets)
# what is the logarithm of zero?
log(0)

# add a value to remove zeros before taking the log
hist(log(survey$siblings_num) + 1)
hist(log(survey$pets) + 1)

# data is not normally distributed, so Spearman & Kendall rank correlation instead
?cor.test
cor.test(survey$siblings_num, survey$pets, method="spearman")
cor.test(survey$siblings_num, survey$pets, method="kendall")

# visualize
plot(survey$siblings_num, survey$pets)


# next data set
auto <- read.table("Automobile_R.txt",header=T)
# examine
str(auto)
summary(auto)
auto <- na.omit(auto)
# this shows the correlation coefficients for the entire data set (only possible if your data set contains only numeric data)
cor(auto)

# let's correlate automobile width with length
# try it yourself first

# checking normality
hist(auto$width)
hist(auto$length)
shapiro.test(auto$width)
shapiro.test(auto$length)

# if not normal, check for zeros before taking log
table(auto$width)
table(auto$length)

# check normality for log data accordingly (add 1 if zeros present)
hist(log(auto$width))
hist(log(auto$length))
shapiro.test(log(auto$width))
shapiro.test(log(auto$length))

# visualize
plot(auto$width, auto$length)

# if the data is not normally distributed, we use Spearman & Kendall rank correlation test
cor.test(auto$length, auto$width, method="spearman")
cor.test(auto$length, auto$width, method="kendall")

# assignment: find variables in the auto data set that are normally distributed based on the Shapiro test and find the correlation
