# load data
baseball = read.csv('baseball.csv')
str(baseball)

# All teams do not make it to the playoffs, thus the NAs, so remove them first
playoffs_data = subset(baseball, RankPlayoffs != 'NA') 

# check the correlation between between the ranks in the playoffs (performance in the playoffs)
# and the ranks in the regular season or the number of wins the regular season

# normality assumptions
hist(log(playoffs_data$W))
shapiro.test(log(playoffs_data$W))
hist(log(playoffs_data$RankSeason))
shapiro.test(log(playoffs_data$RankSeason))
hist(log(playoffs_data$RankPlayoffs))
shapiro.test(log(playoffs_data$RankPlayoffs))

# correlation
cor(playoffs_data$RankPlayoffs, playoffs_data$RankSeason)
cor(playoffs_data$RankPlayoffs, log(playoffs_data$W))


# compare linear models' R-squared & SSE
model1 = lm(W ~ OBP, data=baseball)
model2 = lm(W ~ BA, data=baseball)
SSE1 = sum(model1$residuals^2)
SSE2 = sum(model2$residuals^2)


# ANOVA

# check normality
boxplot(W ~ League, data=baseball)

# check sample sizes
table(baseball$League)

# check homogeneity of variances
fligner.test(W ~ League, data=baseball)

# models
model3 = aov(W ~ League, data=baseball)
summary(model3)
