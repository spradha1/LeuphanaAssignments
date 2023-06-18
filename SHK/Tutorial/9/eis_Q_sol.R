eis <- read.csv("eis.csv")

# CHANGE categorical variables' data types to FACTORS (aov requires categorical variables to be factors)
eis$ice_cream = factor(eis$ice_cream)
eis$puzzle_type = factor(eis$puzzle_type)
eis$adult = factor(eis$adult)


#1. Do people who prefer strawberry ice cream (flavor 1) score significantly better in the video test? (One-Way-Anova)

table(eis$ice_cream)  # balanced sample sizes assumptions (fails)

fligner.test(video ~ ice_cream, data=eis)  # homogeneity of variance assumption (pass)

# check normality (fails)
# with boxplot
boxplot(video ~ ice_cream, data=eis) # boxplot also shows strawberry ice cream people don't score the highest
# with Shapiro-Wilk test
# check for all groups 1, 2, 3: here you see another way to subset
shapiro.test(eis$video[eis$ice_cream==1])
# try log to make it normal
eis$log_video = log(eis$video)
shapiro.test(eis$log_video[eis$ice_cream==1]) # still fails

# 1-way ANOVA
model1 = aov(video ~ ice_cream, data=eis)   # accept null-hypothesis
summary(model1)

TukeyHSD(model1) # shows flavor 1 is not significantly different/higher from either group


#2. Can the score in the puzzle game explain the score in the video game? (Linear Regression)

model2 = lm(video ~ puzzle, data=eis)
summary(model2)
# p-value suggests it is significant, but low R-squared value indicates it is not a good model


#3. Do people who like chocolate ice cream and are puzzle pros score significantly higher on the puzzle games test? (Two-Way-ANOVA)

# Same assumptions as Q#1

boxplot(puzzle ~ ice_cream:puzzle_type, data=eis)
model3 = aov(puzzle ~ ice_cream:puzzle_type, data=eis) # only interaction term included in the model to get its isolated effect
summary(model3)

# check every p-value that involves chocolate(2)-pro group, and all of them should be significant
TukeyHSD(model3)
# some comparisons turn out not to be significant, although chocolate(2)-pro group is the highest
# if the boxplot does not clearly show the comparison, check the diff column in Tukey's Test, diff +ve implies, first level is higher 


#4. Is there a difference in frequency in adults and children concerning puzzle-type preference?

chisq.test(eis$puzzle_type, eis$adult)
# p-value turns out to be higher than 0.05, which means the groupings do not influence each other

