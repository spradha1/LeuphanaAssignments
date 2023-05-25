# two-way ANOVA

yields1 = read.table("yields.txt", header=T)
str(yields1)
summary(yields1)

# ANOVA assumptions
# check normality
boxplot(yield ~ soil, data=yields1)
boxplot(yield ~ fertilizer, data=yields1)
# check sample sizes across groups for all categories
table(yields1$soil)
table(yields1$fertilizer)
# check homoscedasticity
fligner.test(yield ~ soil, data=yields1)
fligner.test(yield ~ fertilizer, data=yields1)
fligner.test(yield ~ interaction(soil, fertilizer), data=yields1)

# don't forget: residuals are also supposed to be normally distributed (check with histogram & Shapiro-Wilk test)

# two independent variables instead of one --> soil type and fertilizer
model1 = aov(yield ~ soil + fertilizer, data=yields1)
# or use formula 'yield ~ .' to include every individual variable as explanatory variables in the model
summary(model1)
TukeyHSD(model1)

# interaction variables
model2 = aov(yield ~ soil:fertilizer, data=yields1)
summary(model2)
boxplot(yield ~ soil:fertilizer, data=yields1)
TukeyHSD(model2)

# additive & interaction altogether: a*b -> a + b + a:b
model3 = aov(yield ~ soil*fertilizer, data=yields1)
# or use formula 'yield ~ (soil + fertilizer)^2'
# or use formula 'yield ~ .^2' to include every two-way interaction in addition to individual variables as explanatory variables in the model
summary(model3)
TukeyHSD(model3)


# model reduction
# you start with the most complex model and reduce it one variable at a time to the simple model (Occam's razor!)
rmodel = aov(yield ~ .^2, data=yields1)
rmodel = update(rmodel, . ~ . - soil:fertilizer, data=yields1)
rmodel = update(rmodel, . ~ . - fertilizer, data=yields1)
summary(rmodel)


######### three-way interaction ##########

yields2 <- read.table("splityield.txt", header=T)
summary(yields2)
str(yields2)

# examine dataset
table(yields2$fertilizer, yields2$density, yields2$irrigation)
boxplot(yield ~ irrigation, data=yields2)

# TODO: check ANOVA assumptions

int_model <- aov(yield ~ irrigation*density*fertilizer, data=yields2)
int_model <- aov(yield ~ irrigation*density*fertilizer + Error(block), data=yields2)
int_model <- aov(yield ~ irrigation*density*fertilizer + Error(block/irrigation), data=yields2)
int_model <- aov(yield ~ irrigation*density*fertilizer + Error(block/irrigation/density), data=yields2)
summary(int_model)


########## exercise ################
data(ToothGrowth)
summary(ToothGrowth)
str(ToothGrowth)

interaction.plot(ToothGrowth$supp, ToothGrowth$dose, ToothGrowth$len)
?interaction.plot

# TODO: perform two-way ANOVA & model reduction with 'len' as the response variable, but wait there is something wrong
