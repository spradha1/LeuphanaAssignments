# Tutorial day 3

# simple calculations
vals <- c(9, 2, 6,	9, 4,	7, 4,	2, 5,	6, 8,	7, 2)
summary(vals)
mean(vals)
median(vals)
sd(vals)
var(vals)

# visualize
boxplot(vals)
hist(vals)
# let's see how we can make this more "normal"
hist(log(vals))


# survey data
survey <- read.csv("survey23.csv", header=T, sep=",")
attach(survey)
str(survey)


# Chi-Squared Test
# Link between OS & breakfast?
table(breakfast, OS)

# this is how you filter columns
subset(OS, OS=="Apple")
# sub-select for several factor levels
subset(OS, OS=="Neither" | OS=="Both")

chi = chisq.test(table(breakfast, OS))
chi
chi$expected # shows expected values

# critical value for confidence & degrees of freedom
# you can also check the chi-square table
qchisq(0.95, 3)

# Let's exclude all those NAs
survey_new <- na.omit(survey)
summary(survey_new)


# here is what you do if you want to save the new table as a csv file
write.csv(survey_new, "survey23_new.csv")
survey_new <- read.csv("survey23_new.csv")

# NOTE: if your dataset introduces new objects with same name, it will override the old variables
attach(survey_new)
