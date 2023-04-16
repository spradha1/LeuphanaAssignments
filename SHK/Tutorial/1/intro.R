# display current R session details
sessionInfo()

# check
sd(1:10)
x + 2

# If you downloaded and installed R in a location other than the United States, you might encounter some formatting issues later in this class due to language differences. To fix this, you will need to type in your R console:
Sys.setlocale("LC_ALL", "C")
# set language
Sys.setenv(LANG = "en")


# valid variable names: starts with letters, only include numbers, letters & underscores
# valid: jon, jones, jon_jones, jon4jones
# invalid: jon jones, 4jon, _jon, jon! 
# remove variables from the environment

# data type check
class(jon)

# vectors
v1 = c(3, 2, 8)
v2 = 1:3
v3 = seq(1, 10, 1)

# vector operations
v1 + v2
v1 - v2
v1 * v2
v1 / v2
v3 > 5

# vector access
v3[1]
v3[c(3, 6)]
v3[3:6]
v3[c(TRUE, FALSE)]

# which index
which(v3 > 5)

# remove variables from your environment
rm(var)
rm(list=ls()) # all vars

# install package
install.packages("caTools")
# load library
library(caTools)

# help
help.search('standard deviation')
?sd

# graph
plot(v1, v2)

# while creating a new R file, save it with the '.R' extension as in <something.R>
# do not save your R environment on closing RStudio
