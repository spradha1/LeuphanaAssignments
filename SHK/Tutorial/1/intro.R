# If you downloaded and installed R in a location other than the United States, you might encounter some formatting issues later in this class due to language differences. To fix this, you will need to type in your R console:
Sys.setlocale("LC_ALL", "C")

# set language
Sys.setenv(LANG = "en")

# remove variables from the environment
rm(var)
rm(list=ls()) # all vars

# help
?plot
help.search('standard deviation')
