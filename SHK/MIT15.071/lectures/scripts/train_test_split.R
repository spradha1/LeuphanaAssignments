# split data into training & testing sets

library(caTools)

# gives back a boolean vector with TRUE for observations to be trained
# dataframe can also be picked to be one of the columns for proper sampling of target
split = sample.split(df, SplitRatio = 0.75)

train_df = subset(df, split == TRUE)
test_df = subset(df, split == FALSE)