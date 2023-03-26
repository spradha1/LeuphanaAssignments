# data preparation

# rename columns
colnames(df) = c("name1", ....)

# removing column
df$col1 = NULL

# removing duplicates
df = unique(df)