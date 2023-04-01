# Imputation of NAs with mice
library(mice)

imputed = complete(mice(df))
df$col1 = imputed$col1

# imputing NAs with a value
df[is.na(df)] = 0