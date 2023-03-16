# Imputation of NAs

library(mice)

imputed = complete(mice(df))
df$col1 = imputed$col1