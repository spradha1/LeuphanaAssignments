# type_conversions

# chr to Date
dates = strptime(chrs, format="%m/%d/%y %H:%M")

# factors to numeric
nums = as.numeric(as.character(factors))

# factors to ordered factors
ordered_facs = factor(facs, ordered=TRUE, levels=c("val1", "val2", "..."))