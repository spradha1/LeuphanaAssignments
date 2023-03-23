# text inspection

# corpus to DocumentTermMatrix (frequency table)
freqs = DocumentTermMatrix(corpus)
inspect(freqs[1:5,1:10])

# get terms with frequency above a certain threshold (lowfreq)
findFreqTerms(freqs, lowfreq=10)

# remove terms appearing in less docs
sparse = removeSparseTerms(freqs, 0.98) # keep terms appearing in more than 2% of the docs

# DocumentTermMatrix to dataframe
# has all the terms as independent variables
sparse_df = as.data.frame(as.matrix(sparse))

# make variable names legal for R
colnames(sparse_df) = make.names(colnames(sparse_df))
