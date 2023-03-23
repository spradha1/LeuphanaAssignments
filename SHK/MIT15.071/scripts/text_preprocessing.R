# text data preprocessing
library(tm)
library(SnowballC)

# convert df column with text into corpus
corpus = Corpus(VectorSource(df$text))

# to lower case
corpus = tm_map(corpus, tolower)

# remove punctuation
corpus = tm_map(corpus, removePunctuation)

# remove words
corpus = tm_map(corpus, removeWords, c(stopwords("english"), 'word1', ...))

# stemming
corpus = tm_map(corpus, stemDocument)
