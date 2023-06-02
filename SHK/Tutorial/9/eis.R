# The data was collected on 200 random citizens and are scores on various tests, including a video game and a puzzle. 
# The data also includes the person’s favorite flavor of ice cream – vanilla (1), chocolate (2) or strawberry (3).
eis <- read.csv("eis.csv", header=T)
str(eis)
summary(eis)

# some important commands

# filtering dataset
eis_mit_adults = subset(eis, adult=='1')
eis_video_uber_50 = subset(eis, video>50)

# getting group stats: two ways
# aggregate gives back a dataframe, tapply gives back a vector
aggregate(video ~ adult, data=eis, FUN=sum)
tapply(eis$video, eis$adult, FUN=median)


########### plotting ##############

# scatter plot
plot(eis$video, eis$puzzle,
  col='chocolate', type='p',
  # for points
  cex=2, pch=15,
  # for line
  lwd=3, lty=1,
  xlab='xlabel', ylab='ylabel', main='title'
)

# boxplot
boxplot(
  video ~ adult, data=eis,
  col=c('green', 'yellow'), boxwex=0.3,
  xlab='xlabel', ylab='ylabel', main='title'
)

# histogram
hist(
  eis$video,
  col='darkred', border='cyan',
  breaks=c(20, 30, 40, 50, 60, 70, 80),
  xlab='xlabel', ylab='ylabel', main='title'
)

# barplot
barplot(
  table(eis$adult, eis$ice_cream),
  col=c('black', 'skyblue'),
  beside=T,
  xlab='xlabel', ylab='ylabel', main='title'
)

# legend
# can only be used to add onto a plot
# using position keyword
legend("topright", legend=c(0, 1), fill=c('black', 'skyblue'))
# using position coordinates
legend(8, 48, legend=c(0, 1), fill=c('black', 'skyblue'), title='adult')


# sets graphical parameters
?par
par(col='green')
# clear plot window
graphics.off()


# saving plots

# function should match file extension: use '.png' filename when using png() command
# other options: png, pdf, tiff, bmp, ps, jpeg
png('filename.png')

# plot commands .....

# shuts down plot window & save
dev.off()
