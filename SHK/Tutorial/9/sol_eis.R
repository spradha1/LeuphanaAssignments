# solutions
plot(
  eis$puzzle, eis$video,
  pch=17, col="purple",
  ylab="video score", xlab ="puzzle score", main="Video vs. puzzle score"
)

boxplot(
  video ~ ice_cream, data=eis,
  col=c("white","chocolate","pink"), boxwex=0.5,
  xlab="ice cream flavor", ylab="video score"
)

barplot(
  table(eis$ice_cream, eis$puzzle_type),
  beside=T,
  col=c('white', 'brown', 'pink'),
  main="Puzzle type meets ice cream flavour", ylab ="number of people", xlab="puzzle type"
)
legend("topright", legend=c('1:vanila', '2:choco', '3:strawberry'), fill=c('white', 'brown', 'pink'))
