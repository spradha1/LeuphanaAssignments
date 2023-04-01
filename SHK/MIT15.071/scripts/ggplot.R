# ggplot2 basics: data, aesthetic mapping & geometric objects

library(ggplot2)

# ggplot object with data and aesthetic mapping of axes
plot1 = ggplot(df, aes(x=col1, y=col2))

# plot as combo of functions
# geom_points for scatterplot
plot1 + geom_point(color="darkorchid", size=2, shape=8)

# save plot
plot1 = plot1 + xlab("x-label") + ylab("y-label") + ggtitle("Title text")
pdf('location.pdf')
print(plot1)
dev.off()


########## advanced plots ############

# color argument creates splits or gradient over specified data column
plot2 = ggplot(df, aes(x=col1, y=col2, color=col3)) + geom_point()
# add regression line to plot
plot2 = plot2 + stat_smooth(method="lm", color="tan4") + scale_color_brewer(palette="dark2")
# confidence interval shaded around regression line (default = 0.95)
plot2 = plot2 + stat_smooth(method="lm", level=0.99)
# to eliminate confidence interval shading
plot2 = plot2 + stat_smooth(method="lm", se=FALSE)

# multiple lines grouped by categorical variable
plot2 = ggplot(df, aes(x=col1, y=col2, group=col3_cat)) + geom_line(aes(color=col3_cat, linewidth=1.5))

# heatmap
ggplot(df, aes(x=col1, y=col2)) + geom_tile(data=df, aes(fill=col3_num)) + scale_fill_gradient(name="scale_legend_title", low="Green", high="Orange") + theme(axis.title.y = element_blank())

####### bars #########
# stat = "identity" for calculating sum of 'y' grouped by 'x'
ggplot(intl, aes(x=Region, y=PercentOfIntl)) + geom_bar(stat = "identity", fill="navy") + geom_text(aes(label=PercentOfIntl), vjust=-0.7) + ylab("Percent of International Students") + theme(axis.title.x = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1))

###### histogram ######
# right-closed & left-open by default
# facet_grid: forms a matrix of plots divvied up by levels on either side of the formula argument
ggplot(data = df, aes(x=col1)) + geom_histogram(binwidth = 5, fill="Blue", color="Orangered") + facet_grid(col1 ~ col2)
# histogram with bars divided by categorical attribute, pick colors with user-defined palette
# overlap instead of stacking with position = "identity"
ggplot(data = df, aes(x=age, fill=col1)) + geom_histogram(binwidth = 5, position = "identity", alpha=0.5) + scale_fill_manual(values=c("Cyan", "Darkred"))

### maps ###
# sort map 'group' attribute with 'order'
mapdf = mapdf[order(mapdf$group, mapdf$order),]
# plot map data
ggplot(mapdf, aes(x=long, y=lat, group=group)) + geom_polygon(aes(fill="column/color"), color="border_color") + coord_map("mercator")
