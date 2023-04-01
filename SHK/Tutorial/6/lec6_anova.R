getwd()
setwd("")

###########################################
restaurant<-read.table("boxplot.txt", header=TRUE)
attach(restaurant)#data from Khan Academy video "Constructing a box and whisker plot"
restaurant
median(miles)
mean(miles)
length(miles)
range(miles)

summary(miles)

boxplot(miles)
boxplot(miles,ylab="miles",boxwex=0.5,col="seagreen")
?boxplot

# let's see how this would look like with outliers
# this is how you add an observation to a dataframe
# create a new name to keep the old dataframe
new<-rbind(restaurant,40)

boxplot(new)
# add a very large outlier and our boxplot will get squashed
new2<-rbind(restaurant,1000)
boxplot(new2)


########################################################
#### ANOVA example
dat<-read.table("yields.txt",header=T)
str(dat)
dat
summary(dat)
attach(dat)
boxplot(yield~soil,ylab="yield [mg]")
#software always sorts alphabetical, if you want to sort from lowest to highest:change order by defining soil to be an ordered factor
soil<-factor(soil,levels=c("sand","clay","loam"))
#plot again
boxplot(yield~soil,ylab="yield [mg]")

#################################
ANOVA
#two possibilities to calculate ANOVA
#1.
model<-lm(yield~soil)
anova(model)
summary(model)

#2.
model<-aov(yield~soil)
summary(model)


hist(resid(model))
shapiro.test(resid(model))

# testing for homogeneity of variance, Fligner-Killeen Test
fligner.test(yield~soil)

# you can also use the plot(model) command, just make sure you adjust the name of your model, this model is called "model"
plot(model)

# you have to choose option 2 (see above) for a posthoc-Test = to test every factor level against each other
# Post Hoc test
TukeyHSD(model)

# two-way ANOVA, two factors instead of one--> soil type and fertilizer
model2<-aov(yield~soil+fertilizer)
summary(model2)

boxplot(yield~soil:fertilizer)

TukeyHSD(model2)
#factorial design allows to test for interaction
model3<-aov(yield~soil*fertilizer)
summary(model3)

#model simplifcition
#you start with the most complex model and test it against the more simple model (occam's razor!)
model3<-aov(yield~soil*fertilizer)
model2<-aov(yield~soil+fertilizer)
anova(model2,model3)
#if it is not significant, choose the simpler model (occam's razor!)
model1<-aov(yield~soil)
anova(model1,model2)
#create the Nullmodel 
model0<-aov(yield~1)
anova(model0,model1)


###next try on your own with splityields data
split<-read.table("splityield.txt",header=T)
attach(split)
summary(split)
