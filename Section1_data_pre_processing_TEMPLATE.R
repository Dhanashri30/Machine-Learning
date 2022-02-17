# --------------------------------------------------------
# ST4061 / ST6041
# 2021-2022
# Eric Wolsztynski
# ...
# Exercises Section 1: Daat pre-processing
# --------------------------------------------------------

library(glmnet)
library(survival)

###############################################################
### Exercise 1: effect of scaling
###############################################################

dat = iris[1:100,]
dat$Species = droplevels(dat$Species)
x = dat[,1:4]
y = dat$Species

# (1)
pca.unscaled = prcomp(x)
pca.scaled = prcomp(x,scale=TRUE)
par(mfrow=c(2,2))
plot(pca.unscaled) # scree plot 
biplot(pca.unscaled) # biplot 
# ... same for scaled, and analyse

# (2)
logreg.unscaled = glm(Species~., data=dat) # make this work
# repeat for scaled analysis and discuss

# (3)
glmnet(...) # make it work
# run LASSO like you know how to and discuss...

###############################################################
### Exercise 2: data imputation
###############################################################

dat = lung
summary(dat)
names(dat)
boxplot(dat$meal.cal~dat$sex)
# can you think of other ways of analysing this?

# run a 1st Cox model:
cox.mod = coxph( Surv(time,status) ~ ., data=dat)
summary(cox.mod)

# now perform very crude data imputation (to "fill" missing values):
sum(is.na(dat$sex)) # no missing values for 'sex'
ina = which(is.na(dat$meal.cal)) # locations of missing values for meal.cal
dat.imp1 = dat
dat.imp1$meal.cal[ina] = median(dat$meal.cal,na.rm=TRUE) # explain this
cox.mod.imp1 = coxph( Surv(time,status) ~ ., data=dat.imp1)
summary(cox.mod.imp1)

# continue by experimenting with other forms of imputation, 
# including one that is gender specific 

###############################################################
### Exercise 3: data imputation
###############################################################

# Replicate steps from Question 2

###############################################################
### Exercise 4: resampling
###############################################################

# You don't need help for this one!

###############################################################
### Exercise 5: resampling (CV vs bootstrapping)
###############################################################

# Implement this simple analysis and discuss - think about 
# (sub)sample sizes!

