setwd("~/ST4061/ML2/Exercise Bank/CA1")

rm(list=ls())
library(ISLR)
library(randomForest)
library(gbm)
library(pROC)


set.seed(4061) 
n = nrow(Caravan) 
dat = Caravan[sample(1:n, n, replace=FALSE), ] 
dat$Purchase = as.factor(as.numeric(dat$Purchase=="Yes")) 
i.train = sample(1:n, round(.7*n), replace=FALSE) 
x.train = dat[i.train, -ncol(dat)] 
y.train = dat$Purchase[i.train] 
x.test = dat[-i.train, -ncol(dat)] 
y.test = dat$Purchase[-i.train] 


#1
set.seed(4061)
glm1 = glm(y.train~., data = x.train, family = 'binomial')

glm1.p = predict(glm1, newdata = x.test, type='response')

tb = table(y.test, glm1.p>0.5) #got this wrong

acc1 = sum(diag(tb))/sum(tb)
#miss classification rate
1 - acc1


#2
set.seed(4061)

rf2 = randomForest(y.train~., data=x.train, ntree=100)

rf2.p = predict(rf2, newdata = x.test, type='response')

tb2 = table(y.test, rf2.p)
acc2 = sum(diag(tb2))/sum(tb2)

#miss classification rate
1 - acc2


#3
set.seed(4061)

gb.y.train = (y.train==1)

gbm3 = gbm(gb.y.train~., data = x.train, n.trees = 100)

gbm3.p = predict(gbm3, newdata = x.test, n.trees=100)

tb3 = table(y.test, gbm3.p>0) #cause log probs?

acc3 = sum(diag(tb3))/sum(tb3)

#miss classification rate
1 - acc3


#4
#get probabilities
rf2.prob = predict(rf2, newdata = x.test, type='prob')


glm1.r = roc(y.test, glm1.p)
rf2.r = roc(y.test, rf2.prob[,2])
gbm3.r = roc(y.test, gbm3.p)

par(mfrow=c(1,3))
plot(glm1.r, main='GLM')
plot(rf2.r, main='Random Forest')
plot(gbm3.r, main='GBM')
par(mfrow=c(1,1))

#auc
glm1.r$auc
rf2.r$auc
gbm3.r$auc


#5
#% of 1 i.e not purchase
sum(as.numeric(y.test)-1)/length(y.test)

1-acc1
1-acc2
1-acc3


#given the accuracy of all 3 models is worse or equivalent to just guessing
# purchase for everything I am not very satisfied with these models

#of the 3 GBM gave the best performance but it is the same as guessing Purchase for all

#There ROC curves are poor and the auc is not particularly impressive


#6

#Random forest is worst of all 3 and it is possibly over fitting

#gbm was best of all 3 however this is not good as explained earlier