setwd("~/ST4061/ML2/Exercise Bank/CA1")
rm(list=ls())

library(mlbench) 
data(Sonar)  
N = nrow(Sonar) 
P = ncol(Sonar)-1 
M = 150   
set.seed(1) 
mdata = Sonar[sample(1:N),] 
itrain = sample(1:N,M) 
x = mdata[,-ncol(mdata)] 
y = mdata$Class 
xm = as.matrix(x) 


#1
N-M
#58 observations in test set


#2
library(glmnet)
set.seed(1)
#i missed family = binomial here originally 
lam = cv.glmnet(xm[itrain,], y[itrain],  alpha = 1, family='binomial')

#optimal reg param
lam$lambda.min


#3
glm = glmnet(xm[itrain,], y[itrain], alpha = 1, lambda =lam$lambda.min, family='binomial' )

coef(glm)

# (Intercept)   4.2103625
# V1          -21.7797320
# V2            .        
# V3            .        
# V4           -2.9373874
# V5            .        
# V6            .        
# V7            2.0588415
# V8            2.0189062
# V9            .        
# V10           .        
# V11          -4.7076324
# V12          -2.3433984


#4
library(tree)

tr4 = tree(y[itrain]~., data=x[itrain,])

plot(tr4)
text(tr4)

summary(tr4)
length(summary(tr4)$used)

# Variables actually used in tree construction:
#   [1] "V11" "V59" "V24" "V20" "V3"  "V16" "V49" "V55" "V31" "V1" 
# 10 were used

#5

library(randomForest)

rf5 = randomForest(y[itrain]~., data=x[itrain,])

varImpPlot(rf5)


#6

tr4.p = predict(tr4, newdata = x[-itrain,], type = 'class')
rf5.p = predict(rf5, newdata = x[-itrain,], type = 'class')

library(caret)

tr4.cm = confusionMatrix(y[-itrain], tr4.p)
rf5.cm = confusionMatrix(y[-itrain], rf5.p)

#confusion tables
tr4.cm$table
rf5.cm$table

#error rates
1-tr4.cm$overall[1]
1-rf5.cm$overall[1]


#7
library(pROC)

tr4.prob = predict(tr4, newdata = x[-itrain,], type = 'vector')
rf5.prob = predict(rf5, newdata = x[-itrain,], type = 'prob')

tr4.roc = roc(y[-itrain], tr4.prob[,2])
rf5.roc = roc(y[-itrain], rf5.prob[,2])

par(mfrow=c(1,2))
plot(tr4.roc, main='Tree')
plot(rf5.roc, main='Random Forest')
par(mfrow=c(1,1))

#AUC
tr4.roc$auc
rf5.roc$auc

# Random forest is more accurate has lower error rate higher auc and better ROC 
# (i.e up and to the right)

