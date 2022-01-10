library(tidyverse)
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(lubridate)
library(readxl)
library(mice)
library(car)
library(caret)
library(magrittr)
library(readxl)
library (caTools)
library(glmnet)
library(Amelia)

#Load the Data
houses_training <- read.csv("/Users/jacquelinemak/MMA /MMA 867 Predictive Modelling/Assignment 1/house-prices-advanced-regression-techniques/train.csv", stringsAsFactors = T)
houses_testing <- read.csv("/Users/jacquelinemak/MMA /MMA 867 Predictive Modelling/Assignment 1/house-prices-advanced-regression-techniques/test.csv", stringsAsFactors = T)

#Explore the data
str(houses_training)
head(houses_training)
summary(houses_training)

#Significant cleaning of data
#check for duplicates

distinct(houses_training)
dim(houses_training)

#High level view of what data is missing
missmap(houses_training, col=c('grey', 'steelblue'), y.cex=0.5, x.cex=0.8)

#Number of missing data for each variable
sort(sapply(houses_training, function(x) { sum(is.na(x)) }), decreasing=TRUE)
#PoolQC, MiscFeature, Alley & Fence are missing most their data, so let's remove these columns
#also remove ID column

exclude <-c('PoolQC', 'MiscFeature', 'Alley', 'Fence')
include <- setdiff(names(houses_training), exclude)
houses_training <- houses_training[include]

#For the rest of the data, we can use the mice command, cart = will work for all variable types(numeric, logical, factor)
#we will run 1 imputation

imputed_houses_training <- mice(houses_training, m=1, method='cart', printFlag=FALSE)

#explore the imputed data to ensure everything looks ok
#there should be a positive correlation with LotFrontage (Linear feet of street connected to property) with LotArea (Lot size in square feet)
xyplot(imputed_houses_training, LotFrontage ~ LotArea)

#explore the distribution of the imputed data
densityplot(imputed_houses_training, ~LotFrontage) #looks normal

#merge imputed values to our original data set
houses_training_complete <- complete(imputed_houses_training)

#Confirm there are no NA data
sum(sapply(houses_training_complete, function(x) { sum(is.na(x)) }))

#### Apply the same concept to the testing data #### ----------------
str(houses_testing)
head(houses_testing)
summary(houses_testing)

#High level view of what data is missing in testing set
missmap(houses_testing, col=c('grey', 'steelblue'), y.cex=0.5, x.cex=0.8)

#Number of missing data for each variable
sort(sapply(houses_testing, function(x) { sum(is.na(x)) }), decreasing=TRUE)

exclude <-c('PoolQC', 'MiscFeature', 'Alley', 'Fence')
include <- setdiff(names(houses_testing), exclude)
houses_testing <- houses_testing[include]

#For the rest of the data, we can use the mice command
imputed_houses_testing <- mice(houses_testing, m=1, method='cart', printFlag=FALSE)

#merge imputed values to our original data set
houses_testing_complete <- complete(imputed_houses_testing)

#test set doesn't have SalePrice variable, so we need to add it to ensure data and test sets have the same amount of variables
houses_testing_complete$SalePrice = NA

#combine train and test set (now both have same amount of features - 77)
houses_data <- rbind(houses_training_complete, houses_testing_complete)

#remove Utilities as it has zero variance
houses_data = houses_data[,-9]

#validate there are no missing data except for SalePrice
sort(sapply(houses_data, function(x) { sum(is.na(x)) }), decreasing=TRUE)

#-----------
#Log Transformation of SalePrice Variable - make distribution of the target variable normal
#Tranform it by taking log.

# Plot histogram of SalePrice Variable - Right skewed
qplot(SalePrice, data = houses_data, bins = 50, main = "Right skewed distribution")

## Log transformation of the target variable
houses_data$SalePrice <- log(houses_data$SalePrice + 1)

## Normal distribution after transformation
qplot(SalePrice, data = houses_data, bins = 50, main = "Normal distribution after log transformation")

#-----------------------------

#look dimensions of train and test set combined
dim(houses_data)
str(houses_data)
tail(houses_data)

#------------------ Splitting data in train and test set

library (caTools)
set.seed(6)

training_houses_data <- houses_data[1:1460,]
testing_houses_data <- houses_data[1461:2919,]

x.train <- subset(training_houses_data, training_houses_data[,1]<=1000)


training_houses_dataX=training_houses_data[c(1:75)]
testing_houses_dataX=testing_houses_data[c(1:75)]

#-------------------------------
#Build a regression model for house price prediction  
#let's start with a simple linear regression 

reg1=lm(SalePrice~OverallCond,data=training_houses_data) 
summary(reg1)
#passed p-test and f-test
plot(reg1)
plot(density(resid(reg1)))

# Extract individual components of the fit 
summary(reg1)$coefficients
summary(reg1)$residuals
summary(reg1)$r.squared
summary(reg1)$sigma

# Predictions 
# Use the fitted model to predict response in the testing data
predicted.saleprices.reg1<-exp(predict(reg1, testing_houses_dataX)) 
# Confidence intervals
predicted.saleprices.reg1.CI<-exp(predict(reg1, testing_houses_dataX, interval = "confidence"))
# Prediction intervals
predicted.saleprices.reg1.PI<-exp(predict(reg1, testing_houses_dataX, interval = "prediction"))

# Multiple linear regression - Use all predictors available ------------------

reg2 <- lm(SalePrice ~ ., training_houses_data)
summary(reg2)
plot(reg2)
plot(density(resid(reg2)))

predicted.saleprices.reg2<-exp(predict(reg2, testing_houses_dataX))

write.csv(predicted.saleprices.reg2, "/Users/jacquelinemak/MMA /MMA 867 Predictive Modelling/Predicted Sale Prices_v2.csv") 

#-----Log-y linear regression model

#Log Transformation of SalePrice Variable - make distribution of the target variable normal
#Tranform it by taking log.

# Plot histogram of SalePrice Variable - Right skewed
qplot(SalePrice, data = houses_data, bins = 50, main = "Right skewed distribution")

## Log transformation of the target variable
houses_data$SalePrice <- log(houses_data$SalePrice + 1)

## Normal distribution after transformation
qplot(SalePrice, data = houses_data, bins = 50, main = "Normal distribution after log transformation")

#Log-y linear regression model
reg3.logY<-lm(log(SalePrice)~., data=training_houses_data) #added "log()" for response only

summary(reg3.logY)
par(mfrow=c(2,2)) 
plot(reg3.logY) 

#All three models have heterscedascity, collinearity and residuals that are not normally distributed. 

#------------ Lasso regression

#Splitting train dataset into Training and Validation to evaluate the model

inx <- sample.split(seq_len(nrow(training_houses_data)), 0.75)
inTrain <- training_houses_data[inx, ]
inPrediction <- training_houses_data[!inx, ]

inPredictionX=inPrediction[c(1:75)];
actual_saleprices=inPrediction[76];

library(glmnet)

#create the y variable and matrix (capital X) of x variables 
#(will make the code below easier to read + will ensure that all interactions exist)
y.inTrain<-(inTrain$SalePrice)
X<-model.matrix(~., training_houses_data)[,-1] 
X<-cbind(training_houses_data$Id,X)

# split X into training and validation set in Training data
X.inTrain<-X[inx,]
X.validation<-X[!inx,]

#LASSO (alpha=1)
lasso.fit<-glmnet(x = X.inTrain, y = y.inTrain, alpha = 1)
plot(lasso.fit, xvar = "lambda")

#selecting the best penalty lambda
crossval <-  cv.glmnet(x = X.inTrain, y = y.inTrain, alpha = 1) 
plot(crossval) 
penalty.lasso <- crossval$lambda.min #this will tell u what the best lambda is#determine optimal penalty parameter, lambda
log(penalty.lasso) #see where it was on the graph #-5.8949 is the best lambda (lowest point in lambda
plot(crossval,xlim=c(-8.5,-6),ylim=c(0.006,0.008)) # lets zoom-in
lasso.opt.fit <-glmnet(x = X.inTrain, y = y.inTrain, alpha = 1, lambda = penalty.lasso) #estimate the model with the optimal penalty
coef(lasso.opt.fit) #resultant model coefficients of the lasso.opt.fit model

lasso.validation <- exp(predict(lasso.opt.fit, s = penalty.lasso, newx =X.validation))

lasso.validation.MSE <- mean((lasso.validation-inPrediction[,76] )^2) #calculate and display MSE in the validation set
lasso.validation.MAPE <- mean(abs(lasso.validation-inPrediction[,76])/inPrediction[,76]*100) # MAPE: mean absolute percentage error 
#the MAPE is easier to interpret than MSE, gives you a 5.16%, your error is only 5.167% 

#---------------------Ridge Regression (alpha=0)

ridge.fit<-glmnet(x = X.inTrain, y = y.inTrain, alpha = 0)
plot(ridge.fit, xvar = "lambda") #the coefficient of ridge gradually converge to 0

#selecting the best penalty lambda
crossval.ridge <-  cv.glmnet(x = X.inTrain, y = y.inTrain, alpha = 0)
plot(crossval.ridge) 
penalty.ridge <- crossval.ridge$lambda.min 
log(penalty.ridge) #best lambda is -3.255425
ridge.opt.fit <-glmnet(x = X.inTrain, y = y.inTrain, alpha = 0, lambda = penalty.ridge) #estimate the model with that
coef(ridge.opt.fit) #retains more variable than LASSO

ridge.testing <- exp(predict(ridge.opt.fit, s = penalty.ridge, newx =X.validation))
ridge.testing.MSE <- mean((ridge.testing- inPrediction[,76] )^2) #calculate and display MSE  in the testing set
ridge.testing.MAPE <-mean(abs(ridge.testing-inPrediction[,76])/inPrediction[,76]*100)  # MAPE: mean absolute percentage error 
#ERROR IS 6.439% which is a little higher than LASSO

# MODEL SELECTION: comparing the prediction error in the VALIDATION set
crossval # LASSO: min MSE in validation set is 0.007632
crossval.ridge # Ridge: min MSE in validation set is 0.01293
# LASSO is better, so use it for prediction

# MODEL ASSESSMENT: Report the accuracy of the final chosen model: LASSO in the validation set
lasso.validation.MSE 
lasso.validation.MAPE

#--------------Retraining on whole training set and final submission using Lasso in the TESTING set
#define observations in testing set
training_houses_data <- houses_data[1:1460,]
testing_houses_data <- houses_data[1461:2919,]

#split the testing set in x part and y part 
testing_houses_dataX=testing_houses_data[c(1:75)]
#last column is my response
predicted_sale_prices_final=testing_houses_data[c(76)] # responses in the testing set

### Regularizations (LASSO)

library(glmnet)

#create the y variable and matrix (capital X) of x variables 
#(will make the code below easier to read + will ensure that all interactions exist)
y_Train<-log(training_houses_data$SalePrice)
X_final<-model.matrix(~., houses_data) #take out the intercept variable (first column is all 1s, so remove it)

# split X into training and validation set in Training data
X.Train<-X_final[1:1460,]
X.Test<-X_final[1461:2919,]

#LASSO (alpha=1)
lasso.fit<-glmnet(x = X.Train, y = y_Train, alpha = 1)
plot(lasso.fit, xvar = "lambda")

#selecting the best penalty lambda
crossval <-  cv.glmnet(x = X.Train, y = y_Train, alpha = 1) 
plot(crossval) 
penalty.lasso <- crossval$lambda.min #this will tell u what the best lambda is#determine optimal penalty parameter, lambda
log(penalty.lasso) #see where it was on the graph #-5.8949 is the best lambda (lowest point in lambda, slight more than 320 variables)
plot(crossval,xlim=c(-8.5,-6),ylim=c(0.006,0.008)) # lets zoom-in
lasso.opt.fit <-glmnet(x = X.Train, y = y_Train, alpha = 1, lambda = penalty.lasso) #estimate the model with the optimal penalty
coef(lasso.opt.fit) #resultant model coefficients of the lasso.opt.fit model

lasso.testing <- exp(predict(lasso.opt.fit, s = penalty.lasso, newx =X.Test))

#------------end 
