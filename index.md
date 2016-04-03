# Practical Machine Learning Final Project
Sner88  
1 april 2016  



## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement Â– a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this data set, the participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants toto predict the manner in which praticipants did the exercise.

The dependent variable or response is the “classe” variable in the training set.

## Data

Loading libraries


```r
# loading libraries
library(Matrix)
library(lme4)
library(pbkrtest)
library(lattice)
library(ggplot2)
library(caret)
library(e1071)
suppressMessages(library(randomForest))
library(AppliedPredictiveModeling)
library(tree)
library(devtools)
```

Loading and examine the data

```r
# loading and examine the data
setwd("/Users/renswajon/Coursera")

trainingOriginal = read.csv(file="pml-training.csv", header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings = c('NA','','#DIV/0!'))
testOriginal = read.csv(file="pml-testing.csv", header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings = c('NA','','#DIV/0!'))

trainingOriginal$classe = as.factor(trainingOriginal$classe) 

dim(trainingOriginal)
```

```
## [1] 19622   160
```

```r
dim(testOriginal)
```

```
## [1]  20 160
```

```r
summary(trainingOriginal$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

## Feature selection
Here I try to clean the variables which are unusefull by pre-screening the data and remove unrelevant variables or missing values:

```r
NAs = apply(trainingOriginal,2,function(x) {sum(is.na(x))}) 
trainingOriginal = trainingOriginal[,which(NAs == 0)]
NAs = apply(testOriginal,2,function(x) {sum(is.na(x))}) 
testOriginal = testOriginal[,which(NAs == 0)]
```

## Pre-processing the variables

```r
preProc = which(lapply(trainingOriginal, class) %in% "numeric")

preObj = preProcess(trainingOriginal[,preProc],method=c('knnImpute', 'center', 'scale'))
train = predict(preObj, trainingOriginal[,preProc])
train$classe = trainingOriginal$classe

test = predict(preObj,testOriginal[,preProc])
```


## Split data to training and testing for cross validation
Here I split the data into one set for training and one set for cross validation. The cross validation set will be used as the train control method for the model:

```r
set.seed(30000)

inTrain = createDataPartition(train$classe, p = 3/4, list=FALSE)
training = train[inTrain,]
crossValidation = train[-inTrain,]
dim(training);dim(test)
```

```
## [1] 14718    28
```

```
## [1] 20 27
```

## Train model using Random Forrest
I create the train model using rf. I've tried other models, however the Random Forest model seems the most accurate.

```r
fit = train(classe ~., method="rf", data=training, trControl=trainControl(method='cv'), number=5, allowParallel=TRUE )

save(fit,file="/Users/renswajon/Coursera/fit.R")
```

## Accuracy results and prediction error
Check on accuracy of the training set:

```r
train_Pred <- predict(fit, training)
confusionMatrix(train_Pred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

Check on accuracy of the cross-validation set:

```r
cross_Pred <- predict(fit, crossValidation)
confusionMatrix(cross_Pred, crossValidation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    4    0    0    0
##          B    0  943    7    0    0
##          C    0    2  841    9    2
##          D    0    0    7  795    3
##          E    0    0    0    0  896
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9931          
##                  95% CI : (0.9903, 0.9952)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9912          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9937   0.9836   0.9888   0.9945
## Specificity            0.9989   0.9982   0.9968   0.9976   1.0000
## Pos Pred Value         0.9971   0.9926   0.9848   0.9876   1.0000
## Neg Pred Value         1.0000   0.9985   0.9965   0.9978   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1923   0.1715   0.1621   0.1827
## Detection Prevalence   0.2853   0.1937   0.1741   0.1642   0.1827
## Balanced Accuracy      0.9994   0.9960   0.9902   0.9932   0.9972
```
Check on results:

```r
test_Pred = predict(fit, test)
test_Pred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
##


