# Practical Machine Learning - Course Project
Wayne Chan  
January 21, 2017  



## Introduction

This document is the final report of Coursera's course named Practical Machine Learning, as part of the Specialization in Data Science. It was built up by R programming with caret package, which provides a wide range functions of machine learning algorithms for modeling and prediction. The purpose is to study how to built up a machine learning algorithms and choose the better model for prediction. 

This project is concerned with identifying the exercise categories of 6 participants performed exercise. The exercise categories include five exercise: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Two datasets will be use in this project, trainging data and testing data. Trainging data is use for model building and testing dataset is use for further predictions.



## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3xsbS5bVX



## Data Preparetion and Cleaning

There are two source data files, training data and test data, which will download from website.


```r
# read the source date from websits
TrainData <- read.csv("./pml-training.csv", na.strings=c("NA",'#DIV/0!',""))
TestData <- read.csv("./pml-testing.csv", na.strings=c("NA",'#DIV/0!',""))
dim(TrainData);dim(TestData)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

The training data contained 19622 rows of data with 160 columns and the test data has 20 rows with same columns as training data.

Since the training data contained some variables with lots of missing values, we will do some data cleaning to reduce redundant varialbs before modeling.

1. remove variables with zero variance

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
TrainData <- TrainData[, -nearZeroVar(TrainData)]
```

2. remove variables with NAs greater than 90%

```r
NAVars <-sapply(TrainData, function(x) mean(is.na(x))>0.9)
TrainData <- TrainData[, NAVars==F]
```

3. remove first ID column "X"

```r
TrainData <- TrainData[, -1]
```

Synchronize the structure of Training data and test data

```r
TestData <- TestData[colnames(TrainData[,names(TrainData)!="classe"])]
dim(TrainData);dim(TestData)
```

```
## [1] 19622    58
```

```
## [1] 20 57
```



## Data Spliting

The sample size of training data set is large enough, hence i will split the data into two partitions to perform cross-validation, a training subset with 60% of the original training data set and the remaining 40% to be testing subset.


```r
set.seed(2017)
inTrain = createDataPartition(TrainData$classe, p=.60)[[1]]
SubTrain = TrainData[ inTrain,]
SubTest = TrainData[-inTrain,]
nrow(SubTrain);nrow(SubTest)
```

```
## [1] 11776
```

```
## [1] 7846
```

```r
summary(SubTrain$classe);summary(SubTest$classe)
```

```
##    A    B    C    D    E 
## 3348 2279 2054 1930 2165
```

```
##    A    B    C    D    E 
## 2232 1518 1368 1286 1442
```



## Machine Learning Algorithms Modeling

I will try to fit the model by different methods of classifications and compare the results after modeling.



### Method 1: Linear Discriminant Analysis

Modeling:

```r
ModelLDA <- train(classe~., data=SubTrain, preProcess=c("pca","scale","center"), method="lda")
```

```
## Loading required package: MASS
```

```r
ModelLDA
```

```
## Linear Discriminant Analysis 
## 
## 11776 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction (79), scaled
##  (79), centered (79) 
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results:
## 
##   Accuracy   Kappa   
##   0.7120842  0.637318
## 
## 
```

Cross-validation & Confusion Matrix:

```r
PredictLDA <- predict(ModelLDA, newdata=SubTest)
confusionMatrix(PredictLDA, SubTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1712  200   15    0    0
##          B  428  923  287   48    0
##          C   91  376  956  197   10
##          D    0   19  103  831  303
##          E    1    0    7  210 1129
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7075          
##                  95% CI : (0.6973, 0.7175)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6319          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7670   0.6080   0.6988   0.6462   0.7829
## Specificity            0.9617   0.8794   0.8960   0.9352   0.9660
## Pos Pred Value         0.8884   0.5474   0.5865   0.6616   0.8382
## Neg Pred Value         0.9121   0.9034   0.9337   0.9310   0.9518
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2182   0.1176   0.1218   0.1059   0.1439
## Detection Prevalence   0.2456   0.2149   0.2077   0.1601   0.1717
## Balanced Accuracy      0.8644   0.7437   0.7974   0.7907   0.8744
```



### Method 2: Decision Tree

Modeling:

```r
require("rattle")
```

```
## Loading required package: rattle
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
require("rpart")
```

```
## Loading required package: rpart
```

```r
ModelDT <- rpart(classe~., data=SubTrain, method="class")
ModelDT
```

```
## n= 11776 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##   1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
##     2) cvtd_timestamp=02/12/2011 13:32,02/12/2011 13:33,02/12/2011 14:56,02/12/2011 14:57,05/12/2011 11:23,05/12/2011 11:24,05/12/2011 14:22,05/12/2011 14:23,28/11/2011 14:13,30/11/2011 17:10,30/11/2011 17:11 5758 2423 A (0.58 0.29 0.13 0 0)  
##       4) cvtd_timestamp=02/12/2011 13:32,02/12/2011 13:33,02/12/2011 14:56,05/12/2011 11:23,05/12/2011 14:22,28/11/2011 14:13,30/11/2011 17:10 2327  186 A (0.92 0.08 0 0 0)  
##         8) raw_timestamp_part_1< 1.322833e+09 1732    0 A (1 0 0 0 0) *
##         9) raw_timestamp_part_1>=1.322833e+09 595  186 A (0.69 0.31 0 0 0)  
##          18) accel_dumbbell_y< 64.5 403    0 A (1 0 0 0 0) *
##          19) accel_dumbbell_y>=64.5 192    6 B (0.031 0.97 0 0 0) *
##       5) cvtd_timestamp=02/12/2011 14:57,05/12/2011 11:24,05/12/2011 14:23,30/11/2011 17:11 3431 1953 B (0.35 0.43 0.22 0 0)  
##        10) roll_forearm< 123.5 2037  915 A (0.55 0.44 0.0088 0 0)  
##          20) magnet_dumbbell_y< 422.5 1571  478 A (0.7 0.29 0.011 0 0)  
##            40) raw_timestamp_part_1< 1.323084e+09 1029  118 A (0.89 0.11 0.0087 0 0) *
##            41) raw_timestamp_part_1>=1.323084e+09 542  191 B (0.34 0.65 0.017 0 0)  
##              82) num_window< 68.5 144    0 A (1 0 0 0 0) *
##              83) num_window>=68.5 398   47 B (0.095 0.88 0.023 0 0) *
##          21) magnet_dumbbell_y>=422.5 466   29 B (0.062 0.94 0 0 0) *
##        11) roll_forearm>=123.5 1394  653 C (0.052 0.42 0.53 0 0)  
##          22) raw_timestamp_part_1< 1.322673e+09 305   41 B (0.13 0.87 0 0 0) *
##          23) raw_timestamp_part_1>=1.322673e+09 1089  348 C (0.028 0.29 0.68 0 0)  
##            46) magnet_belt_x< 1.5 270   77 B (0.048 0.71 0.24 0 0) *
##            47) magnet_belt_x>=1.5 819  142 C (0.022 0.15 0.83 0 0) *
##     3) cvtd_timestamp=02/12/2011 13:34,02/12/2011 13:35,02/12/2011 14:58,02/12/2011 14:59,05/12/2011 11:25,05/12/2011 14:24,28/11/2011 14:14,28/11/2011 14:15,30/11/2011 17:12 6018 3853 E (0.0022 0.1 0.22 0.32 0.36)  
##       6) cvtd_timestamp=02/12/2011 13:34,02/12/2011 14:58,28/11/2011 14:14 2530 1440 C (0.0051 0.24 0.43 0.28 0.04)  
##        12) raw_timestamp_part_1< 1.32249e+09 355   13 B (0.037 0.96 0 0 0) *
##        13) raw_timestamp_part_1>=1.32249e+09 2175 1085 C (0 0.13 0.5 0.33 0.046)  
##          26) raw_timestamp_part_1< 1.322838e+09 1709  619 C (0 0.16 0.64 0.2 0)  
##            52) pitch_belt< -43.15 275   31 B (0 0.89 0.11 0 0) *
##            53) pitch_belt>=-43.15 1434  375 C (0 0.02 0.74 0.24 0)  
##             106) roll_belt>=0.53 1291  232 C (0 0.022 0.82 0.16 0)  
##               212) magnet_belt_y>=551 1203  144 C (0 0.024 0.88 0.096 0) *
##               213) magnet_belt_y< 551 88    0 D (0 0 0 1 0) *
##             107) roll_belt< 0.53 143    0 D (0 0 0 1 0) *
##          27) raw_timestamp_part_1>=1.322838e+09 466  100 D (0 0 0 0.79 0.21)  
##            54) raw_timestamp_part_1< 1.322838e+09 364    0 D (0 0 0 1 0) *
##            55) raw_timestamp_part_1>=1.322838e+09 102    2 E (0 0 0 0.02 0.98) *
##       7) cvtd_timestamp=02/12/2011 13:35,02/12/2011 14:59,05/12/2011 11:25,05/12/2011 14:24,28/11/2011 14:15,30/11/2011 17:12 3488 1423 E (0 0 0.059 0.35 0.59)  
##        14) roll_belt< 125.5 2486 1283 D (0 0 0.082 0.48 0.43)  
##          28) accel_forearm_x< -88.5 1292  356 D (0 0 0.07 0.72 0.21)  
##            56) magnet_belt_y>=578.5 1122  213 D (0 0 0.081 0.81 0.11) *
##            57) magnet_belt_y< 578.5 170   27 E (0 0 0 0.16 0.84) *
##          29) accel_forearm_x>=-88.5 1194  381 E (0 0 0.095 0.22 0.68) *
##        15) roll_belt>=125.5 1002   15 E (0 0 0 0.015 0.99) *
```

```r
fancyRpartPlot(ModelDT)
```

![](Course_Project_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

Cross-validation & Confusion Matrix:

```r
PredictDT <- predict(ModelDT, newdata=SubTest, type="class")
confusionMatrix(PredictDT, SubTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2110   66    6    0    0
##          B  109 1361   82    0    0
##          C   13   91 1138   82    0
##          D    0    0   65 1013   85
##          E    0    0   77  191 1357
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8895          
##                  95% CI : (0.8824, 0.8964)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8603          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9453   0.8966   0.8319   0.7877   0.9411
## Specificity            0.9872   0.9698   0.9713   0.9771   0.9582
## Pos Pred Value         0.9670   0.8769   0.8595   0.8710   0.8351
## Neg Pred Value         0.9785   0.9751   0.9647   0.9592   0.9863
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2689   0.1735   0.1450   0.1291   0.1730
## Detection Prevalence   0.2781   0.1978   0.1687   0.1482   0.2071
## Balanced Accuracy      0.9663   0.9332   0.9016   0.8824   0.9496
```



### Method 3: Generalized Boosted Models

Modeling:

```r
ModelGBM <- train(classe~., data=SubTrain, preProcess=c("pca","scale","center"), method="gbm")
```

```
## Loading required package: gbm
```

```
## Warning: package 'gbm' was built under R version 3.3.2
```

```
## Loading required package: survival
```

```
## Warning: package 'survival' was built under R version 3.3.2
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0977
##      2        1.5499             nan     0.1000    0.0722
##      3        1.5034             nan     0.1000    0.0676
##      4        1.4611             nan     0.1000    0.0580
##      5        1.4247             nan     0.1000    0.0511
##      6        1.3933             nan     0.1000    0.0398
##      7        1.3686             nan     0.1000    0.0444
##      8        1.3409             nan     0.1000    0.0345
##      9        1.3186             nan     0.1000    0.0300
##     10        1.2994             nan     0.1000    0.0356
##     20        1.1381             nan     0.1000    0.0168
##     40        0.9539             nan     0.1000    0.0080
##     60        0.8324             nan     0.1000    0.0054
##     80        0.7461             nan     0.1000    0.0045
##    100        0.6810             nan     0.1000    0.0045
##    120        0.6304             nan     0.1000    0.0024
##    140        0.5887             nan     0.1000    0.0015
##    150        0.5704             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1491
##      2        1.5169             nan     0.1000    0.1172
##      3        1.4423             nan     0.1000    0.0923
##      4        1.3841             nan     0.1000    0.0841
##      5        1.3312             nan     0.1000    0.0711
##      6        1.2857             nan     0.1000    0.0684
##      7        1.2439             nan     0.1000    0.0666
##      8        1.2027             nan     0.1000    0.0581
##      9        1.1673             nan     0.1000    0.0537
##     10        1.1335             nan     0.1000    0.0504
##     20        0.9146             nan     0.1000    0.0284
##     40        0.6946             nan     0.1000    0.0108
##     60        0.5795             nan     0.1000    0.0063
##     80        0.4987             nan     0.1000    0.0048
##    100        0.4403             nan     0.1000    0.0028
##    120        0.3949             nan     0.1000    0.0021
##    140        0.3561             nan     0.1000    0.0014
##    150        0.3414             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2000
##      2        1.4846             nan     0.1000    0.1381
##      3        1.3968             nan     0.1000    0.1268
##      4        1.3188             nan     0.1000    0.1065
##      5        1.2533             nan     0.1000    0.0927
##      6        1.1959             nan     0.1000    0.0809
##      7        1.1448             nan     0.1000    0.0729
##      8        1.1012             nan     0.1000    0.0660
##      9        1.0607             nan     0.1000    0.0600
##     10        1.0216             nan     0.1000    0.0545
##     20        0.7767             nan     0.1000    0.0207
##     40        0.5604             nan     0.1000    0.0093
##     60        0.4503             nan     0.1000    0.0052
##     80        0.3773             nan     0.1000    0.0036
##    100        0.3245             nan     0.1000    0.0015
##    120        0.2849             nan     0.1000    0.0029
##    140        0.2508             nan     0.1000    0.0009
##    150        0.2369             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0951
##      2        1.5511             nan     0.1000    0.0674
##      3        1.5098             nan     0.1000    0.0586
##      4        1.4743             nan     0.1000    0.0475
##      5        1.4447             nan     0.1000    0.0455
##      6        1.4170             nan     0.1000    0.0399
##      7        1.3916             nan     0.1000    0.0329
##      8        1.3701             nan     0.1000    0.0385
##      9        1.3465             nan     0.1000    0.0301
##     10        1.3260             nan     0.1000    0.0298
##     20        1.1700             nan     0.1000    0.0202
##     40        0.9788             nan     0.1000    0.0108
##     60        0.8564             nan     0.1000    0.0064
##     80        0.7716             nan     0.1000    0.0046
##    100        0.7055             nan     0.1000    0.0028
##    120        0.6546             nan     0.1000    0.0027
##    140        0.6117             nan     0.1000    0.0015
##    150        0.5931             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1510
##      2        1.5199             nan     0.1000    0.1098
##      3        1.4533             nan     0.1000    0.0994
##      4        1.3917             nan     0.1000    0.0781
##      5        1.3443             nan     0.1000    0.0693
##      6        1.3013             nan     0.1000    0.0605
##      7        1.2622             nan     0.1000    0.0525
##      8        1.2295             nan     0.1000    0.0546
##      9        1.1947             nan     0.1000    0.0495
##     10        1.1637             nan     0.1000    0.0475
##     20        0.9544             nan     0.1000    0.0248
##     40        0.7254             nan     0.1000    0.0106
##     60        0.5986             nan     0.1000    0.0061
##     80        0.5162             nan     0.1000    0.0042
##    100        0.4555             nan     0.1000    0.0038
##    120        0.4097             nan     0.1000    0.0038
##    140        0.3715             nan     0.1000    0.0020
##    150        0.3551             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1927
##      2        1.4908             nan     0.1000    0.1402
##      3        1.4053             nan     0.1000    0.1096
##      4        1.3378             nan     0.1000    0.1044
##      5        1.2739             nan     0.1000    0.0911
##      6        1.2190             nan     0.1000    0.0777
##      7        1.1716             nan     0.1000    0.0706
##      8        1.1283             nan     0.1000    0.0665
##      9        1.0875             nan     0.1000    0.0537
##     10        1.0528             nan     0.1000    0.0460
##     20        0.8180             nan     0.1000    0.0214
##     40        0.5953             nan     0.1000    0.0092
##     60        0.4741             nan     0.1000    0.0049
##     80        0.3937             nan     0.1000    0.0030
##    100        0.3390             nan     0.1000    0.0024
##    120        0.2976             nan     0.1000    0.0024
##    140        0.2638             nan     0.1000    0.0016
##    150        0.2489             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0976
##      2        1.5494             nan     0.1000    0.0679
##      3        1.5056             nan     0.1000    0.0602
##      4        1.4660             nan     0.1000    0.0476
##      5        1.4359             nan     0.1000    0.0453
##      6        1.4069             nan     0.1000    0.0396
##      7        1.3809             nan     0.1000    0.0387
##      8        1.3569             nan     0.1000    0.0292
##      9        1.3377             nan     0.1000    0.0325
##     10        1.3171             nan     0.1000    0.0299
##     20        1.1650             nan     0.1000    0.0183
##     40        0.9740             nan     0.1000    0.0088
##     60        0.8461             nan     0.1000    0.0077
##     80        0.7570             nan     0.1000    0.0049
##    100        0.6864             nan     0.1000    0.0043
##    120        0.6335             nan     0.1000    0.0027
##    140        0.5905             nan     0.1000    0.0016
##    150        0.5711             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1453
##      2        1.5168             nan     0.1000    0.1073
##      3        1.4479             nan     0.1000    0.0952
##      4        1.3867             nan     0.1000    0.0838
##      5        1.3337             nan     0.1000    0.0635
##      6        1.2923             nan     0.1000    0.0650
##      7        1.2514             nan     0.1000    0.0581
##      8        1.2151             nan     0.1000    0.0492
##      9        1.1838             nan     0.1000    0.0497
##     10        1.1519             nan     0.1000    0.0401
##     20        0.9368             nan     0.1000    0.0220
##     40        0.7109             nan     0.1000    0.0144
##     60        0.5802             nan     0.1000    0.0054
##     80        0.4961             nan     0.1000    0.0051
##    100        0.4355             nan     0.1000    0.0038
##    120        0.3877             nan     0.1000    0.0026
##    140        0.3502             nan     0.1000    0.0017
##    150        0.3343             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1906
##      2        1.4906             nan     0.1000    0.1391
##      3        1.4041             nan     0.1000    0.1279
##      4        1.3238             nan     0.1000    0.1007
##      5        1.2594             nan     0.1000    0.0881
##      6        1.2047             nan     0.1000    0.0743
##      7        1.1584             nan     0.1000    0.0676
##      8        1.1160             nan     0.1000    0.0577
##      9        1.0792             nan     0.1000    0.0586
##     10        1.0429             nan     0.1000    0.0475
##     20        0.8061             nan     0.1000    0.0258
##     40        0.5721             nan     0.1000    0.0094
##     60        0.4529             nan     0.1000    0.0056
##     80        0.3773             nan     0.1000    0.0031
##    100        0.3224             nan     0.1000    0.0017
##    120        0.2814             nan     0.1000    0.0019
##    140        0.2493             nan     0.1000    0.0008
##    150        0.2356             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1010
##      2        1.5515             nan     0.1000    0.0678
##      3        1.5101             nan     0.1000    0.0644
##      4        1.4713             nan     0.1000    0.0521
##      5        1.4391             nan     0.1000    0.0431
##      6        1.4121             nan     0.1000    0.0405
##      7        1.3872             nan     0.1000    0.0356
##      8        1.3644             nan     0.1000    0.0328
##      9        1.3428             nan     0.1000    0.0319
##     10        1.3218             nan     0.1000    0.0303
##     20        1.1672             nan     0.1000    0.0209
##     40        0.9751             nan     0.1000    0.0094
##     60        0.8453             nan     0.1000    0.0076
##     80        0.7602             nan     0.1000    0.0050
##    100        0.6935             nan     0.1000    0.0031
##    120        0.6417             nan     0.1000    0.0026
##    140        0.5977             nan     0.1000    0.0021
##    150        0.5801             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1439
##      2        1.5196             nan     0.1000    0.1085
##      3        1.4526             nan     0.1000    0.0985
##      4        1.3911             nan     0.1000    0.0822
##      5        1.3401             nan     0.1000    0.0646
##      6        1.2982             nan     0.1000    0.0609
##      7        1.2601             nan     0.1000    0.0592
##      8        1.2225             nan     0.1000    0.0498
##      9        1.1911             nan     0.1000    0.0466
##     10        1.1620             nan     0.1000    0.0455
##     20        0.9474             nan     0.1000    0.0260
##     40        0.7164             nan     0.1000    0.0131
##     60        0.5925             nan     0.1000    0.0076
##     80        0.5111             nan     0.1000    0.0038
##    100        0.4500             nan     0.1000    0.0029
##    120        0.4033             nan     0.1000    0.0017
##    140        0.3656             nan     0.1000    0.0016
##    150        0.3496             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1883
##      2        1.4937             nan     0.1000    0.1422
##      3        1.4056             nan     0.1000    0.1131
##      4        1.3350             nan     0.1000    0.0868
##      5        1.2793             nan     0.1000    0.0835
##      6        1.2264             nan     0.1000    0.0788
##      7        1.1776             nan     0.1000    0.0684
##      8        1.1335             nan     0.1000    0.0612
##      9        1.0947             nan     0.1000    0.0626
##     10        1.0557             nan     0.1000    0.0511
##     20        0.8100             nan     0.1000    0.0237
##     40        0.5819             nan     0.1000    0.0091
##     60        0.4613             nan     0.1000    0.0044
##     80        0.3853             nan     0.1000    0.0041
##    100        0.3330             nan     0.1000    0.0024
##    120        0.2915             nan     0.1000    0.0018
##    140        0.2587             nan     0.1000    0.0015
##    150        0.2445             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1017
##      2        1.5490             nan     0.1000    0.0771
##      3        1.5014             nan     0.1000    0.0648
##      4        1.4623             nan     0.1000    0.0540
##      5        1.4281             nan     0.1000    0.0459
##      6        1.3997             nan     0.1000    0.0367
##      7        1.3755             nan     0.1000    0.0355
##      8        1.3526             nan     0.1000    0.0345
##      9        1.3312             nan     0.1000    0.0347
##     10        1.3085             nan     0.1000    0.0283
##     20        1.1569             nan     0.1000    0.0167
##     40        0.9728             nan     0.1000    0.0104
##     60        0.8526             nan     0.1000    0.0062
##     80        0.7634             nan     0.1000    0.0042
##    100        0.6984             nan     0.1000    0.0040
##    120        0.6453             nan     0.1000    0.0029
##    140        0.6031             nan     0.1000    0.0019
##    150        0.5844             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1526
##      2        1.5145             nan     0.1000    0.1196
##      3        1.4419             nan     0.1000    0.0974
##      4        1.3818             nan     0.1000    0.0839
##      5        1.3294             nan     0.1000    0.0651
##      6        1.2888             nan     0.1000    0.0690
##      7        1.2470             nan     0.1000    0.0519
##      8        1.2131             nan     0.1000    0.0515
##      9        1.1807             nan     0.1000    0.0502
##     10        1.1495             nan     0.1000    0.0455
##     20        0.9396             nan     0.1000    0.0257
##     40        0.7150             nan     0.1000    0.0113
##     60        0.5893             nan     0.1000    0.0067
##     80        0.5053             nan     0.1000    0.0044
##    100        0.4448             nan     0.1000    0.0023
##    120        0.4003             nan     0.1000    0.0025
##    140        0.3612             nan     0.1000    0.0015
##    150        0.3432             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1945
##      2        1.4865             nan     0.1000    0.1460
##      3        1.3947             nan     0.1000    0.1163
##      4        1.3225             nan     0.1000    0.1098
##      5        1.2569             nan     0.1000    0.0891
##      6        1.2028             nan     0.1000    0.0797
##      7        1.1535             nan     0.1000    0.0619
##      8        1.1150             nan     0.1000    0.0638
##      9        1.0742             nan     0.1000    0.0563
##     10        1.0384             nan     0.1000    0.0421
##     20        0.7998             nan     0.1000    0.0276
##     40        0.5724             nan     0.1000    0.0101
##     60        0.4546             nan     0.1000    0.0066
##     80        0.3784             nan     0.1000    0.0033
##    100        0.3239             nan     0.1000    0.0026
##    120        0.2828             nan     0.1000    0.0008
##    140        0.2498             nan     0.1000    0.0018
##    150        0.2353             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0926
##      2        1.5538             nan     0.1000    0.0683
##      3        1.5115             nan     0.1000    0.0596
##      4        1.4742             nan     0.1000    0.0480
##      5        1.4452             nan     0.1000    0.0446
##      6        1.4167             nan     0.1000    0.0388
##      7        1.3918             nan     0.1000    0.0412
##      8        1.3646             nan     0.1000    0.0380
##      9        1.3405             nan     0.1000    0.0326
##     10        1.3189             nan     0.1000    0.0302
##     20        1.1575             nan     0.1000    0.0213
##     40        0.9539             nan     0.1000    0.0121
##     60        0.8274             nan     0.1000    0.0080
##     80        0.7396             nan     0.1000    0.0059
##    100        0.6714             nan     0.1000    0.0047
##    120        0.6204             nan     0.1000    0.0020
##    140        0.5751             nan     0.1000    0.0010
##    150        0.5585             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1478
##      2        1.5190             nan     0.1000    0.1110
##      3        1.4493             nan     0.1000    0.1026
##      4        1.3864             nan     0.1000    0.0776
##      5        1.3379             nan     0.1000    0.0774
##      6        1.2901             nan     0.1000    0.0775
##      7        1.2426             nan     0.1000    0.0615
##      8        1.2054             nan     0.1000    0.0503
##      9        1.1733             nan     0.1000    0.0542
##     10        1.1392             nan     0.1000    0.0446
##     20        0.9200             nan     0.1000    0.0268
##     40        0.6964             nan     0.1000    0.0127
##     60        0.5715             nan     0.1000    0.0066
##     80        0.4926             nan     0.1000    0.0033
##    100        0.4337             nan     0.1000    0.0030
##    120        0.3871             nan     0.1000    0.0017
##    140        0.3498             nan     0.1000    0.0017
##    150        0.3345             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2041
##      2        1.4831             nan     0.1000    0.1452
##      3        1.3933             nan     0.1000    0.1294
##      4        1.3163             nan     0.1000    0.0969
##      5        1.2543             nan     0.1000    0.0909
##      6        1.1974             nan     0.1000    0.0775
##      7        1.1490             nan     0.1000    0.0766
##      8        1.1021             nan     0.1000    0.0606
##      9        1.0631             nan     0.1000    0.0646
##     10        1.0229             nan     0.1000    0.0506
##     20        0.7782             nan     0.1000    0.0279
##     40        0.5605             nan     0.1000    0.0121
##     60        0.4468             nan     0.1000    0.0052
##     80        0.3761             nan     0.1000    0.0043
##    100        0.3208             nan     0.1000    0.0027
##    120        0.2820             nan     0.1000    0.0018
##    140        0.2499             nan     0.1000    0.0007
##    150        0.2357             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0970
##      2        1.5505             nan     0.1000    0.0675
##      3        1.5084             nan     0.1000    0.0698
##      4        1.4660             nan     0.1000    0.0593
##      5        1.4307             nan     0.1000    0.0499
##      6        1.3996             nan     0.1000    0.0391
##      7        1.3738             nan     0.1000    0.0337
##      8        1.3522             nan     0.1000    0.0359
##      9        1.3291             nan     0.1000    0.0361
##     10        1.3056             nan     0.1000    0.0303
##     20        1.1454             nan     0.1000    0.0214
##     40        0.9543             nan     0.1000    0.0108
##     60        0.8364             nan     0.1000    0.0072
##     80        0.7476             nan     0.1000    0.0041
##    100        0.6839             nan     0.1000    0.0038
##    120        0.6328             nan     0.1000    0.0027
##    140        0.5885             nan     0.1000    0.0018
##    150        0.5701             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1551
##      2        1.5158             nan     0.1000    0.1072
##      3        1.4505             nan     0.1000    0.1103
##      4        1.3837             nan     0.1000    0.0802
##      5        1.3335             nan     0.1000    0.0711
##      6        1.2905             nan     0.1000    0.0651
##      7        1.2501             nan     0.1000    0.0562
##      8        1.2155             nan     0.1000    0.0508
##      9        1.1830             nan     0.1000    0.0459
##     10        1.1540             nan     0.1000    0.0505
##     20        0.9287             nan     0.1000    0.0227
##     40        0.7057             nan     0.1000    0.0132
##     60        0.5788             nan     0.1000    0.0072
##     80        0.4957             nan     0.1000    0.0024
##    100        0.4363             nan     0.1000    0.0030
##    120        0.3911             nan     0.1000    0.0017
##    140        0.3547             nan     0.1000    0.0015
##    150        0.3384             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1967
##      2        1.4903             nan     0.1000    0.1429
##      3        1.4032             nan     0.1000    0.1122
##      4        1.3332             nan     0.1000    0.0980
##      5        1.2710             nan     0.1000    0.0883
##      6        1.2162             nan     0.1000    0.0798
##      7        1.1672             nan     0.1000    0.0745
##      8        1.1211             nan     0.1000    0.0674
##      9        1.0798             nan     0.1000    0.0540
##     10        1.0448             nan     0.1000    0.0523
##     20        0.7987             nan     0.1000    0.0244
##     40        0.5741             nan     0.1000    0.0095
##     60        0.4537             nan     0.1000    0.0049
##     80        0.3765             nan     0.1000    0.0037
##    100        0.3241             nan     0.1000    0.0030
##    120        0.2814             nan     0.1000    0.0017
##    140        0.2477             nan     0.1000    0.0018
##    150        0.2334             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1025
##      2        1.5488             nan     0.1000    0.0733
##      3        1.5046             nan     0.1000    0.0679
##      4        1.4634             nan     0.1000    0.0560
##      5        1.4312             nan     0.1000    0.0478
##      6        1.4014             nan     0.1000    0.0415
##      7        1.3752             nan     0.1000    0.0343
##      8        1.3539             nan     0.1000    0.0323
##      9        1.3336             nan     0.1000    0.0333
##     10        1.3128             nan     0.1000    0.0285
##     20        1.1698             nan     0.1000    0.0173
##     40        0.9924             nan     0.1000    0.0104
##     60        0.8728             nan     0.1000    0.0076
##     80        0.7895             nan     0.1000    0.0057
##    100        0.7219             nan     0.1000    0.0040
##    120        0.6689             nan     0.1000    0.0025
##    140        0.6263             nan     0.1000    0.0021
##    150        0.6074             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1524
##      2        1.5163             nan     0.1000    0.1215
##      3        1.4428             nan     0.1000    0.0893
##      4        1.3882             nan     0.1000    0.0795
##      5        1.3392             nan     0.1000    0.0693
##      6        1.2958             nan     0.1000    0.0590
##      7        1.2572             nan     0.1000    0.0546
##      8        1.2223             nan     0.1000    0.0499
##      9        1.1898             nan     0.1000    0.0416
##     10        1.1633             nan     0.1000    0.0395
##     20        0.9596             nan     0.1000    0.0222
##     40        0.7430             nan     0.1000    0.0123
##     60        0.6164             nan     0.1000    0.0045
##     80        0.5310             nan     0.1000    0.0050
##    100        0.4661             nan     0.1000    0.0026
##    120        0.4167             nan     0.1000    0.0024
##    140        0.3784             nan     0.1000    0.0020
##    150        0.3625             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1926
##      2        1.4899             nan     0.1000    0.1416
##      3        1.4024             nan     0.1000    0.1202
##      4        1.3296             nan     0.1000    0.0972
##      5        1.2716             nan     0.1000    0.0918
##      6        1.2171             nan     0.1000    0.0747
##      7        1.1723             nan     0.1000    0.0631
##      8        1.1332             nan     0.1000    0.0580
##      9        1.0972             nan     0.1000    0.0528
##     10        1.0634             nan     0.1000    0.0489
##     20        0.8273             nan     0.1000    0.0262
##     40        0.5987             nan     0.1000    0.0127
##     60        0.4730             nan     0.1000    0.0061
##     80        0.3970             nan     0.1000    0.0047
##    100        0.3393             nan     0.1000    0.0034
##    120        0.2967             nan     0.1000    0.0019
##    140        0.2636             nan     0.1000    0.0015
##    150        0.2477             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0982
##      2        1.5524             nan     0.1000    0.0739
##      3        1.5081             nan     0.1000    0.0559
##      4        1.4734             nan     0.1000    0.0566
##      5        1.4391             nan     0.1000    0.0476
##      6        1.4087             nan     0.1000    0.0384
##      7        1.3838             nan     0.1000    0.0358
##      8        1.3603             nan     0.1000    0.0363
##      9        1.3371             nan     0.1000    0.0320
##     10        1.3172             nan     0.1000    0.0266
##     20        1.1683             nan     0.1000    0.0180
##     40        0.9811             nan     0.1000    0.0100
##     60        0.8600             nan     0.1000    0.0058
##     80        0.7691             nan     0.1000    0.0053
##    100        0.7025             nan     0.1000    0.0044
##    120        0.6504             nan     0.1000    0.0030
##    140        0.6065             nan     0.1000    0.0022
##    150        0.5869             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1450
##      2        1.5205             nan     0.1000    0.1160
##      3        1.4490             nan     0.1000    0.1059
##      4        1.3864             nan     0.1000    0.0766
##      5        1.3377             nan     0.1000    0.0701
##      6        1.2950             nan     0.1000    0.0626
##      7        1.2562             nan     0.1000    0.0527
##      8        1.2239             nan     0.1000    0.0546
##      9        1.1905             nan     0.1000    0.0429
##     10        1.1620             nan     0.1000    0.0447
##     20        0.9538             nan     0.1000    0.0204
##     40        0.7308             nan     0.1000    0.0118
##     60        0.6014             nan     0.1000    0.0057
##     80        0.5124             nan     0.1000    0.0055
##    100        0.4508             nan     0.1000    0.0030
##    120        0.4030             nan     0.1000    0.0015
##    140        0.3625             nan     0.1000    0.0015
##    150        0.3462             nan     0.1000    0.0024
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1989
##      2        1.4875             nan     0.1000    0.1399
##      3        1.4019             nan     0.1000    0.1202
##      4        1.3287             nan     0.1000    0.1053
##      5        1.2651             nan     0.1000    0.0816
##      6        1.2137             nan     0.1000    0.0748
##      7        1.1678             nan     0.1000    0.0691
##      8        1.1260             nan     0.1000    0.0589
##      9        1.0893             nan     0.1000    0.0544
##     10        1.0532             nan     0.1000    0.0522
##     20        0.8100             nan     0.1000    0.0248
##     40        0.5775             nan     0.1000    0.0104
##     60        0.4645             nan     0.1000    0.0071
##     80        0.3852             nan     0.1000    0.0029
##    100        0.3291             nan     0.1000    0.0034
##    120        0.2857             nan     0.1000    0.0017
##    140        0.2516             nan     0.1000    0.0011
##    150        0.2381             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0829
##      2        1.5584             nan     0.1000    0.0647
##      3        1.5190             nan     0.1000    0.0566
##      4        1.4832             nan     0.1000    0.0527
##      5        1.4510             nan     0.1000    0.0357
##      6        1.4268             nan     0.1000    0.0373
##      7        1.4032             nan     0.1000    0.0345
##      8        1.3810             nan     0.1000    0.0373
##      9        1.3575             nan     0.1000    0.0290
##     10        1.3374             nan     0.1000    0.0319
##     20        1.1848             nan     0.1000    0.0220
##     40        0.9892             nan     0.1000    0.0111
##     60        0.8650             nan     0.1000    0.0073
##     80        0.7770             nan     0.1000    0.0062
##    100        0.7097             nan     0.1000    0.0027
##    120        0.6562             nan     0.1000    0.0011
##    140        0.6143             nan     0.1000    0.0026
##    150        0.5947             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1470
##      2        1.5204             nan     0.1000    0.1076
##      3        1.4544             nan     0.1000    0.0975
##      4        1.3944             nan     0.1000    0.0767
##      5        1.3463             nan     0.1000    0.0700
##      6        1.3026             nan     0.1000    0.0624
##      7        1.2645             nan     0.1000    0.0532
##      8        1.2302             nan     0.1000    0.0591
##      9        1.1935             nan     0.1000    0.0429
##     10        1.1653             nan     0.1000    0.0433
##     20        0.9583             nan     0.1000    0.0266
##     40        0.7287             nan     0.1000    0.0095
##     60        0.6039             nan     0.1000    0.0058
##     80        0.5178             nan     0.1000    0.0041
##    100        0.4561             nan     0.1000    0.0029
##    120        0.4096             nan     0.1000    0.0019
##    140        0.3720             nan     0.1000    0.0015
##    150        0.3557             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1914
##      2        1.4912             nan     0.1000    0.1523
##      3        1.3986             nan     0.1000    0.1096
##      4        1.3284             nan     0.1000    0.1031
##      5        1.2654             nan     0.1000    0.0844
##      6        1.2131             nan     0.1000    0.0742
##      7        1.1659             nan     0.1000    0.0655
##      8        1.1253             nan     0.1000    0.0605
##      9        1.0877             nan     0.1000    0.0586
##     10        1.0520             nan     0.1000    0.0502
##     20        0.8158             nan     0.1000    0.0217
##     40        0.5861             nan     0.1000    0.0106
##     60        0.4679             nan     0.1000    0.0061
##     80        0.3905             nan     0.1000    0.0045
##    100        0.3346             nan     0.1000    0.0018
##    120        0.2909             nan     0.1000    0.0016
##    140        0.2579             nan     0.1000    0.0016
##    150        0.2440             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0870
##      2        1.5549             nan     0.1000    0.0663
##      3        1.5139             nan     0.1000    0.0528
##      4        1.4816             nan     0.1000    0.0465
##      5        1.4528             nan     0.1000    0.0430
##      6        1.4247             nan     0.1000    0.0385
##      7        1.4010             nan     0.1000    0.0378
##      8        1.3777             nan     0.1000    0.0314
##      9        1.3582             nan     0.1000    0.0302
##     10        1.3393             nan     0.1000    0.0297
##     20        1.1917             nan     0.1000    0.0175
##     40        1.0055             nan     0.1000    0.0117
##     60        0.8825             nan     0.1000    0.0070
##     80        0.7922             nan     0.1000    0.0043
##    100        0.7225             nan     0.1000    0.0036
##    120        0.6673             nan     0.1000    0.0029
##    140        0.6225             nan     0.1000    0.0019
##    150        0.6031             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1425
##      2        1.5222             nan     0.1000    0.1186
##      3        1.4506             nan     0.1000    0.0869
##      4        1.3975             nan     0.1000    0.0835
##      5        1.3477             nan     0.1000    0.0629
##      6        1.3071             nan     0.1000    0.0602
##      7        1.2692             nan     0.1000    0.0509
##      8        1.2372             nan     0.1000    0.0512
##      9        1.2045             nan     0.1000    0.0445
##     10        1.1765             nan     0.1000    0.0393
##     20        0.9632             nan     0.1000    0.0273
##     40        0.7402             nan     0.1000    0.0109
##     60        0.6094             nan     0.1000    0.0064
##     80        0.5200             nan     0.1000    0.0050
##    100        0.4556             nan     0.1000    0.0030
##    120        0.4050             nan     0.1000    0.0026
##    140        0.3679             nan     0.1000    0.0018
##    150        0.3503             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1979
##      2        1.4886             nan     0.1000    0.1386
##      3        1.4034             nan     0.1000    0.1157
##      4        1.3318             nan     0.1000    0.0911
##      5        1.2749             nan     0.1000    0.0839
##      6        1.2217             nan     0.1000    0.0771
##      7        1.1743             nan     0.1000    0.0672
##      8        1.1328             nan     0.1000    0.0601
##      9        1.0946             nan     0.1000    0.0501
##     10        1.0620             nan     0.1000    0.0514
##     20        0.8235             nan     0.1000    0.0225
##     40        0.5961             nan     0.1000    0.0094
##     60        0.4707             nan     0.1000    0.0045
##     80        0.3923             nan     0.1000    0.0061
##    100        0.3352             nan     0.1000    0.0029
##    120        0.2928             nan     0.1000    0.0029
##    140        0.2593             nan     0.1000    0.0012
##    150        0.2457             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0972
##      2        1.5500             nan     0.1000    0.0792
##      3        1.5023             nan     0.1000    0.0611
##      4        1.4644             nan     0.1000    0.0525
##      5        1.4325             nan     0.1000    0.0444
##      6        1.4049             nan     0.1000    0.0375
##      7        1.3803             nan     0.1000    0.0353
##      8        1.3579             nan     0.1000    0.0367
##      9        1.3356             nan     0.1000    0.0298
##     10        1.3173             nan     0.1000    0.0268
##     20        1.1634             nan     0.1000    0.0184
##     40        0.9724             nan     0.1000    0.0103
##     60        0.8472             nan     0.1000    0.0068
##     80        0.7598             nan     0.1000    0.0048
##    100        0.6911             nan     0.1000    0.0043
##    120        0.6375             nan     0.1000    0.0035
##    140        0.5941             nan     0.1000    0.0032
##    150        0.5750             nan     0.1000    0.0025
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1513
##      2        1.5170             nan     0.1000    0.1114
##      3        1.4482             nan     0.1000    0.1054
##      4        1.3838             nan     0.1000    0.0854
##      5        1.3328             nan     0.1000    0.0679
##      6        1.2908             nan     0.1000    0.0666
##      7        1.2498             nan     0.1000    0.0577
##      8        1.2142             nan     0.1000    0.0530
##      9        1.1812             nan     0.1000    0.0432
##     10        1.1534             nan     0.1000    0.0432
##     20        0.9412             nan     0.1000    0.0263
##     40        0.7104             nan     0.1000    0.0105
##     60        0.5868             nan     0.1000    0.0058
##     80        0.5047             nan     0.1000    0.0038
##    100        0.4423             nan     0.1000    0.0037
##    120        0.3951             nan     0.1000    0.0021
##    140        0.3557             nan     0.1000    0.0012
##    150        0.3412             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1963
##      2        1.4878             nan     0.1000    0.1570
##      3        1.3902             nan     0.1000    0.1306
##      4        1.3110             nan     0.1000    0.0966
##      5        1.2519             nan     0.1000    0.0842
##      6        1.1987             nan     0.1000    0.0735
##      7        1.1527             nan     0.1000    0.0658
##      8        1.1122             nan     0.1000    0.0666
##      9        1.0718             nan     0.1000    0.0580
##     10        1.0350             nan     0.1000    0.0495
##     20        0.7984             nan     0.1000    0.0249
##     40        0.5707             nan     0.1000    0.0100
##     60        0.4564             nan     0.1000    0.0058
##     80        0.3813             nan     0.1000    0.0031
##    100        0.3255             nan     0.1000    0.0024
##    120        0.2840             nan     0.1000    0.0020
##    140        0.2509             nan     0.1000    0.0018
##    150        0.2367             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1034
##      2        1.5483             nan     0.1000    0.0692
##      3        1.5054             nan     0.1000    0.0543
##      4        1.4713             nan     0.1000    0.0569
##      5        1.4351             nan     0.1000    0.0485
##      6        1.4046             nan     0.1000    0.0400
##      7        1.3788             nan     0.1000    0.0328
##      8        1.3575             nan     0.1000    0.0348
##      9        1.3345             nan     0.1000    0.0285
##     10        1.3161             nan     0.1000    0.0308
##     20        1.1623             nan     0.1000    0.0151
##     40        0.9677             nan     0.1000    0.0115
##     60        0.8441             nan     0.1000    0.0073
##     80        0.7537             nan     0.1000    0.0055
##    100        0.6863             nan     0.1000    0.0034
##    120        0.6331             nan     0.1000    0.0031
##    140        0.5890             nan     0.1000    0.0017
##    150        0.5695             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1560
##      2        1.5140             nan     0.1000    0.1091
##      3        1.4445             nan     0.1000    0.1003
##      4        1.3810             nan     0.1000    0.0755
##      5        1.3333             nan     0.1000    0.0757
##      6        1.2866             nan     0.1000    0.0639
##      7        1.2469             nan     0.1000    0.0568
##      8        1.2101             nan     0.1000    0.0514
##      9        1.1775             nan     0.1000    0.0456
##     10        1.1475             nan     0.1000    0.0391
##     20        0.9356             nan     0.1000    0.0268
##     40        0.7072             nan     0.1000    0.0121
##     60        0.5798             nan     0.1000    0.0072
##     80        0.4959             nan     0.1000    0.0057
##    100        0.4371             nan     0.1000    0.0037
##    120        0.3900             nan     0.1000    0.0027
##    140        0.3532             nan     0.1000    0.0012
##    150        0.3371             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1981
##      2        1.4841             nan     0.1000    0.1460
##      3        1.3912             nan     0.1000    0.1303
##      4        1.3113             nan     0.1000    0.0973
##      5        1.2506             nan     0.1000    0.0850
##      6        1.1980             nan     0.1000    0.0810
##      7        1.1481             nan     0.1000    0.0687
##      8        1.1052             nan     0.1000    0.0646
##      9        1.0649             nan     0.1000    0.0620
##     10        1.0260             nan     0.1000    0.0464
##     20        0.7958             nan     0.1000    0.0239
##     40        0.5642             nan     0.1000    0.0106
##     60        0.4504             nan     0.1000    0.0055
##     80        0.3754             nan     0.1000    0.0040
##    100        0.3239             nan     0.1000    0.0035
##    120        0.2805             nan     0.1000    0.0013
##    140        0.2488             nan     0.1000    0.0014
##    150        0.2349             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1012
##      2        1.5492             nan     0.1000    0.0747
##      3        1.5047             nan     0.1000    0.0636
##      4        1.4663             nan     0.1000    0.0520
##      5        1.4337             nan     0.1000    0.0438
##      6        1.4061             nan     0.1000    0.0386
##      7        1.3824             nan     0.1000    0.0374
##      8        1.3590             nan     0.1000    0.0333
##      9        1.3370             nan     0.1000    0.0346
##     10        1.3150             nan     0.1000    0.0299
##     20        1.1564             nan     0.1000    0.0176
##     40        0.9614             nan     0.1000    0.0112
##     60        0.8335             nan     0.1000    0.0076
##     80        0.7447             nan     0.1000    0.0051
##    100        0.6762             nan     0.1000    0.0035
##    120        0.6228             nan     0.1000    0.0028
##    140        0.5822             nan     0.1000    0.0019
##    150        0.5644             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1529
##      2        1.5177             nan     0.1000    0.1234
##      3        1.4420             nan     0.1000    0.0936
##      4        1.3824             nan     0.1000    0.0745
##      5        1.3355             nan     0.1000    0.0702
##      6        1.2907             nan     0.1000    0.0662
##      7        1.2482             nan     0.1000    0.0592
##      8        1.2119             nan     0.1000    0.0553
##      9        1.1767             nan     0.1000    0.0523
##     10        1.1444             nan     0.1000    0.0479
##     20        0.9297             nan     0.1000    0.0256
##     40        0.6964             nan     0.1000    0.0117
##     60        0.5767             nan     0.1000    0.0074
##     80        0.4980             nan     0.1000    0.0022
##    100        0.4394             nan     0.1000    0.0015
##    120        0.3937             nan     0.1000    0.0021
##    140        0.3585             nan     0.1000    0.0017
##    150        0.3428             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1965
##      2        1.4878             nan     0.1000    0.1571
##      3        1.3919             nan     0.1000    0.1189
##      4        1.3187             nan     0.1000    0.1039
##      5        1.2539             nan     0.1000    0.0871
##      6        1.1991             nan     0.1000    0.0757
##      7        1.1517             nan     0.1000    0.0694
##      8        1.1078             nan     0.1000    0.0586
##      9        1.0714             nan     0.1000    0.0576
##     10        1.0353             nan     0.1000    0.0536
##     20        0.7954             nan     0.1000    0.0261
##     40        0.5633             nan     0.1000    0.0087
##     60        0.4507             nan     0.1000    0.0048
##     80        0.3775             nan     0.1000    0.0034
##    100        0.3245             nan     0.1000    0.0015
##    120        0.2859             nan     0.1000    0.0020
##    140        0.2538             nan     0.1000    0.0017
##    150        0.2404             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1029
##      2        1.5456             nan     0.1000    0.0751
##      3        1.4987             nan     0.1000    0.0661
##      4        1.4572             nan     0.1000    0.0491
##      5        1.4265             nan     0.1000    0.0471
##      6        1.3970             nan     0.1000    0.0405
##      7        1.3710             nan     0.1000    0.0371
##      8        1.3473             nan     0.1000    0.0338
##      9        1.3254             nan     0.1000    0.0302
##     10        1.3050             nan     0.1000    0.0258
##     20        1.1479             nan     0.1000    0.0193
##     40        0.9587             nan     0.1000    0.0091
##     60        0.8418             nan     0.1000    0.0058
##     80        0.7565             nan     0.1000    0.0038
##    100        0.6911             nan     0.1000    0.0031
##    120        0.6405             nan     0.1000    0.0027
##    140        0.5960             nan     0.1000    0.0029
##    150        0.5761             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1598
##      2        1.5101             nan     0.1000    0.1107
##      3        1.4415             nan     0.1000    0.1018
##      4        1.3779             nan     0.1000    0.0768
##      5        1.3289             nan     0.1000    0.0655
##      6        1.2880             nan     0.1000    0.0684
##      7        1.2450             nan     0.1000    0.0530
##      8        1.2119             nan     0.1000    0.0527
##      9        1.1791             nan     0.1000    0.0498
##     10        1.1476             nan     0.1000    0.0456
##     20        0.9344             nan     0.1000    0.0219
##     40        0.7117             nan     0.1000    0.0113
##     60        0.5876             nan     0.1000    0.0046
##     80        0.5035             nan     0.1000    0.0045
##    100        0.4420             nan     0.1000    0.0030
##    120        0.3964             nan     0.1000    0.0026
##    140        0.3567             nan     0.1000    0.0013
##    150        0.3420             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2012
##      2        1.4859             nan     0.1000    0.1525
##      3        1.3933             nan     0.1000    0.1182
##      4        1.3203             nan     0.1000    0.1019
##      5        1.2562             nan     0.1000    0.0890
##      6        1.1997             nan     0.1000    0.0717
##      7        1.1536             nan     0.1000    0.0675
##      8        1.1101             nan     0.1000    0.0555
##      9        1.0741             nan     0.1000    0.0492
##     10        1.0419             nan     0.1000    0.0556
##     20        0.7965             nan     0.1000    0.0255
##     40        0.5719             nan     0.1000    0.0109
##     60        0.4557             nan     0.1000    0.0046
##     80        0.3788             nan     0.1000    0.0048
##    100        0.3253             nan     0.1000    0.0025
##    120        0.2818             nan     0.1000    0.0024
##    140        0.2486             nan     0.1000    0.0017
##    150        0.2349             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0931
##      2        1.5529             nan     0.1000    0.0726
##      3        1.5086             nan     0.1000    0.0527
##      4        1.4760             nan     0.1000    0.0551
##      5        1.4423             nan     0.1000    0.0467
##      6        1.4141             nan     0.1000    0.0412
##      7        1.3885             nan     0.1000    0.0369
##      8        1.3652             nan     0.1000    0.0366
##      9        1.3419             nan     0.1000    0.0284
##     10        1.3232             nan     0.1000    0.0278
##     20        1.1707             nan     0.1000    0.0180
##     40        0.9824             nan     0.1000    0.0098
##     60        0.8594             nan     0.1000    0.0067
##     80        0.7723             nan     0.1000    0.0048
##    100        0.7070             nan     0.1000    0.0041
##    120        0.6518             nan     0.1000    0.0032
##    140        0.6084             nan     0.1000    0.0035
##    150        0.5877             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1523
##      2        1.5172             nan     0.1000    0.1242
##      3        1.4421             nan     0.1000    0.0958
##      4        1.3828             nan     0.1000    0.0773
##      5        1.3356             nan     0.1000    0.0634
##      6        1.2959             nan     0.1000    0.0615
##      7        1.2561             nan     0.1000    0.0596
##      8        1.2197             nan     0.1000    0.0513
##      9        1.1863             nan     0.1000    0.0452
##     10        1.1566             nan     0.1000    0.0401
##     20        0.9526             nan     0.1000    0.0236
##     40        0.7246             nan     0.1000    0.0094
##     60        0.5968             nan     0.1000    0.0060
##     80        0.5134             nan     0.1000    0.0044
##    100        0.4499             nan     0.1000    0.0036
##    120        0.4006             nan     0.1000    0.0024
##    140        0.3603             nan     0.1000    0.0015
##    150        0.3436             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1957
##      2        1.4911             nan     0.1000    0.1500
##      3        1.3992             nan     0.1000    0.1148
##      4        1.3285             nan     0.1000    0.1024
##      5        1.2649             nan     0.1000    0.0923
##      6        1.2077             nan     0.1000    0.0764
##      7        1.1594             nan     0.1000    0.0707
##      8        1.1156             nan     0.1000    0.0611
##      9        1.0771             nan     0.1000    0.0491
##     10        1.0460             nan     0.1000    0.0481
##     20        0.8100             nan     0.1000    0.0234
##     40        0.5829             nan     0.1000    0.0113
##     60        0.4627             nan     0.1000    0.0049
##     80        0.3846             nan     0.1000    0.0038
##    100        0.3283             nan     0.1000    0.0023
##    120        0.2867             nan     0.1000    0.0022
##    140        0.2537             nan     0.1000    0.0015
##    150        0.2391             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1000
##      2        1.5488             nan     0.1000    0.0777
##      3        1.5009             nan     0.1000    0.0577
##      4        1.4653             nan     0.1000    0.0559
##      5        1.4304             nan     0.1000    0.0420
##      6        1.4028             nan     0.1000    0.0385
##      7        1.3784             nan     0.1000    0.0363
##      8        1.3555             nan     0.1000    0.0313
##      9        1.3352             nan     0.1000    0.0329
##     10        1.3147             nan     0.1000    0.0270
##     20        1.1649             nan     0.1000    0.0167
##     40        0.9780             nan     0.1000    0.0122
##     60        0.8530             nan     0.1000    0.0100
##     80        0.7670             nan     0.1000    0.0049
##    100        0.7011             nan     0.1000    0.0033
##    120        0.6449             nan     0.1000    0.0021
##    140        0.6035             nan     0.1000    0.0023
##    150        0.5836             nan     0.1000    0.0025
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1573
##      2        1.5142             nan     0.1000    0.1220
##      3        1.4394             nan     0.1000    0.0905
##      4        1.3823             nan     0.1000    0.0765
##      5        1.3351             nan     0.1000    0.0715
##      6        1.2901             nan     0.1000    0.0604
##      7        1.2526             nan     0.1000    0.0494
##      8        1.2212             nan     0.1000    0.0497
##      9        1.1899             nan     0.1000    0.0483
##     10        1.1596             nan     0.1000    0.0477
##     20        0.9440             nan     0.1000    0.0254
##     40        0.7227             nan     0.1000    0.0120
##     60        0.5944             nan     0.1000    0.0067
##     80        0.5068             nan     0.1000    0.0047
##    100        0.4444             nan     0.1000    0.0023
##    120        0.3962             nan     0.1000    0.0015
##    140        0.3575             nan     0.1000    0.0016
##    150        0.3407             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1977
##      2        1.4892             nan     0.1000    0.1464
##      3        1.3986             nan     0.1000    0.1248
##      4        1.3224             nan     0.1000    0.1013
##      5        1.2600             nan     0.1000    0.0917
##      6        1.2040             nan     0.1000    0.0701
##      7        1.1600             nan     0.1000    0.0668
##      8        1.1177             nan     0.1000    0.0560
##      9        1.0816             nan     0.1000    0.0546
##     10        1.0472             nan     0.1000    0.0514
##     20        0.8064             nan     0.1000    0.0261
##     40        0.5748             nan     0.1000    0.0121
##     60        0.4565             nan     0.1000    0.0065
##     80        0.3756             nan     0.1000    0.0050
##    100        0.3208             nan     0.1000    0.0017
##    120        0.2804             nan     0.1000    0.0028
##    140        0.2471             nan     0.1000    0.0013
##    150        0.2332             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1050
##      2        1.5458             nan     0.1000    0.0773
##      3        1.4985             nan     0.1000    0.0637
##      4        1.4591             nan     0.1000    0.0521
##      5        1.4267             nan     0.1000    0.0483
##      6        1.3962             nan     0.1000    0.0442
##      7        1.3690             nan     0.1000    0.0398
##      8        1.3437             nan     0.1000    0.0344
##      9        1.3226             nan     0.1000    0.0326
##     10        1.3017             nan     0.1000    0.0266
##     20        1.1455             nan     0.1000    0.0206
##     40        0.9471             nan     0.1000    0.0096
##     60        0.8232             nan     0.1000    0.0059
##     80        0.7330             nan     0.1000    0.0042
##    100        0.6668             nan     0.1000    0.0034
##    120        0.6139             nan     0.1000    0.0025
##    140        0.5720             nan     0.1000    0.0023
##    150        0.5526             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1603
##      2        1.5120             nan     0.1000    0.1136
##      3        1.4389             nan     0.1000    0.0959
##      4        1.3778             nan     0.1000    0.0753
##      5        1.3302             nan     0.1000    0.0813
##      6        1.2817             nan     0.1000    0.0614
##      7        1.2422             nan     0.1000    0.0654
##      8        1.2016             nan     0.1000    0.0559
##      9        1.1674             nan     0.1000    0.0448
##     10        1.1390             nan     0.1000    0.0499
##     20        0.9164             nan     0.1000    0.0255
##     40        0.6937             nan     0.1000    0.0104
##     60        0.5659             nan     0.1000    0.0058
##     80        0.4837             nan     0.1000    0.0045
##    100        0.4254             nan     0.1000    0.0021
##    120        0.3793             nan     0.1000    0.0019
##    140        0.3416             nan     0.1000    0.0015
##    150        0.3268             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2019
##      2        1.4869             nan     0.1000    0.1556
##      3        1.3927             nan     0.1000    0.1263
##      4        1.3159             nan     0.1000    0.0962
##      5        1.2551             nan     0.1000    0.0894
##      6        1.2007             nan     0.1000    0.0867
##      7        1.1491             nan     0.1000    0.0744
##      8        1.1020             nan     0.1000    0.0595
##      9        1.0636             nan     0.1000    0.0554
##     10        1.0279             nan     0.1000    0.0523
##     20        0.7822             nan     0.1000    0.0223
##     40        0.5581             nan     0.1000    0.0096
##     60        0.4412             nan     0.1000    0.0055
##     80        0.3652             nan     0.1000    0.0034
##    100        0.3137             nan     0.1000    0.0033
##    120        0.2725             nan     0.1000    0.0015
##    140        0.2405             nan     0.1000    0.0016
##    150        0.2265             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1023
##      2        1.5467             nan     0.1000    0.0774
##      3        1.4989             nan     0.1000    0.0639
##      4        1.4597             nan     0.1000    0.0514
##      5        1.4274             nan     0.1000    0.0479
##      6        1.3974             nan     0.1000    0.0393
##      7        1.3732             nan     0.1000    0.0362
##      8        1.3504             nan     0.1000    0.0317
##      9        1.3304             nan     0.1000    0.0317
##     10        1.3110             nan     0.1000    0.0294
##     20        1.1570             nan     0.1000    0.0178
##     40        0.9677             nan     0.1000    0.0102
##     60        0.8462             nan     0.1000    0.0075
##     80        0.7559             nan     0.1000    0.0049
##    100        0.6897             nan     0.1000    0.0034
##    120        0.6368             nan     0.1000    0.0028
##    140        0.5936             nan     0.1000    0.0020
##    150        0.5760             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1553
##      2        1.5145             nan     0.1000    0.1247
##      3        1.4393             nan     0.1000    0.0916
##      4        1.3832             nan     0.1000    0.0777
##      5        1.3357             nan     0.1000    0.0698
##      6        1.2919             nan     0.1000    0.0604
##      7        1.2541             nan     0.1000    0.0523
##      8        1.2215             nan     0.1000    0.0508
##      9        1.1895             nan     0.1000    0.0500
##     10        1.1578             nan     0.1000    0.0441
##     20        0.9491             nan     0.1000    0.0245
##     40        0.7190             nan     0.1000    0.0126
##     60        0.5891             nan     0.1000    0.0049
##     80        0.5084             nan     0.1000    0.0035
##    100        0.4458             nan     0.1000    0.0044
##    120        0.3964             nan     0.1000    0.0027
##    140        0.3578             nan     0.1000    0.0016
##    150        0.3410             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2077
##      2        1.4801             nan     0.1000    0.1497
##      3        1.3872             nan     0.1000    0.1151
##      4        1.3160             nan     0.1000    0.0950
##      5        1.2569             nan     0.1000    0.0965
##      6        1.1988             nan     0.1000    0.0706
##      7        1.1553             nan     0.1000    0.0716
##      8        1.1111             nan     0.1000    0.0625
##      9        1.0729             nan     0.1000    0.0546
##     10        1.0385             nan     0.1000    0.0466
##     20        0.8026             nan     0.1000    0.0277
##     40        0.5717             nan     0.1000    0.0089
##     60        0.4570             nan     0.1000    0.0060
##     80        0.3798             nan     0.1000    0.0035
##    100        0.3275             nan     0.1000    0.0022
##    120        0.2854             nan     0.1000    0.0011
##    140        0.2519             nan     0.1000    0.0011
##    150        0.2374             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0975
##      2        1.5482             nan     0.1000    0.0763
##      3        1.5009             nan     0.1000    0.0602
##      4        1.4633             nan     0.1000    0.0516
##      5        1.4299             nan     0.1000    0.0480
##      6        1.3985             nan     0.1000    0.0419
##      7        1.3726             nan     0.1000    0.0357
##      8        1.3498             nan     0.1000    0.0382
##      9        1.3260             nan     0.1000    0.0308
##     10        1.3060             nan     0.1000    0.0290
##     20        1.1483             nan     0.1000    0.0206
##     40        0.9611             nan     0.1000    0.0118
##     60        0.8405             nan     0.1000    0.0068
##     80        0.7545             nan     0.1000    0.0045
##    100        0.6907             nan     0.1000    0.0026
##    120        0.6359             nan     0.1000    0.0024
##    140        0.5912             nan     0.1000    0.0017
##    150        0.5750             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1568
##      2        1.5115             nan     0.1000    0.1237
##      3        1.4345             nan     0.1000    0.1008
##      4        1.3712             nan     0.1000    0.0789
##      5        1.3210             nan     0.1000    0.0725
##      6        1.2761             nan     0.1000    0.0705
##      7        1.2334             nan     0.1000    0.0580
##      8        1.1960             nan     0.1000    0.0456
##      9        1.1667             nan     0.1000    0.0504
##     10        1.1358             nan     0.1000    0.0410
##     20        0.9232             nan     0.1000    0.0211
##     40        0.7030             nan     0.1000    0.0100
##     60        0.5795             nan     0.1000    0.0062
##     80        0.4980             nan     0.1000    0.0031
##    100        0.4377             nan     0.1000    0.0021
##    120        0.3913             nan     0.1000    0.0035
##    140        0.3537             nan     0.1000    0.0010
##    150        0.3393             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2141
##      2        1.4769             nan     0.1000    0.1532
##      3        1.3814             nan     0.1000    0.1227
##      4        1.3061             nan     0.1000    0.0991
##      5        1.2443             nan     0.1000    0.0832
##      6        1.1918             nan     0.1000    0.0785
##      7        1.1431             nan     0.1000    0.0679
##      8        1.1010             nan     0.1000    0.0625
##      9        1.0609             nan     0.1000    0.0555
##     10        1.0261             nan     0.1000    0.0472
##     20        0.7884             nan     0.1000    0.0262
##     40        0.5625             nan     0.1000    0.0086
##     60        0.4501             nan     0.1000    0.0060
##     80        0.3778             nan     0.1000    0.0032
##    100        0.3248             nan     0.1000    0.0020
##    120        0.2825             nan     0.1000    0.0018
##    140        0.2511             nan     0.1000    0.0011
##    150        0.2365             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1019
##      2        1.5495             nan     0.1000    0.0770
##      3        1.5043             nan     0.1000    0.0633
##      4        1.4653             nan     0.1000    0.0468
##      5        1.4355             nan     0.1000    0.0466
##      6        1.4063             nan     0.1000    0.0386
##      7        1.3820             nan     0.1000    0.0345
##      8        1.3600             nan     0.1000    0.0338
##      9        1.3381             nan     0.1000    0.0319
##     10        1.3166             nan     0.1000    0.0313
##     20        1.1612             nan     0.1000    0.0165
##     40        0.9674             nan     0.1000    0.0112
##     60        0.8426             nan     0.1000    0.0065
##     80        0.7540             nan     0.1000    0.0050
##    100        0.6854             nan     0.1000    0.0037
##    120        0.6324             nan     0.1000    0.0027
##    140        0.5905             nan     0.1000    0.0022
##    150        0.5734             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1653
##      2        1.5139             nan     0.1000    0.1233
##      3        1.4385             nan     0.1000    0.0966
##      4        1.3788             nan     0.1000    0.0781
##      5        1.3291             nan     0.1000    0.0678
##      6        1.2881             nan     0.1000    0.0592
##      7        1.2499             nan     0.1000    0.0563
##      8        1.2145             nan     0.1000    0.0512
##      9        1.1821             nan     0.1000    0.0480
##     10        1.1521             nan     0.1000    0.0461
##     20        0.9391             nan     0.1000    0.0229
##     40        0.7076             nan     0.1000    0.0130
##     60        0.5849             nan     0.1000    0.0062
##     80        0.5017             nan     0.1000    0.0019
##    100        0.4425             nan     0.1000    0.0029
##    120        0.3934             nan     0.1000    0.0034
##    140        0.3551             nan     0.1000    0.0019
##    150        0.3394             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2016
##      2        1.4885             nan     0.1000    0.1525
##      3        1.3965             nan     0.1000    0.1145
##      4        1.3270             nan     0.1000    0.0979
##      5        1.2670             nan     0.1000    0.0919
##      6        1.2091             nan     0.1000    0.0702
##      7        1.1638             nan     0.1000    0.0689
##      8        1.1215             nan     0.1000    0.0579
##      9        1.0840             nan     0.1000    0.0472
##     10        1.0527             nan     0.1000    0.0610
##     20        0.7976             nan     0.1000    0.0211
##     40        0.5722             nan     0.1000    0.0112
##     60        0.4570             nan     0.1000    0.0064
##     80        0.3803             nan     0.1000    0.0030
##    100        0.3260             nan     0.1000    0.0023
##    120        0.2845             nan     0.1000    0.0017
##    140        0.2511             nan     0.1000    0.0011
##    150        0.2374             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0935
##      2        1.5527             nan     0.1000    0.0723
##      3        1.5076             nan     0.1000    0.0639
##      4        1.4678             nan     0.1000    0.0523
##      5        1.4339             nan     0.1000    0.0450
##      6        1.4046             nan     0.1000    0.0399
##      7        1.3795             nan     0.1000    0.0358
##      8        1.3559             nan     0.1000    0.0368
##      9        1.3323             nan     0.1000    0.0312
##     10        1.3122             nan     0.1000    0.0296
##     20        1.1575             nan     0.1000    0.0174
##     40        0.9698             nan     0.1000    0.0107
##     60        0.8466             nan     0.1000    0.0074
##     80        0.7580             nan     0.1000    0.0047
##    100        0.6906             nan     0.1000    0.0043
##    120        0.6346             nan     0.1000    0.0028
##    140        0.5921             nan     0.1000    0.0017
##    150        0.5741             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1535
##      2        1.5148             nan     0.1000    0.1094
##      3        1.4473             nan     0.1000    0.0967
##      4        1.3872             nan     0.1000    0.0869
##      5        1.3348             nan     0.1000    0.0645
##      6        1.2936             nan     0.1000    0.0666
##      7        1.2542             nan     0.1000    0.0576
##      8        1.2192             nan     0.1000    0.0518
##      9        1.1872             nan     0.1000    0.0473
##     10        1.1570             nan     0.1000    0.0470
##     20        0.9497             nan     0.1000    0.0262
##     40        0.7199             nan     0.1000    0.0145
##     60        0.5945             nan     0.1000    0.0073
##     80        0.5103             nan     0.1000    0.0065
##    100        0.4483             nan     0.1000    0.0022
##    120        0.4030             nan     0.1000    0.0028
##    140        0.3662             nan     0.1000    0.0010
##    150        0.3497             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1970
##      2        1.4893             nan     0.1000    0.1495
##      3        1.3982             nan     0.1000    0.1147
##      4        1.3254             nan     0.1000    0.0950
##      5        1.2659             nan     0.1000    0.0828
##      6        1.2138             nan     0.1000    0.0728
##      7        1.1674             nan     0.1000    0.0654
##      8        1.1269             nan     0.1000    0.0584
##      9        1.0895             nan     0.1000    0.0556
##     10        1.0535             nan     0.1000    0.0459
##     20        0.8152             nan     0.1000    0.0258
##     40        0.5819             nan     0.1000    0.0091
##     60        0.4639             nan     0.1000    0.0062
##     80        0.3860             nan     0.1000    0.0021
##    100        0.3311             nan     0.1000    0.0020
##    120        0.2915             nan     0.1000    0.0009
##    140        0.2595             nan     0.1000    0.0009
##    150        0.2448             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0899
##      2        1.5562             nan     0.1000    0.0671
##      3        1.5149             nan     0.1000    0.0609
##      4        1.4788             nan     0.1000    0.0478
##      5        1.4496             nan     0.1000    0.0457
##      6        1.4223             nan     0.1000    0.0371
##      7        1.3989             nan     0.1000    0.0369
##      8        1.3756             nan     0.1000    0.0317
##      9        1.3546             nan     0.1000    0.0289
##     10        1.3358             nan     0.1000    0.0239
##     20        1.1893             nan     0.1000    0.0187
##     40        1.0027             nan     0.1000    0.0110
##     60        0.8774             nan     0.1000    0.0073
##     80        0.7883             nan     0.1000    0.0037
##    100        0.7214             nan     0.1000    0.0040
##    120        0.6668             nan     0.1000    0.0033
##    140        0.6218             nan     0.1000    0.0032
##    150        0.6001             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1426
##      2        1.5209             nan     0.1000    0.1234
##      3        1.4472             nan     0.1000    0.0883
##      4        1.3948             nan     0.1000    0.0725
##      5        1.3499             nan     0.1000    0.0687
##      6        1.3078             nan     0.1000    0.0544
##      7        1.2738             nan     0.1000    0.0580
##      8        1.2371             nan     0.1000    0.0498
##      9        1.2053             nan     0.1000    0.0482
##     10        1.1753             nan     0.1000    0.0440
##     20        0.9647             nan     0.1000    0.0296
##     40        0.7340             nan     0.1000    0.0124
##     60        0.6020             nan     0.1000    0.0066
##     80        0.5138             nan     0.1000    0.0050
##    100        0.4485             nan     0.1000    0.0029
##    120        0.4001             nan     0.1000    0.0028
##    140        0.3614             nan     0.1000    0.0019
##    150        0.3450             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1811
##      2        1.4963             nan     0.1000    0.1533
##      3        1.4036             nan     0.1000    0.1125
##      4        1.3354             nan     0.1000    0.0944
##      5        1.2772             nan     0.1000    0.0890
##      6        1.2225             nan     0.1000    0.0751
##      7        1.1765             nan     0.1000    0.0582
##      8        1.1389             nan     0.1000    0.0568
##      9        1.1032             nan     0.1000    0.0545
##     10        1.0684             nan     0.1000    0.0478
##     20        0.8297             nan     0.1000    0.0224
##     40        0.5944             nan     0.1000    0.0112
##     60        0.4684             nan     0.1000    0.0049
##     80        0.3903             nan     0.1000    0.0046
##    100        0.3339             nan     0.1000    0.0039
##    120        0.2890             nan     0.1000    0.0020
##    140        0.2555             nan     0.1000    0.0018
##    150        0.2408             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0946
##      2        1.5534             nan     0.1000    0.0737
##      3        1.5076             nan     0.1000    0.0608
##      4        1.4701             nan     0.1000    0.0477
##      5        1.4402             nan     0.1000    0.0465
##      6        1.4114             nan     0.1000    0.0398
##      7        1.3874             nan     0.1000    0.0380
##      8        1.3628             nan     0.1000    0.0309
##      9        1.3428             nan     0.1000    0.0361
##     10        1.3202             nan     0.1000    0.0263
##     20        1.1733             nan     0.1000    0.0214
##     40        0.9823             nan     0.1000    0.0109
##     60        0.8587             nan     0.1000    0.0068
##     80        0.7703             nan     0.1000    0.0051
##    100        0.7031             nan     0.1000    0.0035
##    120        0.6513             nan     0.1000    0.0027
##    140        0.6083             nan     0.1000    0.0009
##    150        0.5897             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1483
##      2        1.5177             nan     0.1000    0.1178
##      3        1.4437             nan     0.1000    0.1042
##      4        1.3823             nan     0.1000    0.0763
##      5        1.3360             nan     0.1000    0.0757
##      6        1.2897             nan     0.1000    0.0556
##      7        1.2548             nan     0.1000    0.0588
##      8        1.2186             nan     0.1000    0.0544
##      9        1.1852             nan     0.1000    0.0409
##     10        1.1589             nan     0.1000    0.0452
##     20        0.9446             nan     0.1000    0.0241
##     40        0.7133             nan     0.1000    0.0101
##     60        0.5890             nan     0.1000    0.0056
##     80        0.5089             nan     0.1000    0.0047
##    100        0.4474             nan     0.1000    0.0019
##    120        0.4007             nan     0.1000    0.0020
##    140        0.3619             nan     0.1000    0.0021
##    150        0.3453             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2020
##      2        1.4872             nan     0.1000    0.1501
##      3        1.3993             nan     0.1000    0.1288
##      4        1.3221             nan     0.1000    0.1013
##      5        1.2608             nan     0.1000    0.0868
##      6        1.2082             nan     0.1000    0.0760
##      7        1.1611             nan     0.1000    0.0715
##      8        1.1177             nan     0.1000    0.0547
##      9        1.0833             nan     0.1000    0.0531
##     10        1.0494             nan     0.1000    0.0415
##     20        0.8130             nan     0.1000    0.0248
##     40        0.5841             nan     0.1000    0.0093
##     60        0.4656             nan     0.1000    0.0066
##     80        0.3878             nan     0.1000    0.0032
##    100        0.3292             nan     0.1000    0.0024
##    120        0.2887             nan     0.1000    0.0019
##    140        0.2554             nan     0.1000    0.0008
##    150        0.2409             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.0985
##      2        1.5497             nan     0.1000    0.0767
##      3        1.5039             nan     0.1000    0.0615
##      4        1.4683             nan     0.1000    0.0499
##      5        1.4367             nan     0.1000    0.0401
##      6        1.4113             nan     0.1000    0.0398
##      7        1.3871             nan     0.1000    0.0378
##      8        1.3624             nan     0.1000    0.0355
##      9        1.3398             nan     0.1000    0.0308
##     10        1.3199             nan     0.1000    0.0287
##     20        1.1642             nan     0.1000    0.0167
##     40        0.9688             nan     0.1000    0.0083
##     60        0.8485             nan     0.1000    0.0076
##     80        0.7626             nan     0.1000    0.0059
##    100        0.6958             nan     0.1000    0.0029
##    120        0.6442             nan     0.1000    0.0024
##    140        0.6004             nan     0.1000    0.0010
##    150        0.5812             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1441
##      2        1.5193             nan     0.1000    0.1123
##      3        1.4501             nan     0.1000    0.0927
##      4        1.3943             nan     0.1000    0.0775
##      5        1.3454             nan     0.1000    0.0723
##      6        1.3007             nan     0.1000    0.0654
##      7        1.2606             nan     0.1000    0.0570
##      8        1.2250             nan     0.1000    0.0547
##      9        1.1912             nan     0.1000    0.0443
##     10        1.1629             nan     0.1000    0.0484
##     20        0.9476             nan     0.1000    0.0202
##     40        0.7176             nan     0.1000    0.0080
##     60        0.5941             nan     0.1000    0.0072
##     80        0.5088             nan     0.1000    0.0044
##    100        0.4473             nan     0.1000    0.0035
##    120        0.3997             nan     0.1000    0.0023
##    140        0.3608             nan     0.1000    0.0010
##    150        0.3456             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1899
##      2        1.4917             nan     0.1000    0.1469
##      3        1.4023             nan     0.1000    0.1160
##      4        1.3299             nan     0.1000    0.0998
##      5        1.2667             nan     0.1000    0.0827
##      6        1.2155             nan     0.1000    0.0855
##      7        1.1624             nan     0.1000    0.0716
##      8        1.1188             nan     0.1000    0.0545
##      9        1.0839             nan     0.1000    0.0578
##     10        1.0479             nan     0.1000    0.0568
##     20        0.8053             nan     0.1000    0.0209
##     40        0.5789             nan     0.1000    0.0104
##     60        0.4611             nan     0.1000    0.0067
##     80        0.3842             nan     0.1000    0.0053
##    100        0.3280             nan     0.1000    0.0024
##    120        0.2867             nan     0.1000    0.0020
##    140        0.2531             nan     0.1000    0.0015
##    150        0.2395             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1929
##      2        1.4929             nan     0.1000    0.1408
##      3        1.4067             nan     0.1000    0.1208
##      4        1.3324             nan     0.1000    0.0903
##      5        1.2765             nan     0.1000    0.0794
##      6        1.2272             nan     0.1000    0.0716
##      7        1.1818             nan     0.1000    0.0669
##      8        1.1401             nan     0.1000    0.0608
##      9        1.1032             nan     0.1000    0.0542
##     10        1.0698             nan     0.1000    0.0545
##     20        0.8284             nan     0.1000    0.0211
##     40        0.6011             nan     0.1000    0.0089
##     60        0.4811             nan     0.1000    0.0046
##     80        0.4065             nan     0.1000    0.0018
##    100        0.3519             nan     0.1000    0.0027
##    120        0.3096             nan     0.1000    0.0013
##    140        0.2773             nan     0.1000    0.0010
##    150        0.2629             nan     0.1000    0.0007
```

```r
ModelGBM
```

```
## Stochastic Gradient Boosting 
## 
## 11776 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction (79), scaled
##  (79), centered (79) 
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7207601  0.6459016
##   1                  100      0.7786164  0.7199092
##   1                  150      0.8050886  0.7534837
##   2                   50      0.8032795  0.7512339
##   2                  100      0.8512235  0.8118496
##   2                  150      0.8747298  0.8415516
##   3                   50      0.8399651  0.7975849
##   3                  100      0.8826348  0.8515332
##   3                  150      0.9035504  0.8779748
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

Cross-validation & Confusion Matrix:

```r
PredictGBM <- predict(ModelGBM, newdata=SubTest, type="raw")
confusionMatrix(PredictGBM, SubTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2115  151   14    0    0
##          B   66 1270   90   10    0
##          C   49   94 1257  118   18
##          D    1    3    7 1123   61
##          E    1    0    0   35 1363
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9085          
##                  95% CI : (0.9019, 0.9148)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8842          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9476   0.8366   0.9189   0.8733   0.9452
## Specificity            0.9706   0.9738   0.9569   0.9890   0.9944
## Pos Pred Value         0.9276   0.8844   0.8184   0.9397   0.9743
## Neg Pred Value         0.9790   0.9613   0.9824   0.9755   0.9877
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2696   0.1619   0.1602   0.1431   0.1737
## Detection Prevalence   0.2906   0.1830   0.1958   0.1523   0.1783
## Balanced Accuracy      0.9591   0.9052   0.9379   0.9311   0.9698
```


### Method 4: Bagged CART

Modeling:

```r
ModelBag <- train(classe~., data=SubTrain, preProcess=c("pca","scale","center"), method="treebag")
```

```
## Loading required package: ipred
```

```
## Warning: package 'ipred' was built under R version 3.3.2
```

```
## Loading required package: e1071
```

```
## Warning: package 'e1071' was built under R version 3.3.2
```

```r
ModelBag
```

```
## Bagged CART 
## 
## 11776 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction (79), scaled
##  (79), centered (79) 
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.9521397  0.9394538
## 
## 
```

Cross-validation & Confusion Matrix:

```r
PredictBag <- predict(ModelBag, newdata=SubTest, type="raw")
confusionMatrix(PredictBag, SubTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2184   46    7    0    2
##          B   29 1449   43    3    0
##          C   17   21 1303   54    0
##          D    1    1   15 1206   27
##          E    1    1    0   23 1413
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9629         
##                  95% CI : (0.9585, 0.967)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9531         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9785   0.9545   0.9525   0.9378   0.9799
## Specificity            0.9902   0.9881   0.9858   0.9933   0.9961
## Pos Pred Value         0.9754   0.9508   0.9341   0.9648   0.9826
## Neg Pred Value         0.9914   0.9891   0.9899   0.9879   0.9955
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2784   0.1847   0.1661   0.1537   0.1801
## Detection Prevalence   0.2854   0.1942   0.1778   0.1593   0.1833
## Balanced Accuracy      0.9843   0.9713   0.9691   0.9655   0.9880
```


### Method 5: Random Forest

Modeling:

```r
library("randomForest")
```

```
## Warning: package 'randomForest' was built under R version 3.3.2
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
ModelRF <- train(classe~., data=SubTrain, importance=TRUE, do.trace=TRUE, preProcess=c("pca","scale","center"), method="rf", ntree=100)
```

```
## ntree      OOB      1      2      3      4      5
##     1:  10.40%  6.42% 12.42% 14.63% 13.00%  7.72%
##     2:  10.93%  7.19% 13.50% 13.72% 14.26%  8.26%
##     3:  10.07%  6.85% 13.18% 11.70% 12.26%  8.41%
##     4:   9.25%  5.87% 12.79% 11.48% 11.03%  7.22%
##     5:   8.52%  5.29% 12.38% 10.32%  9.84%  6.77%
##     6:   7.84%  4.83% 10.84%  9.90%  9.49%  6.02%
##     7:   7.24%  4.39% 10.32%  9.34%  8.75%  5.17%
##     8:   6.65%  4.37%  8.95%  8.61%  7.53%  5.21%
##     9:   6.08%  3.67%  7.75%  8.17%  6.99%  5.35%
##    10:   5.56%  3.74%  7.55%  6.92%  6.41%  4.30%
##    11:   5.29%  3.17%  7.61%  6.83%  6.12%  4.00%
##    12:   4.67%  2.95%  6.74%  5.83%  5.45%  3.39%
##    13:   4.30%  2.89%  6.42%  5.24%  5.29%  2.55%
##    14:   3.95%  2.47%  5.44%  5.24%  4.86%  2.69%
##    15:   3.74%  2.38%  5.57%  4.70%  4.22%  2.64%
##    16:   3.31%  2.18%  4.54%  4.22%  4.43%  1.99%
##    17:   3.14%  2.09%  4.05%  4.21%  4.11%  1.99%
##    18:   3.02%  1.94%  4.23%  3.97%  4.01%  1.67%
##    19:   2.92%  1.91%  4.27%  3.53%  4.06%  1.53%
##    20:   2.84%  1.62%  3.91%  3.82%  4.16%  1.53%
##    21:   2.68%  1.76%  3.51%  3.58%  3.64%  1.57%
##    22:   2.57%  1.50%  3.38%  3.58%  3.80%  1.39%
##    23:   2.65%  1.62%  3.56%  3.58%  3.64%  1.57%
##    24:   2.53%  1.62%  3.34%  3.19%  3.74%  1.43%
##    25:   2.46%  1.41%  3.29%  3.48%  3.48%  1.39%
##    26:   2.32%  1.38%  3.29%  3.00%  3.27%  1.30%
##    27:   2.21%  1.26%  2.98%  3.19%  3.06%  1.20%
##    28:   2.27%  1.32%  3.02%  3.09%  3.16%  1.39%
##    29:   2.30%  1.44%  3.07%  3.19%  3.16%  1.25%
##    30:   2.19%  1.35%  2.94%  3.00%  3.06%  1.20%
##    31:   2.14%  1.26%  2.76%  2.95%  3.16%  1.20%
##    32:   2.08%  1.35%  2.45%  2.95%  3.00%  1.20%
##    33:   2.03%  1.29%  2.36%  2.95%  2.90%  1.20%
##    34:   2.06%  1.18%  2.62%  3.19%  2.74%  1.20%
##    35:   2.03%  1.21%  2.54%  2.95%  2.90%  1.16%
##    36:   1.95%  1.12%  2.45%  3.00%  2.74%  1.06%
##    37:   2.10%  1.23%  2.58%  3.05%  3.00%  1.25%
##    38:   2.01%  1.15%  2.54%  2.95%  2.90%  1.16%
##    39:   2.00%  1.18%  2.67%  2.90%  2.74%  1.06%
##    40:   2.01%  1.18%  2.76%  2.80%  2.90%  1.02%
##    41:   1.87%  1.12%  2.45%  2.66%  2.64%  1.02%
##    42:   1.94%  1.18%  2.49%  2.61%  3.00%  1.02%
##    43:   1.95%  1.12%  2.49%  2.71%  3.06%  1.02%
##    44:   1.93%  1.12%  2.54%  2.76%  2.95%  0.88%
##    45:   1.89%  1.12%  2.27%  2.80%  3.11%  0.79%
##    46:   1.90%  1.12%  2.36%  2.61%  3.11%  0.93%
##    47:   1.86%  1.09%  2.40%  2.56%  3.11%  0.74%
##    48:   1.89%  1.15%  2.31%  2.66%  3.06%  0.88%
##    49:   1.83%  1.06%  2.36%  2.56%  2.95%  0.79%
##    50:   1.84%  1.21%  2.31%  2.51%  2.95%  0.74%
##    51:   1.83%  1.21%  2.36%  2.42%  2.95%  0.69%
##    52:   1.76%  1.12%  2.31%  2.32%  2.79%  0.74%
##    53:   1.76%  1.09%  2.18%  2.51%  2.79%  0.74%
##    54:   1.76%  1.18%  2.09%  2.32%  3.06%  0.65%
##    55:   1.68%  1.09%  2.09%  2.22%  2.85%  0.65%
##    56:   1.66%  1.15%  2.00%  2.08%  2.90%  0.60%
##    57:   1.72%  1.06%  2.18%  2.37%  3.00%  0.56%
##    58:   1.73%  1.09%  2.36%  2.32%  2.79%  0.60%
##    59:   1.71%  1.03%  2.22%  2.27%  2.90%  0.65%
##    60:   1.66%  1.06%  2.09%  2.22%  2.85%  0.60%
##    61:   1.72%  1.06%  2.31%  2.32%  2.69%  0.69%
##    62:   1.71%  1.06%  2.31%  2.18%  2.74%  0.74%
##    63:   1.67%  1.03%  2.22%  2.18%  2.69%  0.74%
##    64:   1.64%  1.12%  2.18%  2.13%  2.48%  0.69%
##    65:   1.67%  1.06%  2.27%  2.13%  2.58%  0.79%
##    66:   1.66%  1.12%  2.18%  2.18%  2.58%  0.69%
##    67:   1.68%  1.09%  2.27%  2.13%  2.64%  0.74%
##    68:   1.65%  1.12%  2.18%  2.13%  2.53%  0.69%
##    69:   1.68%  1.18%  2.22%  2.22%  2.58%  0.60%
##    70:   1.66%  1.12%  2.22%  2.13%  2.53%  0.69%
##    71:   1.66%  1.06%  2.31%  2.03%  2.58%  0.74%
##    72:   1.60%  1.09%  2.14%  1.93%  2.53%  0.74%
##    73:   1.60%  1.00%  2.22%  2.03%  2.48%  0.74%
##    74:   1.61%  1.00%  2.22%  1.98%  2.64%  0.69%
##    75:   1.66%  1.09%  2.40%  1.98%  2.53%  0.69%
##    76:   1.64%  1.00%  2.22%  2.03%  2.74%  0.69%
##    77:   1.62%  1.03%  2.22%  1.98%  2.69%  0.65%
##    78:   1.60%  1.06%  2.09%  2.03%  2.69%  0.60%
##    79:   1.62%  1.09%  2.18%  1.98%  2.58%  0.69%
##    80:   1.58%  1.00%  2.18%  1.98%  2.42%  0.74%
##    81:   1.56%  1.06%  2.09%  1.98%  2.48%  0.60%
##    82:   1.52%  1.03%  2.05%  1.84%  2.42%  0.65%
##    83:   1.58%  1.09%  2.18%  1.98%  2.42%  0.60%
##    84:   1.55%  1.06%  2.14%  1.89%  2.32%  0.74%
##    85:   1.55%  1.03%  2.05%  1.98%  2.53%  0.60%
##    86:   1.55%  1.06%  2.14%  1.89%  2.48%  0.60%
##    87:   1.55%  1.06%  2.14%  1.84%  2.48%  0.60%
##    88:   1.49%  1.03%  2.09%  1.79%  2.32%  0.56%
##    89:   1.53%  1.06%  2.05%  1.89%  2.37%  0.65%
##    90:   1.44%  1.00%  1.87%  1.74%  2.37%  0.56%
##    91:   1.49%  1.09%  1.91%  1.93%  2.27%  0.60%
##    92:   1.48%  1.03%  1.87%  1.89%  2.27%  0.69%
##    93:   1.49%  1.00%  1.87%  1.84%  2.27%  0.83%
##    94:   1.50%  1.00%  2.00%  1.84%  2.27%  0.79%
##    95:   1.49%  0.97%  1.96%  1.89%  2.27%  0.74%
##    96:   1.44%  0.97%  1.87%  1.84%  2.21%  0.69%
##    97:   1.47%  0.94%  1.87%  1.89%  2.32%  0.74%
##    98:   1.47%  1.00%  1.87%  1.84%  2.27%  0.74%
##    99:   1.49%  1.09%  1.91%  1.84%  2.27%  0.69%
##   100:   1.47%  0.97%  1.82%  1.84%  2.37%  0.74%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.58%  4.51%  8.96% 11.40%  9.46%  5.62%
##     2:   7.39%  4.96%  8.72% 10.02%  9.84%  5.29%
##     3:   7.20%  4.75%  8.22% 10.14%  8.99%  5.71%
##     4:   7.11%  4.70%  8.93%  9.90%  8.68%  5.04%
##     5:   6.56%  4.29%  7.95%  8.92%  8.13%  5.10%
##     6:   6.11%  3.81%  7.61%  8.57%  7.32%  4.80%
##     7:   5.67%  3.37%  7.63%  7.99%  6.66%  4.18%
##     8:   5.28%  3.14%  7.13%  7.78%  6.16%  3.55%
##     9:   5.02%  3.15%  6.89%  6.93%  5.65%  3.66%
##    10:   4.87%  2.96%  6.68%  6.66%  5.68%  3.59%
##    11:   4.31%  2.60%  6.00%  5.89%  5.03%  3.11%
##    12:   4.12%  2.80%  5.58%  5.64%  4.76%  2.64%
##    13:   3.87%  2.59%  5.22%  5.33%  4.60%  2.46%
##    14:   3.55%  2.24%  4.86%  4.94%  4.33%  2.22%
##    15:   3.34%  2.44%  4.36%  4.40%  3.96%  2.13%
##    16:   3.25%  2.06%  4.45%  4.26%  4.01%  2.27%
##    17:   3.03%  1.91%  4.09%  3.78%  3.96%  2.17%
##    18:   2.92%  1.94%  4.09%  3.63%  3.64%  1.94%
##    19:   2.75%  1.82%  3.83%  3.58%  3.43%  1.71%
##    20:   2.70%  1.76%  3.56%  3.53%  3.27%  1.99%
##    21:   2.60%  1.85%  3.20%  3.39%  3.32%  1.76%
##    22:   2.65%  1.79%  3.56%  3.43%  3.43%  1.62%
##    23:   2.48%  1.62%  3.29%  3.19%  3.22%  1.67%
##    24:   2.43%  1.65%  3.11%  3.00%  3.32%  1.62%
##    25:   2.42%  1.53%  3.02%  3.14%  3.53%  1.53%
##    26:   2.40%  1.53%  3.29%  3.00%  3.22%  1.57%
##    27:   2.34%  1.38%  3.16%  2.80%  3.27%  1.71%
##    28:   2.34%  1.32%  3.20%  2.95%  3.48%  1.43%
##    29:   2.21%  1.44%  2.98%  2.90%  3.06%  1.20%
##    30:   2.34%  1.50%  3.16%  3.09%  3.32%  1.25%
##    31:   2.26%  1.44%  2.98%  3.00%  3.22%  1.25%
##    32:   2.32%  1.47%  3.11%  2.95%  3.27%  1.39%
##    33:   2.20%  1.26%  2.94%  2.76%  3.32%  1.39%
##    34:   2.22%  1.38%  2.94%  2.76%  3.22%  1.39%
##    35:   2.19%  1.38%  2.85%  2.71%  3.11%  1.48%
##    36:   2.22%  1.29%  3.02%  2.80%  3.27%  1.34%
##    37:   2.17%  1.29%  2.85%  2.71%  3.27%  1.39%
##    38:   2.18%  1.23%  2.89%  2.71%  3.37%  1.39%
##    39:   2.16%  1.18%  2.76%  2.85%  3.22%  1.48%
##    40:   2.16%  1.26%  2.71%  2.80%  3.27%  1.39%
##    41:   2.11%  1.23%  2.67%  2.80%  3.06%  1.39%
##    42:   2.10%  1.32%  2.58%  2.71%  3.06%  1.39%
##    43:   2.11%  1.23%  2.62%  2.66%  3.11%  1.53%
##    44:   2.12%  1.26%  2.54%  2.80%  3.11%  1.53%
##    45:   2.08%  1.15%  2.58%  2.80%  3.06%  1.48%
##    46:   2.10%  1.32%  2.58%  2.66%  3.11%  1.39%
##    47:   2.06%  1.23%  2.62%  2.71%  3.06%  1.30%
##    48:   2.08%  1.18%  2.62%  2.85%  3.11%  1.30%
##    49:   2.04%  1.23%  2.62%  2.61%  2.95%  1.34%
##    50:   2.02%  1.23%  2.67%  2.42%  3.11%  1.25%
##    51:   2.01%  1.21%  2.62%  2.42%  3.16%  1.25%
##    52:   2.04%  1.21%  2.58%  2.51%  3.22%  1.30%
##    53:   2.05%  1.26%  2.62%  2.42%  3.16%  1.34%
##    54:   1.99%  1.21%  2.62%  2.32%  3.11%  1.25%
##    55:   1.99%  1.18%  2.58%  2.32%  3.16%  1.30%
##    56:   1.97%  1.18%  2.40%  2.42%  3.11%  1.34%
##    57:   2.00%  1.18%  2.49%  2.42%  3.16%  1.39%
##    58:   2.02%  1.18%  2.54%  2.51%  3.11%  1.39%
##    59:   1.98%  1.15%  2.45%  2.51%  3.16%  1.25%
##    60:   2.00%  1.23%  2.45%  2.56%  3.11%  1.25%
##    61:   2.00%  1.23%  2.49%  2.47%  3.11%  1.25%
##    62:   2.00%  1.23%  2.45%  2.51%  3.11%  1.25%
##    63:   1.94%  1.26%  2.45%  2.22%  3.00%  1.30%
##    64:   1.98%  1.23%  2.49%  2.42%  3.00%  1.30%
##    65:   1.96%  1.29%  2.36%  2.27%  3.11%  1.30%
##    66:   1.98%  1.23%  2.40%  2.42%  3.00%  1.39%
##    67:   2.04%  1.38%  2.54%  2.42%  3.00%  1.34%
##    68:   1.98%  1.26%  2.45%  2.32%  3.00%  1.39%
##    69:   2.03%  1.38%  2.49%  2.27%  3.06%  1.43%
##    70:   2.01%  1.35%  2.45%  2.37%  3.11%  1.30%
##    71:   1.99%  1.26%  2.36%  2.27%  3.16%  1.43%
##    72:   1.98%  1.35%  2.40%  2.18%  3.16%  1.30%
##    73:   2.00%  1.29%  2.45%  2.32%  3.22%  1.25%
##    74:   1.93%  1.23%  2.27%  2.27%  3.16%  1.25%
##    75:   1.98%  1.29%  2.45%  2.37%  3.00%  1.30%
##    76:   1.94%  1.29%  2.36%  2.27%  3.11%  1.20%
##    77:   1.92%  1.21%  2.27%  2.22%  3.16%  1.30%
##    78:   1.92%  1.23%  2.27%  2.32%  3.06%  1.25%
##    79:   1.92%  1.29%  2.31%  2.22%  3.00%  1.25%
##    80:   1.92%  1.29%  2.27%  2.32%  3.00%  1.20%
##    81:   1.94%  1.35%  2.36%  2.32%  2.95%  1.16%
##    82:   1.89%  1.23%  2.36%  2.22%  2.95%  1.20%
##    83:   1.89%  1.26%  2.27%  2.27%  2.90%  1.20%
##    84:   1.86%  1.18%  2.31%  2.22%  2.95%  1.16%
##    85:   1.86%  1.12%  2.36%  2.22%  3.00%  1.16%
##    86:   1.89%  1.21%  2.45%  2.18%  3.00%  1.11%
##    87:   1.86%  1.15%  2.40%  2.18%  3.00%  1.11%
##    88:   1.88%  1.15%  2.36%  2.18%  3.11%  1.16%
##    89:   1.87%  1.15%  2.36%  2.32%  2.95%  1.11%
##    90:   1.89%  1.18%  2.45%  2.22%  3.00%  1.16%
##    91:   1.87%  1.15%  2.45%  2.13%  3.00%  1.16%
##    92:   1.84%  1.12%  2.36%  2.18%  3.06%  1.06%
##    93:   1.87%  1.18%  2.31%  2.22%  3.06%  1.11%
##    94:   1.89%  1.15%  2.40%  2.32%  2.95%  1.16%
##    95:   1.88%  1.09%  2.45%  2.27%  3.00%  1.16%
##    96:   1.90%  1.26%  2.36%  2.18%  3.11%  1.11%
##    97:   1.93%  1.29%  2.45%  2.18%  3.11%  1.11%
##    98:   1.93%  1.23%  2.45%  2.37%  3.06%  1.06%
##    99:   1.92%  1.21%  2.36%  2.32%  3.16%  1.11%
##   100:   1.89%  1.15%  2.36%  2.32%  3.06%  1.16%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.65%  4.75%  9.61%  9.80% 10.73%  5.41%
##     2:   7.89%  4.66% 10.79%  9.13% 10.51%  6.38%
##     3:   7.47%  4.79%  9.90%  8.99%  9.45%  5.86%
##     4:   7.07%  4.24%  9.51%  9.12%  8.89%  5.41%
##     5:   6.85%  4.05%  9.37%  9.20%  8.61%  4.82%
##     6:   5.97%  3.74%  8.10%  7.46%  7.56%  4.42%
##     7:   5.68%  3.10%  7.79%  7.52%  6.92%  4.68%
##     8:   5.23%  2.93%  7.04%  6.89%  6.18%  4.53%
##     9:   4.73%  2.63%  6.15%  6.23%  5.85%  4.13%
##    10:   4.34%  2.73%  5.57%  5.41%  5.23%  3.79%
##    11:   4.15%  2.48%  5.36%  5.39%  4.98%  3.59%
##    12:   3.76%  2.15%  4.95%  5.04%  4.61%  3.07%
##    13:   3.55%  2.18%  4.14%  4.94%  4.44%  2.97%
##    14:   3.43%  1.94%  4.50%  4.74%  4.32%  2.59%
##    15:   3.43%  2.03%  4.27%  4.69%  4.69%  2.45%
##    16:   3.28%  1.65%  4.14%  4.84%  4.38%  2.50%
##    17:   3.15%  1.68%  3.83%  4.79%  4.38%  2.13%
##    18:   2.91%  1.53%  3.74%  4.16%  4.11%  1.99%
##    19:   2.90%  1.62%  3.43%  4.01%  4.27%  2.08%
##    20:   2.75%  1.47%  3.34%  4.06%  3.80%  1.99%
##    21:   2.78%  1.47%  3.43%  4.01%  4.06%  1.85%
##    22:   2.56%  1.35%  3.51%  3.38%  3.69%  1.67%
##    23:   2.66%  1.44%  3.65%  3.63%  3.90%  1.53%
##    24:   2.64%  1.35%  3.38%  3.63%  4.06%  1.71%
##    25:   2.68%  1.47%  3.34%  3.63%  4.22%  1.67%
##    26:   2.61%  1.41%  3.20%  3.43%  4.06%  1.80%
##    27:   2.45%  1.38%  2.98%  3.00%  3.95%  1.76%
##    28:   2.46%  1.32%  3.38%  3.14%  3.64%  1.62%
##    29:   2.51%  1.23%  3.38%  3.43%  3.85%  1.53%
##    30:   2.45%  1.29%  3.34%  3.38%  3.58%  1.43%
##    31:   2.39%  1.29%  3.34%  3.05%  3.53%  1.48%
##    32:   2.43%  1.44%  3.38%  3.24%  3.37%  1.39%
##    33:   2.42%  1.44%  3.20%  3.19%  3.64%  1.34%
##    34:   2.40%  1.41%  3.16%  3.09%  3.64%  1.43%
##    35:   2.31%  1.32%  3.07%  3.00%  3.64%  1.25%
##    36:   2.37%  1.38%  3.07%  3.09%  3.58%  1.43%
##    37:   2.28%  1.29%  2.85%  3.19%  3.43%  1.34%
##    38:   2.31%  1.32%  2.98%  3.05%  3.48%  1.43%
##    39:   2.28%  1.32%  2.98%  2.85%  3.48%  1.48%
##    40:   2.25%  1.29%  3.07%  2.80%  3.43%  1.34%
##    41:   2.19%  1.29%  2.89%  2.80%  3.27%  1.34%
##    42:   2.20%  1.29%  2.76%  2.71%  3.37%  1.53%
##    43:   2.19%  1.29%  2.71%  2.85%  3.22%  1.53%
##    44:   2.14%  1.29%  2.54%  2.76%  3.32%  1.43%
##    45:   2.11%  1.35%  2.58%  2.66%  3.06%  1.43%
##    46:   2.06%  1.23%  2.67%  2.56%  3.11%  1.34%
##    47:   2.05%  1.26%  2.58%  2.56%  3.06%  1.34%
##    48:   2.04%  1.21%  2.62%  2.56%  3.06%  1.34%
##    49:   2.01%  1.21%  2.71%  2.56%  3.00%  1.16%
##    50:   1.99%  1.21%  2.58%  2.51%  3.11%  1.11%
##    51:   1.99%  1.18%  2.49%  2.61%  3.06%  1.20%
##    52:   2.00%  1.23%  2.49%  2.56%  3.11%  1.20%
##    53:   1.99%  1.18%  2.54%  2.56%  3.00%  1.25%
##    54:   1.96%  1.26%  2.49%  2.47%  2.95%  1.16%
##    55:   1.98%  1.23%  2.58%  2.51%  2.95%  1.16%
##    56:   1.93%  1.21%  2.36%  2.56%  3.00%  1.06%
##    57:   1.92%  1.15%  2.49%  2.51%  3.00%  1.02%
##    58:   1.94%  1.26%  2.49%  2.42%  3.00%  1.02%
##    59:   1.94%  1.26%  2.49%  2.42%  2.90%  1.11%
##    60:   1.92%  1.23%  2.49%  2.42%  2.95%  1.02%
##    61:   1.93%  1.21%  2.54%  2.42%  2.95%  1.06%
##    62:   1.89%  1.21%  2.45%  2.27%  3.00%  1.02%
##    63:   1.88%  1.18%  2.40%  2.32%  2.95%  1.06%
##    64:   1.91%  1.21%  2.54%  2.37%  2.90%  1.06%
##    65:   1.90%  1.23%  2.54%  2.37%  2.85%  1.02%
##    66:   1.94%  1.18%  2.58%  2.47%  2.95%  1.11%
##    67:   1.94%  1.23%  2.58%  2.51%  2.79%  1.11%
##    68:   1.96%  1.21%  2.58%  2.47%  2.95%  1.16%
##    69:   1.98%  1.18%  2.71%  2.42%  2.95%  1.20%
##    70:   1.93%  1.21%  2.45%  2.37%  3.06%  1.11%
##    71:   1.92%  1.18%  2.54%  2.42%  2.95%  1.06%
##    72:   1.94%  1.15%  2.54%  2.37%  3.11%  1.11%
##    73:   1.94%  1.21%  2.49%  2.37%  3.16%  1.02%
##    74:   1.88%  1.12%  2.54%  2.42%  2.85%  1.02%
##    75:   1.90%  1.18%  2.54%  2.27%  3.06%  1.02%
##    76:   1.94%  1.21%  2.58%  2.47%  3.00%  0.97%
##    77:   1.94%  1.21%  2.49%  2.51%  2.95%  1.11%
##    78:   1.92%  1.26%  2.54%  2.32%  2.90%  1.06%
##    79:   1.91%  1.23%  2.67%  2.27%  2.85%  1.02%
##    80:   1.93%  1.21%  2.58%  2.42%  2.85%  1.11%
##    81:   1.86%  1.21%  2.54%  2.32%  2.69%  1.02%
##    82:   1.92%  1.21%  2.58%  2.42%  2.74%  1.16%
##    83:   1.92%  1.23%  2.58%  2.42%  2.79%  1.06%
##    84:   1.92%  1.26%  2.45%  2.47%  2.79%  1.11%
##    85:   1.94%  1.23%  2.67%  2.51%  2.79%  1.02%
##    86:   1.88%  1.21%  2.45%  2.47%  2.79%  0.97%
##    87:   1.86%  1.26%  2.45%  2.37%  2.74%  0.93%
##    88:   1.86%  1.23%  2.45%  2.42%  2.74%  0.93%
##    89:   1.83%  1.23%  2.36%  2.42%  2.64%  0.93%
##    90:   1.85%  1.23%  2.45%  2.42%  2.74%  0.88%
##    91:   1.78%  1.21%  2.40%  2.22%  2.58%  0.93%
##    92:   1.77%  1.18%  2.22%  2.27%  2.69%  0.93%
##    93:   1.79%  1.21%  2.22%  2.32%  2.69%  0.97%
##    94:   1.80%  1.15%  2.27%  2.22%  2.85%  1.02%
##    95:   1.79%  1.15%  2.27%  2.27%  2.74%  1.02%
##    96:   1.76%  1.21%  2.14%  2.27%  2.69%  0.93%
##    97:   1.74%  1.18%  2.14%  2.22%  2.69%  0.93%
##    98:   1.78%  1.23%  2.09%  2.32%  2.74%  0.97%
##    99:   1.80%  1.23%  2.14%  2.27%  2.85%  0.97%
##   100:   1.80%  1.23%  2.14%  2.37%  2.85%  0.88%
## ntree      OOB      1      2      3      4      5
##     1:   9.76%  4.82% 14.66% 15.75%  8.17%  8.75%
##     2:  10.27%  5.26% 14.66% 13.62% 10.86%  9.85%
##     3:   9.69%  5.21% 13.99% 12.61%  9.97%  9.10%
##     4:   9.37%  5.30% 13.03% 12.41%  9.46%  8.91%
##     5:   8.50%  4.76% 11.88% 10.91%  8.93%  8.14%
##     6:   8.02%  4.61% 11.73% 10.20%  8.84%  6.63%
##     7:   7.71%  4.51% 10.76% 10.48%  8.54%  6.06%
##     8:   7.07%  3.90% 10.11%  9.63%  8.40%  5.13%
##     9:   6.29%  3.45%  9.37%  8.05%  7.16%  4.98%
##    10:   5.94%  3.19%  9.28%  7.57%  6.95%  4.18%
##    11:   5.38%  3.11%  7.89%  7.75%  5.78%  3.62%
##    12:   4.91%  3.08%  7.47%  6.31%  5.82%  2.89%
##    13:   4.51%  2.72%  7.29%  5.95%  4.91%  2.65%
##    14:   4.03%  2.51%  6.13%  5.71%  4.25%  2.35%
##    15:   3.78%  2.36%  5.85%  5.11%  4.00%  2.35%
##    16:   3.34%  2.24%  5.50%  4.42%  3.55%  1.53%
##    17:   3.33%  2.06%  4.88%  4.67%  4.05%  1.72%
##    18:   3.11%  1.83%  4.61%  4.37%  3.75%  1.72%
##    19:   3.01%  1.83%  4.39%  4.08%  3.65%  1.77%
##    20:   2.90%  1.83%  4.21%  4.13%  3.59%  1.39%
##    21:   2.76%  1.80%  4.30%  3.88%  3.15%  1.20%
##    22:   2.48%  1.68%  3.41%  3.78%  2.80%  1.20%
##    23:   2.38%  1.74%  3.50%  3.09%  2.80%  1.10%
##    24:   2.34%  1.74%  3.46%  3.24%  2.75%  0.86%
##    25:   2.36%  1.80%  3.59%  3.00%  2.75%  0.96%
##    26:   2.23%  1.56%  3.54%  3.00%  2.65%  0.77%
##    27:   2.10%  1.36%  3.19%  2.90%  2.70%  0.77%
##    28:   2.06%  1.27%  3.10%  3.09%  2.60%  0.67%
##    29:   1.97%  1.30%  2.92%  2.75%  2.50%  0.77%
##    30:   2.00%  1.24%  3.01%  2.95%  2.50%  0.72%
##    31:   1.97%  1.15%  3.06%  2.85%  2.30%  0.96%
##    32:   1.78%  1.06%  2.66%  2.65%  2.35%  0.62%
##    33:   1.80%  1.24%  2.70%  2.55%  2.30%  0.53%
##    34:   1.72%  0.91%  2.53%  2.75%  2.15%  0.77%
##    35:   1.73%  1.06%  2.39%  2.60%  2.20%  0.81%
##    36:   1.72%  1.06%  2.53%  2.50%  2.15%  0.77%
##    37:   1.77%  0.97%  2.66%  2.70%  2.15%  0.81%
##    38:   1.71%  1.03%  2.44%  2.50%  2.10%  0.86%
##    39:   1.66%  0.94%  2.61%  2.36%  2.05%  0.77%
##    40:   1.58%  0.85%  2.61%  2.21%  1.90%  0.72%
##    41:   1.60%  0.88%  2.75%  2.21%  1.90%  0.67%
##    42:   1.67%  0.97%  2.61%  2.50%  1.90%  0.77%
##    43:   1.63%  1.06%  2.57%  2.31%  1.90%  0.62%
##    44:   1.64%  1.03%  2.39%  2.46%  1.90%  0.77%
##    45:   1.55%  0.88%  2.30%  2.36%  1.90%  0.72%
##    46:   1.63%  0.88%  2.48%  2.60%  2.00%  0.62%
##    47:   1.64%  1.00%  2.44%  2.46%  1.95%  0.72%
##    48:   1.63%  0.88%  2.53%  2.36%  2.00%  0.81%
##    49:   1.58%  1.03%  2.13%  2.36%  2.10%  0.62%
##    50:   1.52%  0.91%  2.13%  2.16%  2.00%  0.77%
##    51:   1.55%  0.91%  2.22%  2.26%  1.95%  0.77%
##    52:   1.52%  0.94%  2.13%  2.06%  2.00%  0.81%
##    53:   1.48%  0.94%  1.91%  2.11%  2.10%  0.67%
##    54:   1.44%  0.91%  1.91%  2.16%  1.90%  0.67%
##    55:   1.50%  0.91%  1.99%  2.26%  2.10%  0.62%
##    56:   1.38%  0.85%  1.73%  1.96%  2.05%  0.62%
##    57:   1.45%  0.88%  1.99%  1.87%  2.10%  0.77%
##    58:   1.44%  0.94%  2.04%  1.96%  1.95%  0.62%
##    59:   1.39%  0.85%  2.04%  1.92%  1.80%  0.67%
##    60:   1.42%  0.94%  2.04%  1.92%  1.85%  0.62%
##    61:   1.40%  0.97%  1.99%  1.77%  1.80%  0.72%
##    62:   1.40%  0.91%  1.95%  1.92%  1.80%  0.72%
##    63:   1.35%  0.91%  1.86%  1.72%  1.80%  0.72%
##    64:   1.33%  0.91%  1.86%  1.87%  1.65%  0.62%
##    65:   1.32%  0.83%  1.82%  1.87%  1.75%  0.62%
##    66:   1.32%  0.88%  1.86%  1.82%  1.70%  0.62%
##    67:   1.36%  0.91%  1.91%  1.87%  1.70%  0.67%
##    68:   1.29%  0.94%  1.60%  1.77%  1.75%  0.62%
##    69:   1.33%  0.94%  1.82%  1.77%  1.70%  0.67%
##    70:   1.37%  0.97%  1.68%  1.87%  1.80%  0.77%
##    71:   1.37%  0.97%  1.77%  1.82%  1.95%  0.57%
##    72:   1.36%  0.91%  1.64%  1.92%  1.95%  0.67%
##    73:   1.32%  0.91%  1.55%  1.92%  1.90%  0.62%
##    74:   1.31%  0.94%  1.55%  1.92%  1.70%  0.67%
##    75:   1.27%  0.88%  1.60%  1.77%  1.70%  0.62%
##    76:   1.24%  0.85%  1.46%  1.87%  1.60%  0.67%
##    77:   1.21%  0.85%  1.42%  1.82%  1.60%  0.57%
##    78:   1.23%  0.85%  1.51%  1.77%  1.60%  0.67%
##    79:   1.27%  0.83%  1.51%  1.92%  1.70%  0.67%
##    80:   1.29%  0.85%  1.51%  1.87%  1.70%  0.81%
##    81:   1.25%  0.83%  1.46%  1.87%  1.65%  0.72%
##    82:   1.23%  0.85%  1.46%  1.82%  1.65%  0.62%
##    83:   1.25%  0.80%  1.55%  1.87%  1.60%  0.72%
##    84:   1.26%  0.85%  1.46%  1.87%  1.70%  0.67%
##    85:   1.27%  0.88%  1.51%  2.01%  1.55%  0.67%
##    86:   1.23%  0.85%  1.42%  1.82%  1.70%  0.62%
##    87:   1.23%  0.83%  1.51%  1.87%  1.60%  0.62%
##    88:   1.27%  0.83%  1.42%  2.06%  1.60%  0.72%
##    89:   1.27%  0.83%  1.42%  2.06%  1.70%  0.62%
##    90:   1.27%  0.85%  1.37%  2.06%  1.75%  0.57%
##    91:   1.27%  0.83%  1.33%  2.06%  1.70%  0.72%
##    92:   1.29%  0.88%  1.37%  2.11%  1.65%  0.72%
##    93:   1.30%  0.94%  1.42%  2.06%  1.65%  0.67%
##    94:   1.32%  0.94%  1.46%  2.16%  1.70%  0.57%
##    95:   1.25%  0.88%  1.37%  2.06%  1.65%  0.53%
##    96:   1.27%  0.88%  1.42%  2.01%  1.65%  0.67%
##    97:   1.27%  0.85%  1.42%  2.06%  1.65%  0.62%
##    98:   1.29%  0.91%  1.42%  2.16%  1.65%  0.57%
##    99:   1.26%  0.88%  1.37%  2.11%  1.60%  0.57%
##   100:   1.26%  0.85%  1.46%  2.06%  1.60%  0.57%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.68%  4.26% 10.11% 10.97%  8.07%  6.69%
##     2:   7.71%  4.06% 10.99%  9.55%  7.84%  8.13%
##     3:   7.95%  4.53% 10.63% 10.97%  8.12%  7.51%
##     4:   7.31%  4.67% 10.02%  9.68%  7.66%  6.03%
##     5:   6.74%  4.37%  8.98%  8.96%  7.27%  5.46%
##     6:   6.50%  4.28%  8.29%  8.69%  7.49%  5.12%
##     7:   5.74%  4.00%  7.58%  7.72%  6.43%  3.99%
##     8:   5.24%  3.42%  6.98%  7.38%  5.79%  3.70%
##     9:   4.82%  3.18%  7.09%  6.13%  5.12%  3.46%
##    10:   4.31%  2.86%  5.92%  5.49%  4.73%  3.34%
##    11:   4.04%  2.35%  5.67%  5.77%  4.50%  2.90%
##    12:   3.89%  2.28%  5.70%  5.12%  4.59%  2.65%
##    13:   3.59%  2.13%  5.11%  4.97%  3.92%  2.64%
##    14:   3.59%  2.13%  5.36%  4.52%  3.81%  2.93%
##    15:   3.28%  1.92%  4.65%  3.88%  4.01%  2.73%
##    16:   3.23%  1.86%  4.52%  3.98%  4.15%  2.44%
##    17:   3.05%  1.80%  4.25%  3.73%  4.05%  2.16%
##    18:   3.07%  1.83%  4.34%  3.54%  4.00%  2.39%
##    19:   2.95%  1.68%  4.30%  3.63%  3.59%  2.25%
##    20:   2.96%  1.80%  4.12%  3.78%  3.54%  2.20%
##    21:   2.76%  1.65%  3.85%  3.34%  3.54%  2.06%
##    22:   2.77%  1.56%  3.81%  3.54%  3.49%  2.16%
##    23:   2.75%  1.59%  3.94%  3.44%  3.44%  2.01%
##    24:   2.68%  1.47%  3.59%  3.68%  3.49%  1.92%
##    25:   2.61%  1.53%  3.50%  3.29%  3.30%  2.06%
##    26:   2.53%  1.44%  3.32%  3.44%  3.10%  2.01%
##    27:   2.53%  1.36%  3.23%  3.63%  3.15%  2.01%
##    28:   2.50%  1.21%  3.46%  3.59%  3.30%  1.72%
##    29:   2.47%  1.21%  3.54%  3.54%  2.95%  1.87%
##    30:   2.32%  1.18%  3.37%  3.14%  2.85%  1.72%
##    31:   2.34%  1.27%  3.15%  3.59%  2.70%  1.68%
##    32:   2.25%  1.21%  3.10%  3.29%  2.65%  1.63%
##    33:   2.29%  1.24%  3.01%  3.34%  2.80%  1.72%
##    34:   2.34%  1.24%  3.23%  3.34%  2.85%  1.72%
##    35:   2.28%  1.21%  3.15%  3.19%  2.85%  1.68%
##    36:   2.27%  1.18%  3.32%  3.14%  2.80%  1.53%
##    37:   2.27%  1.12%  3.10%  3.19%  3.05%  1.58%
##    38:   2.28%  1.12%  3.19%  3.14%  3.05%  1.58%
##    39:   2.18%  1.15%  3.10%  3.00%  2.80%  1.48%
##    40:   2.23%  1.18%  3.01%  3.05%  3.00%  1.58%
##    41:   2.18%  1.15%  3.10%  2.80%  2.90%  1.58%
##    42:   2.18%  1.12%  3.15%  2.80%  3.00%  1.48%
##    43:   2.23%  1.09%  3.23%  3.05%  3.00%  1.48%
##    44:   2.21%  1.15%  3.10%  3.05%  2.90%  1.48%
##    45:   2.12%  1.15%  3.10%  2.75%  2.80%  1.39%
##    46:   2.07%  1.03%  2.97%  2.80%  2.70%  1.48%
##    47:   2.14%  1.03%  3.06%  2.90%  2.85%  1.53%
##    48:   2.15%  1.00%  3.06%  3.05%  2.90%  1.44%
##    49:   2.08%  1.03%  3.01%  2.90%  2.80%  1.29%
##    50:   2.06%  1.03%  2.84%  2.80%  2.80%  1.48%
##    51:   2.12%  1.06%  2.97%  2.95%  2.90%  1.39%
##    52:   2.08%  1.12%  2.88%  2.80%  2.85%  1.34%
##    53:   2.14%  1.06%  3.06%  3.00%  2.85%  1.39%
##    54:   2.11%  1.09%  2.97%  2.85%  2.90%  1.39%
##    55:   2.17%  1.09%  3.06%  3.00%  3.00%  1.34%
##    56:   2.10%  1.06%  2.88%  2.90%  2.95%  1.34%
##    57:   2.11%  1.03%  2.88%  3.05%  2.95%  1.29%
##    58:   2.07%  1.00%  2.88%  3.00%  2.85%  1.29%
##    59:   2.09%  1.03%  2.88%  3.00%  2.90%  1.29%
##    60:   2.06%  1.00%  2.75%  2.80%  2.90%  1.48%
##    61:   2.09%  1.00%  2.84%  2.95%  3.00%  1.34%
##    62:   2.10%  0.97%  2.88%  3.00%  2.90%  1.44%
##    63:   2.06%  0.94%  2.75%  3.00%  2.90%  1.44%
##    64:   2.10%  1.00%  2.79%  3.09%  2.80%  1.48%
##    65:   2.01%  1.03%  2.75%  2.80%  2.80%  1.29%
##    66:   2.07%  1.03%  2.92%  2.90%  2.85%  1.29%
##    67:   2.11%  1.03%  2.97%  3.00%  2.90%  1.34%
##    68:   2.07%  0.97%  3.06%  2.85%  2.85%  1.29%
##    69:   2.07%  1.03%  3.06%  2.80%  2.80%  1.29%
##    70:   2.05%  0.97%  2.97%  2.90%  2.75%  1.29%
##    71:   2.09%  1.03%  3.01%  2.85%  2.90%  1.29%
##    72:   2.05%  1.00%  2.84%  2.90%  2.85%  1.29%
##    73:   2.09%  1.03%  2.88%  2.95%  3.00%  1.25%
##    74:   2.06%  1.03%  2.84%  2.85%  3.00%  1.20%
##    75:   2.06%  1.00%  2.88%  2.90%  2.85%  1.29%
##    76:   2.04%  1.03%  2.79%  2.85%  2.90%  1.25%
##    77:   2.06%  0.97%  2.84%  2.90%  3.00%  1.29%
##    78:   2.03%  1.00%  2.75%  2.85%  2.90%  1.29%
##    79:   2.05%  0.97%  2.84%  2.80%  3.05%  1.25%
##    80:   2.02%  1.00%  2.70%  2.90%  2.95%  1.20%
##    81:   2.03%  1.00%  2.79%  2.80%  3.00%  1.20%
##    82:   1.99%  1.00%  2.70%  2.80%  2.80%  1.25%
##    83:   1.98%  1.00%  2.66%  2.85%  2.75%  1.25%
##    84:   1.99%  0.97%  2.84%  2.75%  2.75%  1.25%
##    85:   1.98%  1.00%  2.70%  2.85%  2.70%  1.25%
##    86:   1.97%  1.00%  2.66%  2.75%  2.80%  1.25%
##    87:   1.96%  1.00%  2.70%  2.70%  2.70%  1.29%
##    88:   2.00%  1.00%  2.70%  2.75%  2.85%  1.29%
##    89:   1.93%  1.00%  2.61%  2.55%  2.80%  1.25%
##    90:   1.94%  1.00%  2.61%  2.70%  2.75%  1.25%
##    91:   1.89%  0.97%  2.53%  2.65%  2.70%  1.20%
##    92:   1.95%  1.00%  2.57%  2.70%  2.80%  1.29%
##    93:   1.93%  0.97%  2.57%  2.65%  2.75%  1.29%
##    94:   1.94%  1.00%  2.48%  2.70%  2.80%  1.29%
##    95:   1.91%  0.97%  2.57%  2.60%  2.70%  1.29%
##    96:   1.91%  0.97%  2.48%  2.55%  2.80%  1.34%
##    97:   1.88%  0.94%  2.39%  2.55%  2.70%  1.39%
##    98:   1.93%  1.00%  2.57%  2.55%  2.80%  1.29%
##    99:   1.89%  0.91%  2.53%  2.50%  2.80%  1.29%
##   100:   1.88%  0.94%  2.48%  2.50%  2.75%  1.29%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.50%  3.41% 12.20% 10.18%  7.84%  6.23%
##     2:   8.10%  3.71% 12.54% 10.58%  8.82%  7.26%
##     3:   7.75%  3.87% 11.84%  9.94%  8.61%  6.77%
##     4:   7.14%  3.84% 10.38%  9.04%  8.28%  6.12%
##     5:   6.76%  4.09%  9.15%  8.64%  7.85%  5.74%
##     6:   6.07%  3.51%  8.63%  8.23%  6.49%  5.02%
##     7:   5.50%  3.31%  8.06%  6.76%  6.37%  4.24%
##     8:   5.33%  3.17%  7.89%  6.95%  5.70%  4.15%
##     9:   4.75%  2.49%  6.90%  6.04%  5.61%  4.03%
##    10:   4.64%  2.77%  6.81%  5.75%  5.47%  3.48%
##    11:   4.23%  2.55%  6.29%  5.33%  4.99%  2.94%
##    12:   3.98%  2.37%  5.74%  5.37%  4.57%  2.79%
##    13:   3.69%  2.07%  5.42%  4.53%  4.42%  2.93%
##    14:   3.56%  2.33%  5.37%  4.23%  3.96%  2.59%
##    15:   3.14%  1.89%  4.08%  4.03%  3.71%  2.73%
##    16:   3.19%  2.09%  4.03%  4.28%  3.60%  2.59%
##    17:   2.95%  1.83%  3.81%  3.68%  3.80%  2.30%
##    18:   2.82%  1.74%  3.68%  3.63%  3.65%  2.06%
##    19:   2.73%  1.59%  3.41%  3.68%  3.60%  2.06%
##    20:   2.70%  1.53%  3.63%  3.54%  3.74%  1.77%
##    21:   2.64%  1.53%  3.37%  3.44%  3.74%  1.82%
##    22:   2.52%  1.44%  3.50%  3.14%  3.34%  1.82%
##    23:   2.50%  1.44%  3.32%  3.19%  3.39%  1.77%
##    24:   2.51%  1.39%  3.37%  3.24%  3.39%  1.82%
##    25:   2.41%  1.39%  3.06%  3.19%  3.34%  1.72%
##    26:   2.38%  1.36%  3.01%  3.00%  3.39%  1.77%
##    27:   2.34%  1.39%  3.01%  2.95%  3.30%  1.68%
##    28:   2.31%  1.33%  3.28%  2.90%  2.95%  1.68%
##    29:   2.34%  1.33%  3.23%  2.80%  3.20%  1.72%
##    30:   2.39%  1.33%  3.32%  3.05%  3.30%  1.58%
##    31:   2.37%  1.24%  3.32%  3.14%  3.20%  1.63%
##    32:   2.38%  1.33%  3.32%  2.85%  3.20%  1.82%
##    33:   2.31%  1.24%  2.97%  2.95%  3.39%  1.68%
##    34:   2.30%  1.18%  3.23%  3.00%  3.10%  1.68%
##    35:   2.26%  1.27%  3.06%  2.85%  3.15%  1.58%
##    36:   2.21%  1.15%  3.15%  3.05%  2.85%  1.48%
##    37:   2.23%  1.15%  3.32%  3.09%  2.90%  1.34%
##    38:   2.20%  1.12%  3.23%  3.14%  2.80%  1.34%
##    39:   2.13%  1.12%  3.19%  2.90%  2.75%  1.29%
##    40:   2.07%  1.12%  3.01%  3.00%  2.65%  1.15%
##    41:   2.14%  1.12%  3.10%  2.90%  2.95%  1.25%
##    42:   2.07%  1.09%  3.01%  2.70%  2.90%  1.25%
##    43:   2.06%  1.09%  3.01%  2.70%  2.80%  1.25%
##    44:   2.10%  1.12%  2.97%  2.75%  3.00%  1.25%
##    45:   2.13%  1.09%  3.15%  2.90%  2.90%  1.25%
##    46:   2.10%  1.09%  3.06%  2.85%  2.75%  1.34%
##    47:   2.06%  0.97%  3.15%  2.85%  2.80%  1.15%
##    48:   2.09%  1.00%  3.19%  3.05%  2.70%  1.15%
##    49:   2.06%  1.00%  3.10%  3.00%  2.65%  1.20%
##    50:   2.08%  1.06%  3.15%  2.80%  2.85%  1.15%
##    51:   2.06%  1.00%  2.92%  2.80%  2.95%  1.25%
##    52:   1.96%  1.00%  2.88%  2.65%  2.70%  1.15%
##    53:   1.96%  1.06%  2.70%  2.70%  2.80%  1.10%
##    54:   1.94%  1.03%  2.66%  2.65%  2.75%  1.15%
##    55:   1.96%  1.03%  2.92%  2.46%  2.85%  1.10%
##    56:   1.91%  0.97%  2.97%  2.46%  2.85%  0.86%
##    57:   1.98%  1.03%  2.92%  2.65%  2.80%  1.05%
##    58:   1.93%  1.00%  3.01%  2.65%  2.70%  0.81%
##    59:   1.90%  1.00%  2.79%  2.70%  2.65%  0.91%
##    60:   1.92%  0.94%  2.88%  2.70%  2.85%  0.81%
##    61:   1.92%  1.03%  2.79%  2.65%  2.65%  1.01%
##    62:   1.92%  0.97%  2.75%  2.75%  2.75%  0.96%
##    63:   1.86%  0.91%  2.66%  2.60%  2.75%  0.96%
##    64:   1.88%  0.94%  2.75%  2.60%  2.75%  0.91%
##    65:   1.96%  0.94%  2.88%  2.70%  2.90%  1.01%
##    66:   1.89%  0.97%  2.79%  2.55%  2.80%  0.86%
##    67:   1.92%  1.00%  2.84%  2.60%  2.85%  0.86%
##    68:   1.85%  0.91%  2.79%  2.50%  2.80%  0.81%
##    69:   1.89%  0.94%  2.61%  2.60%  2.85%  1.01%
##    70:   1.87%  0.94%  2.66%  2.60%  2.80%  0.91%
##    71:   1.83%  0.88%  2.57%  2.60%  2.75%  0.96%
##    72:   1.88%  0.91%  2.70%  2.60%  2.85%  0.91%
##    73:   1.89%  0.94%  2.66%  2.55%  2.90%  1.01%
##    74:   1.88%  0.97%  2.61%  2.60%  2.80%  0.96%
##    75:   1.91%  0.97%  2.75%  2.65%  2.70%  1.05%
##    76:   1.91%  1.00%  2.61%  2.65%  2.90%  0.96%
##    77:   1.92%  1.00%  2.70%  2.65%  2.80%  1.01%
##    78:   1.85%  0.88%  2.66%  2.55%  2.90%  0.86%
##    79:   1.86%  0.97%  2.61%  2.55%  2.85%  0.86%
##    80:   1.84%  0.88%  2.66%  2.65%  2.80%  0.81%
##    81:   1.89%  0.94%  2.75%  2.65%  2.80%  0.86%
##    82:   1.89%  0.91%  2.75%  2.70%  2.80%  0.91%
##    83:   1.86%  0.91%  2.70%  2.60%  2.70%  0.96%
##    84:   1.89%  0.94%  2.70%  2.60%  2.85%  0.96%
##    85:   1.83%  0.85%  2.66%  2.55%  2.80%  0.91%
##    86:   1.89%  0.88%  2.70%  2.60%  2.75%  1.10%
##    87:   1.84%  0.88%  2.66%  2.50%  2.80%  0.96%
##    88:   1.85%  0.88%  2.66%  2.60%  2.75%  0.96%
##    89:   1.87%  0.94%  2.70%  2.60%  2.60%  1.05%
##    90:   1.87%  0.94%  2.66%  2.60%  2.65%  1.05%
##    91:   1.84%  0.83%  2.61%  2.60%  2.75%  1.05%
##    92:   1.85%  0.85%  2.66%  2.60%  2.75%  1.01%
##    93:   1.87%  0.85%  2.70%  2.60%  2.75%  1.05%
##    94:   1.85%  0.80%  2.75%  2.60%  2.70%  1.05%
##    95:   1.89%  0.88%  2.75%  2.50%  2.75%  1.15%
##    96:   1.90%  0.85%  2.79%  2.55%  2.80%  1.15%
##    97:   1.93%  0.88%  2.88%  2.55%  2.80%  1.15%
##    98:   1.90%  0.91%  2.75%  2.55%  2.75%  1.15%
##    99:   1.93%  0.94%  2.79%  2.55%  2.80%  1.15%
##   100:   1.89%  0.85%  2.79%  2.55%  2.65%  1.20%
## ntree      OOB      1      2      3      4      5
##     1:  11.07%  8.42% 16.22%  9.11% 12.59% 10.37%
##     2:  10.01%  6.76% 14.81%  9.13% 11.39%  9.56%
##     3:  10.13%  6.81% 14.59%  9.65% 11.62%  9.54%
##     4:   9.40%  6.27% 12.89%  9.84% 10.48%  9.07%
##     5:   8.87%  5.81% 11.57%  9.51% 10.69%  8.41%
##     6:   8.15%  5.48% 11.15%  8.85%  9.51%  7.12%
##     7:   7.25%  4.88%  9.42%  8.38%  7.87%  6.89%
##     8:   6.74%  4.59%  9.43%  7.34%  7.58%  5.80%
##     9:   6.00%  3.83%  8.49%  7.00%  6.85%  4.92%
##    10:   5.55%  3.77%  7.91%  6.28%  5.97%  4.66%
##    11:   4.90%  3.20%  7.23%  5.88%  5.22%  3.78%
##    12:   4.48%  2.98%  6.87%  5.33%  4.20%  3.64%
##    13:   4.25%  2.69%  7.07%  4.79%  4.04%  3.27%
##    14:   4.13%  2.75%  6.76%  4.45%  4.56%  2.72%
##    15:   3.86%  2.51%  6.16%  4.44%  3.98%  2.81%
##    16:   3.40%  2.11%  5.80%  3.86%  3.67%  2.13%
##    17:   3.22%  1.89%  5.54%  3.57%  3.40%  2.27%
##    18:   3.17%  2.23%  5.24%  3.18%  3.61%  1.99%
##    19:   3.00%  2.11%  5.11%  3.13%  3.35%  1.68%
##    20:   2.89%  1.83%  4.94%  3.33%  3.25%  1.59%
##    21:   2.73%  1.92%  4.68%  2.84%  3.09%  1.45%
##    22:   2.69%  1.89%  4.85%  2.75%  3.14%  1.18%
##    23:   2.49%  1.74%  4.33%  2.75%  2.83%  1.13%
##    24:   2.40%  1.83%  4.07%  2.31%  2.93%  1.13%
##    25:   2.40%  1.77%  4.07%  2.41%  2.88%  1.18%
##    26:   2.34%  1.65%  4.03%  2.22%  2.93%  1.22%
##    27:   2.28%  1.53%  3.94%  2.22%  3.04%  1.09%
##    28:   2.30%  1.59%  4.16%  2.07%  2.88%  1.13%
##    29:   2.21%  1.44%  3.98%  2.31%  2.62%  1.04%
##    30:   2.11%  1.28%  3.85%  2.17%  2.46%  1.18%
##    31:   2.14%  1.19%  3.77%  2.56%  2.56%  1.09%
##    32:   2.01%  1.10%  3.64%  2.27%  2.56%  0.95%
##    33:   2.06%  1.13%  3.64%  2.46%  2.62%  0.91%
##    34:   2.09%  1.22%  3.64%  2.36%  2.67%  1.00%
##    35:   1.94%  1.04%  3.59%  2.17%  2.35%  0.95%
##    36:   1.94%  1.01%  3.64%  2.07%  2.56%  0.91%
##    37:   1.96%  1.04%  3.68%  2.03%  2.62%  0.91%
##    38:   1.99%  1.19%  3.55%  2.12%  2.51%  0.95%
##    39:   1.83%  0.85%  3.51%  1.93%  2.41%  0.95%
##    40:   1.88%  0.98%  3.42%  2.07%  2.51%  0.86%
##    41:   1.85%  0.95%  3.64%  1.78%  2.41%  0.91%
##    42:   1.81%  0.95%  3.42%  1.88%  2.30%  0.91%
##    43:   1.77%  0.95%  3.33%  1.88%  2.25%  0.82%
##    44:   1.77%  0.92%  3.33%  1.69%  2.46%  0.86%
##    45:   1.75%  0.98%  3.33%  1.69%  2.25%  0.86%
##    46:   1.75%  1.10%  3.25%  1.74%  2.15%  0.82%
##    47:   1.70%  0.89%  3.12%  1.88%  2.15%  0.86%
##    48:   1.82%  0.98%  3.46%  1.98%  2.20%  0.86%
##    49:   1.76%  0.95%  3.38%  1.88%  2.20%  0.77%
##    50:   1.66%  0.85%  3.07%  2.03%  1.99%  0.72%
##    51:   1.68%  0.92%  3.16%  1.93%  1.99%  0.77%
##    52:   1.66%  0.89%  3.07%  1.98%  1.99%  0.72%
##    53:   1.68%  0.85%  3.12%  1.98%  2.20%  0.68%
##    54:   1.72%  0.89%  3.20%  2.07%  2.09%  0.72%
##    55:   1.74%  0.89%  3.33%  2.12%  2.04%  0.72%
##    56:   1.68%  0.89%  3.25%  1.88%  2.04%  0.72%
##    57:   1.66%  0.92%  3.20%  1.93%  1.88%  0.68%
##    58:   1.68%  0.95%  3.25%  1.83%  1.99%  0.72%
##    59:   1.63%  0.85%  2.99%  1.93%  2.04%  0.72%
##    60:   1.67%  0.92%  3.12%  2.03%  1.94%  0.72%
##    61:   1.64%  0.85%  3.03%  1.98%  2.04%  0.68%
##    62:   1.64%  0.85%  2.94%  2.03%  2.09%  0.68%
##    63:   1.65%  0.85%  2.99%  2.03%  2.15%  0.63%
##    64:   1.62%  0.92%  2.94%  1.83%  2.15%  0.63%
##    65:   1.57%  0.85%  2.86%  1.83%  2.04%  0.63%
##    66:   1.60%  0.85%  2.86%  1.98%  1.99%  0.68%
##    67:   1.58%  0.89%  2.86%  1.88%  1.94%  0.68%
##    68:   1.55%  0.82%  3.03%  1.78%  1.88%  0.54%
##    69:   1.55%  0.85%  2.77%  1.88%  1.94%  0.63%
##    70:   1.55%  0.89%  2.77%  1.83%  1.99%  0.59%
##    71:   1.55%  0.89%  2.77%  1.83%  2.04%  0.59%
##    72:   1.52%  0.79%  2.69%  1.88%  1.99%  0.63%
##    73:   1.52%  0.76%  2.73%  1.93%  1.94%  0.63%
##    74:   1.52%  0.82%  2.69%  1.83%  2.04%  0.59%
##    75:   1.52%  0.85%  2.73%  1.93%  1.83%  0.59%
##    76:   1.51%  0.85%  2.73%  1.93%  1.83%  0.54%
##    77:   1.54%  0.82%  2.77%  1.93%  1.99%  0.54%
##    78:   1.52%  0.79%  2.77%  2.03%  1.88%  0.50%
##    79:   1.50%  0.82%  2.82%  1.88%  1.88%  0.45%
##    80:   1.50%  0.79%  2.77%  1.88%  1.94%  0.50%
##    81:   1.49%  0.79%  2.64%  1.88%  1.99%  0.50%
##    82:   1.48%  0.79%  2.60%  2.03%  1.83%  0.50%
##    83:   1.48%  0.76%  2.64%  1.88%  1.99%  0.50%
##    84:   1.49%  0.79%  2.60%  1.98%  1.99%  0.50%
##    85:   1.48%  0.79%  2.56%  1.98%  1.94%  0.50%
##    86:   1.52%  0.85%  2.69%  1.98%  1.94%  0.50%
##    87:   1.51%  0.82%  2.64%  2.03%  1.94%  0.50%
##    88:   1.49%  0.85%  2.69%  2.03%  1.83%  0.41%
##    89:   1.54%  0.85%  2.73%  2.07%  1.94%  0.45%
##    90:   1.45%  0.82%  2.60%  1.83%  1.88%  0.45%
##    91:   1.47%  0.85%  2.69%  1.93%  1.88%  0.32%
##    92:   1.48%  0.85%  2.60%  1.88%  1.94%  0.45%
##    93:   1.45%  0.82%  2.56%  1.88%  1.94%  0.41%
##    94:   1.44%  0.82%  2.60%  1.93%  1.83%  0.36%
##    95:   1.44%  0.82%  2.64%  1.88%  1.83%  0.36%
##    96:   1.46%  0.85%  2.64%  1.93%  1.83%  0.36%
##    97:   1.42%  0.82%  2.60%  1.78%  1.83%  0.36%
##    98:   1.48%  0.85%  2.56%  1.98%  1.94%  0.41%
##    99:   1.43%  0.92%  2.47%  1.78%  1.88%  0.36%
##   100:   1.40%  0.85%  2.38%  1.78%  1.94%  0.36%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   6.70%  5.59%  9.78%  6.83%  7.54%  4.20%
##     2:   7.14%  5.79%  9.04%  9.03%  8.22%  4.44%
##     3:   6.97%  5.51%  9.31%  9.07%  7.35%  4.35%
##     4:   6.39%  4.30%  8.97%  8.49%  6.74%  4.53%
##     5:   6.19%  4.29%  8.20%  7.77%  7.31%  4.45%
##     6:   5.84%  3.94%  7.74%  7.66%  6.53%  4.38%
##     7:   5.18%  3.36%  7.38%  6.39%  6.21%  3.56%
##     8:   4.97%  3.44%  6.70%  6.17%  5.51%  3.84%
##     9:   4.43%  3.01%  6.06%  5.63%  5.36%  2.90%
##    10:   4.13%  2.84%  5.88%  4.96%  4.59%  3.02%
##    11:   3.70%  2.67%  4.97%  4.50%  4.42%  2.51%
##    12:   3.72%  2.79%  4.65%  4.45%  4.62%  2.68%
##    13:   3.40%  2.45%  4.90%  4.25%  3.57%  2.32%
##    14:   3.27%  2.29%  4.60%  3.71%  3.99%  2.27%
##    15:   3.06%  2.26%  4.25%  3.38%  3.67%  2.18%
##    16:   2.92%  2.20%  4.16%  3.04%  3.67%  1.95%
##    17:   2.88%  2.20%  4.16%  2.99%  3.35%  2.04%
##    18:   2.74%  2.02%  3.94%  2.99%  3.35%  1.81%
##    19:   2.67%  1.80%  3.94%  3.04%  3.14%  1.86%
##    20:   2.55%  1.95%  3.64%  2.84%  2.88%  1.72%
##    21:   2.60%  1.83%  3.72%  2.94%  3.09%  1.81%
##    22:   2.54%  1.86%  3.77%  2.70%  2.88%  1.81%
##    23:   2.41%  1.68%  3.51%  2.80%  2.72%  1.72%
##    24:   2.39%  1.65%  3.46%  2.84%  2.83%  1.54%
##    25:   2.21%  1.59%  3.20%  2.60%  2.51%  1.45%
##    26:   2.25%  1.59%  3.25%  2.65%  2.72%  1.40%
##    27:   2.30%  1.74%  3.16%  2.70%  2.77%  1.45%
##    28:   2.14%  1.44%  3.25%  2.46%  2.41%  1.50%
##    29:   2.15%  1.56%  3.12%  2.51%  2.62%  1.27%
##    30:   2.06%  1.44%  2.94%  2.31%  2.62%  1.36%
##    31:   2.11%  1.47%  3.03%  2.36%  2.67%  1.36%
##    32:   2.10%  1.40%  2.86%  2.36%  2.83%  1.45%
##    33:   2.08%  1.47%  2.69%  2.41%  2.93%  1.31%
##    34:   1.94%  1.44%  2.56%  2.07%  2.72%  1.27%
##    35:   2.00%  1.53%  2.69%  2.07%  2.77%  1.22%
##    36:   1.89%  1.37%  2.73%  1.93%  2.83%  0.91%
##    37:   1.93%  1.47%  2.77%  1.88%  2.77%  1.04%
##    38:   1.90%  1.44%  2.77%  1.93%  2.67%  1.00%
##    39:   1.92%  1.40%  2.77%  1.93%  2.83%  1.00%
##    40:   1.89%  1.44%  2.64%  2.03%  2.88%  0.82%
##    41:   1.92%  1.37%  2.73%  2.03%  2.83%  1.00%
##    42:   1.83%  1.31%  2.60%  2.03%  2.67%  0.91%
##    43:   1.87%  1.47%  2.69%  2.12%  2.62%  0.72%
##    44:   1.88%  1.47%  2.73%  2.03%  2.67%  0.77%
##    45:   1.80%  1.37%  2.64%  1.83%  2.56%  0.86%
##    46:   1.85%  1.44%  2.73%  1.98%  2.56%  0.82%
##    47:   1.83%  1.40%  2.73%  2.12%  2.35%  0.82%
##    48:   1.84%  1.44%  2.73%  2.22%  2.30%  0.77%
##    49:   1.86%  1.40%  2.82%  2.07%  2.51%  0.77%
##    50:   1.79%  1.40%  2.60%  2.07%  2.35%  0.77%
##    51:   1.77%  1.40%  2.60%  1.98%  2.35%  0.72%
##    52:   1.77%  1.40%  2.60%  2.12%  2.25%  0.72%
##    53:   1.76%  1.31%  2.60%  2.07%  2.25%  0.82%
##    54:   1.77%  1.37%  2.73%  1.93%  2.25%  0.77%
##    55:   1.74%  1.44%  2.47%  1.88%  2.30%  0.82%
##    56:   1.70%  1.40%  2.51%  1.83%  2.20%  0.72%
##    57:   1.72%  1.37%  2.60%  1.93%  2.04%  0.86%
##    58:   1.77%  1.44%  2.56%  2.07%  2.20%  0.77%
##    59:   1.77%  1.50%  2.56%  2.03%  2.15%  0.77%
##    60:   1.74%  1.44%  2.56%  2.07%  2.09%  0.72%
##    61:   1.68%  1.47%  2.38%  2.12%  1.88%  0.68%
##    62:   1.70%  1.47%  2.64%  1.88%  1.94%  0.68%
##    63:   1.74%  1.37%  2.64%  2.07%  1.99%  0.82%
##    64:   1.68%  1.37%  2.56%  2.07%  1.88%  0.68%
##    65:   1.69%  1.40%  2.56%  1.98%  1.83%  0.82%
##    66:   1.66%  1.34%  2.51%  2.07%  1.83%  0.68%
##    67:   1.66%  1.40%  2.56%  1.93%  1.83%  0.68%
##    68:   1.68%  1.40%  2.64%  1.98%  1.83%  0.68%
##    69:   1.71%  1.40%  2.60%  1.98%  1.99%  0.72%
##    70:   1.67%  1.40%  2.56%  1.93%  1.88%  0.72%
##    71:   1.64%  1.40%  2.38%  1.93%  1.94%  0.68%
##    72:   1.66%  1.40%  2.43%  1.98%  1.94%  0.72%
##    73:   1.68%  1.40%  2.47%  1.98%  1.94%  0.77%
##    74:   1.66%  1.40%  2.38%  1.98%  1.99%  0.72%
##    75:   1.65%  1.31%  2.34%  1.88%  2.04%  0.86%
##    76:   1.62%  1.34%  2.38%  1.83%  1.94%  0.77%
##    77:   1.69%  1.37%  2.43%  1.93%  2.09%  0.82%
##    78:   1.65%  1.34%  2.43%  1.83%  2.09%  0.72%
##    79:   1.67%  1.28%  2.56%  1.93%  2.04%  0.77%
##    80:   1.67%  1.28%  2.51%  1.98%  2.04%  0.77%
##    81:   1.67%  1.34%  2.47%  1.93%  2.09%  0.72%
##    82:   1.66%  1.31%  2.51%  1.93%  2.04%  0.72%
##    83:   1.72%  1.31%  2.56%  2.03%  2.20%  0.72%
##    84:   1.69%  1.28%  2.51%  2.03%  2.15%  0.72%
##    85:   1.73%  1.34%  2.56%  2.12%  2.09%  0.77%
##    86:   1.74%  1.34%  2.56%  2.07%  2.20%  0.77%
##    87:   1.70%  1.28%  2.51%  2.07%  2.20%  0.68%
##    88:   1.69%  1.28%  2.43%  2.03%  2.20%  0.77%
##    89:   1.65%  1.25%  2.51%  1.98%  1.99%  0.72%
##    90:   1.68%  1.28%  2.51%  1.93%  2.25%  0.68%
##    91:   1.68%  1.34%  2.51%  1.88%  2.20%  0.68%
##    92:   1.68%  1.31%  2.43%  1.98%  2.15%  0.77%
##    93:   1.69%  1.31%  2.43%  1.98%  2.25%  0.72%
##    94:   1.68%  1.40%  2.51%  1.88%  2.04%  0.72%
##    95:   1.65%  1.37%  2.47%  1.83%  2.04%  0.68%
##    96:   1.65%  1.31%  2.56%  1.83%  2.04%  0.68%
##    97:   1.68%  1.37%  2.56%  1.88%  2.04%  0.72%
##    98:   1.69%  1.31%  2.60%  1.88%  2.09%  0.77%
##    99:   1.67%  1.31%  2.60%  1.93%  1.99%  0.72%
##   100:   1.67%  1.34%  2.64%  1.88%  1.99%  0.68%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   6.97%  4.20%  8.53%  8.71%  9.70%  5.46%
##     2:   7.14%  4.56%  8.64%  9.67%  8.60%  5.75%
##     3:   7.20%  4.94%  8.49%  9.21%  8.56%  6.09%
##     4:   6.70%  4.50%  8.69%  9.19%  7.21%  5.04%
##     5:   6.28%  4.09%  8.16%  8.09%  7.59%  4.70%
##     6:   5.86%  4.01%  8.11%  7.23%  7.31%  3.70%
##     7:   5.26%  3.89%  7.18%  6.33%  6.12%  3.53%
##     8:   5.15%  3.74%  6.97%  6.96%  5.79%  3.11%
##     9:   4.65%  3.49%  5.92%  5.44%  5.59%  3.50%
##    10:   4.30%  3.51%  6.11%  4.68%  4.98%  2.65%
##    11:   4.14%  3.41%  5.22%  4.62%  4.70%  3.15%
##    12:   3.95%  3.22%  5.82%  3.83%  4.47%  2.73%
##    13:   3.64%  2.73%  5.25%  3.92%  4.51%  2.27%
##    14:   3.43%  2.69%  4.86%  3.62%  4.14%  2.22%
##    15:   3.32%  2.63%  4.85%  3.52%  3.77%  2.13%
##    16:   3.10%  2.60%  4.42%  3.19%  3.82%  1.77%
##    17:   3.02%  2.38%  4.63%  3.14%  3.40%  1.81%
##    18:   2.90%  2.29%  4.16%  2.94%  3.51%  1.90%
##    19:   2.79%  2.23%  4.20%  2.85%  3.30%  1.63%
##    20:   2.79%  2.26%  3.98%  2.85%  3.56%  1.59%
##    21:   2.74%  2.17%  3.98%  2.70%  3.72%  1.50%
##    22:   2.55%  2.11%  3.77%  2.65%  3.14%  1.31%
##    23:   2.57%  2.02%  3.55%  2.75%  3.56%  1.36%
##    24:   2.43%  1.89%  3.29%  2.70%  3.40%  1.22%
##    25:   2.40%  1.95%  3.20%  2.65%  3.45%  1.09%
##    26:   2.39%  1.80%  3.29%  2.84%  3.30%  1.09%
##    27:   2.39%  1.89%  3.16%  2.99%  3.19%  1.09%
##    28:   2.40%  1.92%  3.29%  2.65%  3.30%  1.18%
##    29:   2.26%  1.77%  3.07%  2.51%  3.19%  1.09%
##    30:   2.28%  1.83%  2.90%  2.60%  3.30%  1.09%
##    31:   2.18%  1.71%  2.90%  2.51%  3.14%  1.00%
##    32:   2.25%  1.77%  3.03%  2.60%  3.24%  0.95%
##    33:   2.26%  1.80%  3.12%  2.60%  3.04%  1.04%
##    34:   2.21%  1.74%  2.94%  2.56%  3.09%  1.04%
##    35:   2.22%  1.89%  3.03%  2.46%  2.93%  1.04%
##    36:   2.23%  1.80%  2.94%  2.60%  3.04%  1.09%
##    37:   2.22%  1.80%  2.94%  2.56%  3.04%  1.04%
##    38:   2.17%  1.74%  2.86%  2.70%  2.98%  0.91%
##    39:   2.19%  1.80%  3.03%  2.56%  2.93%  0.91%
##    40:   2.13%  1.68%  2.99%  2.51%  2.88%  0.91%
##    41:   2.06%  1.53%  2.99%  2.41%  2.77%  0.91%
##    42:   2.06%  1.47%  2.99%  2.56%  2.67%  1.00%
##    43:   2.10%  1.56%  2.99%  2.56%  2.83%  0.91%
##    44:   2.10%  1.50%  3.16%  2.56%  2.83%  0.82%
##    45:   2.01%  1.50%  2.90%  2.46%  2.77%  0.77%
##    46:   2.05%  1.50%  2.90%  2.60%  2.72%  0.86%
##    47:   1.98%  1.40%  3.03%  2.46%  2.62%  0.72%
##    48:   1.94%  1.40%  2.90%  2.41%  2.51%  0.77%
##    49:   1.96%  1.50%  2.82%  2.41%  2.67%  0.72%
##    50:   2.02%  1.53%  3.03%  2.41%  2.67%  0.77%
##    51:   1.95%  1.53%  2.99%  2.22%  2.56%  0.72%
##    52:   1.94%  1.53%  2.99%  2.27%  2.51%  0.68%
##    53:   2.00%  1.50%  3.16%  2.27%  2.72%  0.68%
##    54:   1.96%  1.56%  3.12%  2.17%  2.56%  0.63%
##    55:   2.02%  1.59%  3.20%  2.31%  2.62%  0.63%
##    56:   2.06%  1.59%  3.16%  2.36%  2.72%  0.77%
##    57:   2.00%  1.50%  3.03%  2.46%  2.67%  0.63%
##    58:   2.00%  1.56%  3.03%  2.36%  2.67%  0.68%
##    59:   1.93%  1.44%  2.94%  2.31%  2.56%  0.68%
##    60:   1.94%  1.44%  2.94%  2.31%  2.56%  0.72%
##    61:   1.98%  1.59%  3.03%  2.36%  2.51%  0.63%
##    62:   2.06%  1.68%  3.07%  2.51%  2.56%  0.68%
##    63:   2.00%  1.59%  3.07%  2.31%  2.56%  0.72%
##    64:   1.99%  1.65%  2.94%  2.41%  2.46%  0.68%
##    65:   2.00%  1.68%  2.99%  2.41%  2.41%  0.68%
##    66:   1.98%  1.59%  3.03%  2.41%  2.46%  0.63%
##    67:   2.00%  1.68%  3.03%  2.36%  2.41%  0.68%
##    68:   1.95%  1.56%  2.94%  2.36%  2.41%  0.72%
##    69:   1.94%  1.65%  2.94%  2.36%  2.30%  0.63%
##    70:   1.94%  1.65%  2.94%  2.36%  2.30%  0.59%
##    71:   1.93%  1.68%  2.82%  2.36%  2.30%  0.63%
##    72:   1.90%  1.62%  2.86%  2.31%  2.25%  0.63%
##    73:   1.87%  1.62%  2.77%  2.27%  2.20%  0.63%
##    74:   1.90%  1.59%  2.73%  2.31%  2.41%  0.68%
##    75:   1.90%  1.59%  2.82%  2.31%  2.35%  0.63%
##    76:   1.83%  1.53%  2.69%  2.17%  2.30%  0.68%
##    77:   1.84%  1.50%  2.73%  2.17%  2.35%  0.68%
##    78:   1.87%  1.53%  2.73%  2.22%  2.46%  0.63%
##    79:   1.90%  1.56%  2.77%  2.22%  2.46%  0.72%
##    80:   1.88%  1.59%  2.82%  2.22%  2.41%  0.54%
##    81:   1.92%  1.62%  2.86%  2.17%  2.46%  0.68%
##    82:   1.88%  1.62%  2.82%  2.17%  2.30%  0.63%
##    83:   1.83%  1.62%  2.77%  2.07%  2.30%  0.54%
##    84:   1.83%  1.59%  2.77%  2.12%  2.25%  0.59%
##    85:   1.80%  1.53%  2.60%  2.22%  2.25%  0.59%
##    86:   1.82%  1.50%  2.77%  2.12%  2.25%  0.63%
##    87:   1.82%  1.59%  2.73%  2.07%  2.20%  0.63%
##    88:   1.83%  1.53%  2.77%  2.12%  2.20%  0.68%
##    89:   1.88%  1.56%  2.86%  2.22%  2.20%  0.72%
##    90:   1.83%  1.50%  2.73%  2.22%  2.25%  0.68%
##    91:   1.83%  1.53%  2.69%  2.22%  2.20%  0.72%
##    92:   1.83%  1.59%  2.73%  2.12%  2.25%  0.63%
##    93:   1.83%  1.47%  2.82%  2.12%  2.35%  0.63%
##    94:   1.83%  1.44%  2.90%  2.12%  2.20%  0.68%
##    95:   1.83%  1.47%  2.86%  2.17%  2.25%  0.63%
##    96:   1.83%  1.40%  2.86%  2.22%  2.25%  0.63%
##    97:   1.81%  1.34%  2.86%  2.17%  2.30%  0.63%
##    98:   1.81%  1.50%  2.69%  2.22%  2.30%  0.54%
##    99:   1.79%  1.44%  2.69%  2.17%  2.30%  0.59%
##   100:   1.79%  1.37%  2.77%  2.17%  2.35%  0.54%
## ntree      OOB      1      2      3      4      5
##     1:   9.53%  5.10% 12.73% 12.99%  9.01%  9.93%
##     2:  10.07%  6.50% 13.44% 14.30%  8.73%  9.24%
##     3:   9.53%  6.00% 13.97% 12.33%  7.36%  9.47%
##     4:   9.28%  5.60% 12.23% 12.57%  9.46%  8.44%
##     5:   8.30%  5.05% 11.37% 11.51%  8.13%  7.13%
##     6:   7.93%  5.15% 11.41% 11.19%  7.30%  6.01%
##     7:   6.86%  5.20%  9.50%  9.04%  6.40%  4.99%
##     8:   6.28%  4.00%  8.72%  9.11%  6.35%  4.48%
##     9:   5.75%  4.26%  7.47%  8.67%  5.23%  3.93%
##    10:   5.39%  3.60%  7.03%  8.45%  5.35%  3.59%
##    11:   4.68%  3.33%  5.92%  7.38%  5.07%  2.57%
##    12:   4.39%  2.84%  5.30%  7.35%  4.34%  3.06%
##    13:   3.86%  2.41%  5.08%  6.31%  3.66%  2.70%
##    14:   3.77%  2.68%  4.82%  5.61%  3.76%  2.61%
##    15:   3.48%  2.38%  4.51%  5.31%  3.44%  2.42%
##    16:   3.06%  1.78%  4.03%  5.06%  3.18%  2.01%
##    17:   3.02%  1.87%  3.82%  5.01%  2.97%  2.10%
##    18:   2.91%  2.08%  3.90%  4.62%  2.76%  1.69%
##    19:   2.68%  1.74%  3.69%  4.28%  2.76%  1.51%
##    20:   2.58%  1.50%  3.60%  4.28%  2.66%  1.51%
##    21:   2.45%  1.56%  3.47%  3.73%  2.66%  1.37%
##    22:   2.40%  1.59%  3.03%  3.78%  2.81%  1.32%
##    23:   2.36%  1.53%  2.99%  3.73%  2.76%  1.32%
##    24:   2.31%  1.47%  3.12%  3.34%  2.92%  1.23%
##    25:   2.16%  1.47%  2.60%  3.24%  2.87%  1.10%
##    26:   2.09%  1.35%  2.86%  3.19%  2.55%  0.96%
##    27:   2.10%  1.41%  2.56%  3.14%  2.81%  1.05%
##    28:   2.07%  1.35%  2.64%  3.24%  2.45%  1.14%
##    29:   1.93%  1.20%  2.34%  3.14%  2.40%  1.05%
##    30:   2.00%  1.29%  2.82%  3.10%  2.19%  1.00%
##    31:   1.97%  1.47%  2.60%  3.00%  2.08%  1.00%
##    32:   1.86%  1.38%  2.38%  2.90%  2.03%  0.91%
##    33:   1.85%  1.29%  2.38%  2.95%  2.19%  0.82%
##    34:   1.83%  1.23%  2.30%  2.95%  2.19%  0.87%
##    35:   1.78%  1.29%  2.17%  2.75%  2.29%  0.78%
##    36:   1.77%  1.26%  2.17%  2.80%  2.08%  0.91%
##    37:   1.76%  1.20%  2.17%  2.65%  2.19%  0.96%
##    38:   1.74%  1.11%  2.21%  2.90%  2.14%  0.78%
##    39:   1.66%  1.05%  2.08%  2.95%  1.88%  0.73%
##    40:   1.68%  1.11%  2.17%  2.75%  1.93%  0.82%
##    41:   1.67%  1.26%  2.12%  2.51%  1.98%  0.78%
##    42:   1.67%  1.14%  2.08%  2.51%  1.98%  1.00%
##    43:   1.65%  1.08%  2.25%  2.51%  1.93%  0.82%
##    44:   1.65%  1.02%  2.21%  2.70%  1.98%  0.73%
##    45:   1.62%  1.02%  2.17%  2.56%  2.03%  0.73%
##    46:   1.68%  0.99%  2.38%  2.75%  2.03%  0.68%
##    47:   1.67%  0.99%  2.30%  2.60%  2.19%  0.73%
##    48:   1.61%  0.96%  2.12%  2.41%  2.14%  0.87%
##    49:   1.57%  0.96%  1.99%  2.51%  2.08%  0.73%
##    50:   1.56%  0.90%  1.95%  2.65%  2.14%  0.64%
##    51:   1.54%  0.87%  1.99%  2.46%  2.14%  0.68%
##    52:   1.56%  0.87%  2.12%  2.36%  2.14%  0.78%
##    53:   1.51%  0.90%  1.95%  2.31%  2.14%  0.68%
##    54:   1.54%  0.93%  1.99%  2.36%  2.14%  0.68%
##    55:   1.49%  0.96%  1.99%  2.36%  1.93%  0.55%
##    56:   1.51%  0.99%  2.04%  2.31%  1.88%  0.68%
##    57:   1.43%  0.96%  1.82%  2.31%  1.82%  0.55%
##    58:   1.44%  0.96%  1.91%  2.16%  1.93%  0.55%
##    59:   1.45%  0.93%  1.99%  2.21%  1.88%  0.59%
##    60:   1.42%  0.90%  1.99%  2.26%  1.77%  0.50%
##    61:   1.38%  0.87%  1.86%  2.26%  1.77%  0.50%
##    62:   1.38%  0.93%  1.82%  2.16%  1.77%  0.50%
##    63:   1.38%  0.96%  1.86%  2.21%  1.67%  0.50%
##    64:   1.37%  0.87%  1.91%  2.26%  1.62%  0.50%
##    65:   1.45%  0.96%  1.99%  2.46%  1.72%  0.46%
##    66:   1.35%  0.96%  1.73%  2.26%  1.62%  0.46%
##    67:   1.35%  0.96%  1.86%  2.06%  1.62%  0.50%
##    68:   1.35%  0.96%  1.82%  2.11%  1.67%  0.46%
##    69:   1.32%  0.90%  1.86%  2.06%  1.62%  0.41%
##    70:   1.32%  0.96%  1.78%  2.06%  1.67%  0.41%
##    71:   1.31%  1.02%  1.78%  1.97%  1.56%  0.41%
##    72:   1.32%  0.99%  1.86%  1.97%  1.56%  0.41%
##    73:   1.38%  1.05%  1.86%  2.16%  1.67%  0.37%
##    74:   1.35%  1.02%  1.95%  1.87%  1.72%  0.41%
##    75:   1.35%  1.02%  2.08%  1.72%  1.72%  0.41%
##    76:   1.34%  0.93%  2.04%  1.87%  1.67%  0.46%
##    77:   1.34%  0.90%  2.04%  1.77%  1.72%  0.55%
##    78:   1.30%  0.93%  1.95%  1.82%  1.62%  0.41%
##    79:   1.29%  0.90%  1.91%  1.87%  1.67%  0.37%
##    80:   1.28%  0.93%  1.86%  1.72%  1.77%  0.37%
##    81:   1.22%  0.90%  1.78%  1.57%  1.72%  0.37%
##    82:   1.26%  0.90%  1.82%  1.67%  1.67%  0.46%
##    83:   1.24%  0.99%  1.91%  1.47%  1.56%  0.41%
##    84:   1.23%  0.93%  1.82%  1.62%  1.62%  0.37%
##    85:   1.23%  0.99%  1.86%  1.52%  1.51%  0.41%
##    86:   1.23%  1.02%  1.73%  1.52%  1.67%  0.37%
##    87:   1.21%  0.90%  1.73%  1.62%  1.62%  0.41%
##    88:   1.21%  0.90%  1.73%  1.52%  1.62%  0.46%
##    89:   1.24%  0.93%  1.95%  1.52%  1.62%  0.37%
##    90:   1.21%  0.90%  1.82%  1.52%  1.62%  0.41%
##    91:   1.21%  0.87%  1.82%  1.52%  1.67%  0.37%
##    92:   1.22%  0.84%  1.78%  1.52%  1.77%  0.46%
##    93:   1.19%  0.87%  1.69%  1.52%  1.62%  0.46%
##    94:   1.21%  0.78%  1.78%  1.62%  1.67%  0.46%
##    95:   1.21%  0.87%  1.65%  1.72%  1.62%  0.46%
##    96:   1.18%  0.84%  1.69%  1.62%  1.51%  0.46%
##    97:   1.17%  0.78%  1.69%  1.67%  1.56%  0.41%
##    98:   1.21%  0.81%  1.82%  1.67%  1.62%  0.41%
##    99:   1.22%  0.84%  1.82%  1.62%  1.62%  0.46%
##   100:   1.19%  0.84%  1.69%  1.62%  1.56%  0.46%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.52%  6.49%  8.12%  9.84%  8.76%  5.30%
##     2:   7.51%  5.96%  9.08%  9.34%  8.83%  5.37%
##     3:   7.45%  6.33%  8.53%  9.38%  8.67%  5.26%
##     4:   7.46%  5.54% 10.08%  9.21%  8.34%  5.21%
##     5:   6.80%  4.92%  8.59%  9.28%  8.08%  4.40%
##     6:   6.25%  4.70%  7.80%  8.15%  7.50%  4.17%
##     7:   5.72%  3.97%  7.89%  7.76%  6.51%  3.55%
##     8:   5.36%  3.78%  7.71%  7.17%  6.14%  2.94%
##     9:   5.05%  3.23%  8.02%  6.98%  5.34%  2.64%
##    10:   4.50%  3.12%  7.15%  5.77%  4.47%  2.67%
##    11:   4.09%  2.84%  5.90%  5.64%  4.61%  2.20%
##    12:   3.76%  2.57%  5.66%  5.22%  3.72%  2.24%
##    13:   3.42%  2.11%  5.31%  4.53%  3.66%  2.20%
##    14:   3.00%  1.81%  5.00%  3.84%  3.50%  1.46%
##    15:   3.05%  2.05%  4.77%  4.18%  3.34%  1.46%
##    16:   2.92%  1.96%  4.77%  3.98%  3.03%  1.32%
##    17:   2.74%  1.87%  4.25%  3.79%  2.87%  1.41%
##    18:   2.63%  1.87%  4.21%  3.69%  2.61%  1.19%
##    19:   2.49%  1.81%  3.99%  3.44%  2.61%  0.96%
##    20:   2.46%  1.75%  3.82%  3.64%  2.66%  0.87%
##    21:   2.43%  1.78%  3.56%  3.49%  2.71%  1.00%
##    22:   2.33%  1.68%  3.47%  3.29%  2.55%  1.00%
##    23:   2.28%  1.65%  3.30%  3.49%  2.40%  0.96%
##    24:   2.14%  1.65%  3.25%  2.85%  2.34%  0.87%
##    25:   2.11%  1.62%  3.04%  2.95%  2.24%  0.96%
##    26:   2.10%  1.56%  3.04%  2.95%  2.34%  0.91%
##    27:   1.96%  1.35%  2.78%  3.05%  2.14%  0.87%
##    28:   1.99%  1.32%  2.77%  3.19%  2.03%  1.00%
##    29:   1.94%  1.32%  2.90%  3.00%  1.98%  0.82%
##    30:   1.90%  1.29%  2.77%  2.85%  1.98%  0.96%
##    31:   1.91%  1.35%  2.90%  2.80%  1.98%  0.82%
##    32:   1.94%  1.32%  2.99%  2.80%  2.03%  0.87%
##    33:   1.90%  1.38%  2.86%  2.80%  1.98%  0.78%
##    34:   1.86%  1.47%  2.77%  2.70%  1.93%  0.64%
##    35:   1.89%  1.41%  2.77%  2.75%  2.14%  0.64%
##    36:   1.87%  1.35%  2.77%  2.80%  2.08%  0.64%
##    37:   1.84%  1.41%  2.64%  2.75%  1.98%  0.68%
##    38:   1.83%  1.44%  2.77%  2.70%  1.82%  0.59%
##    39:   1.78%  1.29%  2.86%  2.41%  1.98%  0.64%
##    40:   1.83%  1.29%  2.73%  2.75%  2.03%  0.64%
##    41:   1.78%  1.20%  2.69%  2.70%  1.98%  0.68%
##    42:   1.78%  1.26%  2.77%  2.65%  1.88%  0.64%
##    43:   1.74%  1.23%  2.64%  2.56%  1.98%  0.59%
##    44:   1.77%  1.23%  2.73%  2.51%  2.08%  0.64%
##    45:   1.72%  1.14%  2.60%  2.56%  2.08%  0.59%
##    46:   1.71%  1.20%  2.56%  2.41%  2.03%  0.64%
##    47:   1.69%  1.23%  2.56%  2.46%  1.93%  0.55%
##    48:   1.68%  1.20%  2.64%  2.36%  1.93%  0.55%
##    49:   1.72%  1.14%  2.64%  2.41%  2.14%  0.64%
##    50:   1.70%  1.23%  2.56%  2.46%  1.93%  0.59%
##    51:   1.68%  1.17%  2.47%  2.56%  1.93%  0.59%
##    52:   1.65%  1.11%  2.47%  2.41%  2.03%  0.55%
##    53:   1.63%  1.14%  2.51%  2.26%  1.98%  0.55%
##    54:   1.62%  1.14%  2.51%  2.21%  1.88%  0.64%
##    55:   1.66%  1.08%  2.60%  2.36%  2.03%  0.55%
##    56:   1.62%  1.11%  2.34%  2.46%  2.03%  0.50%
##    57:   1.66%  1.17%  2.34%  2.46%  1.98%  0.64%
##    58:   1.65%  1.08%  2.47%  2.51%  2.03%  0.50%
##    59:   1.58%  1.02%  2.43%  2.36%  1.88%  0.55%
##    60:   1.62%  1.08%  2.47%  2.46%  1.88%  0.55%
##    61:   1.60%  0.99%  2.56%  2.51%  1.82%  0.50%
##    62:   1.62%  1.11%  2.47%  2.46%  1.93%  0.46%
##    63:   1.61%  1.11%  2.43%  2.46%  1.82%  0.55%
##    64:   1.61%  1.08%  2.43%  2.51%  1.93%  0.46%
##    65:   1.60%  1.11%  2.47%  2.41%  1.88%  0.41%
##    66:   1.58%  1.08%  2.43%  2.41%  1.88%  0.41%
##    67:   1.56%  1.08%  2.30%  2.41%  1.93%  0.41%
##    68:   1.56%  1.08%  2.38%  2.36%  1.93%  0.37%
##    69:   1.55%  0.99%  2.38%  2.41%  1.93%  0.37%
##    70:   1.60%  1.11%  2.47%  2.31%  2.03%  0.37%
##    71:   1.57%  1.14%  2.30%  2.26%  2.08%  0.37%
##    72:   1.59%  1.08%  2.38%  2.31%  2.03%  0.46%
##    73:   1.56%  1.08%  2.34%  2.21%  2.08%  0.41%
##    74:   1.55%  1.11%  2.34%  2.16%  2.03%  0.41%
##    75:   1.58%  1.05%  2.34%  2.26%  2.19%  0.41%
##    76:   1.58%  1.14%  2.30%  2.21%  2.14%  0.41%
##    77:   1.57%  1.14%  2.25%  2.26%  2.03%  0.46%
##    78:   1.57%  1.17%  2.30%  2.21%  2.08%  0.37%
##    79:   1.49%  1.11%  2.25%  2.01%  1.98%  0.37%
##    80:   1.58%  1.17%  2.30%  2.16%  2.14%  0.41%
##    81:   1.53%  0.99%  2.30%  2.16%  2.14%  0.41%
##    82:   1.53%  1.02%  2.30%  2.11%  2.08%  0.46%
##    83:   1.56%  1.08%  2.38%  2.16%  2.03%  0.46%
##    84:   1.56%  1.02%  2.34%  2.31%  2.03%  0.46%
##    85:   1.57%  1.08%  2.25%  2.31%  2.03%  0.50%
##    86:   1.58%  1.08%  2.30%  2.26%  2.08%  0.50%
##    87:   1.55%  1.05%  2.17%  2.21%  2.14%  0.50%
##    88:   1.55%  1.08%  2.17%  2.21%  2.08%  0.50%
##    89:   1.55%  1.08%  2.21%  2.16%  2.03%  0.55%
##    90:   1.55%  1.05%  2.25%  2.26%  2.03%  0.46%
##    91:   1.55%  1.08%  2.25%  2.21%  2.03%  0.46%
##    92:   1.53%  1.05%  2.34%  2.06%  2.03%  0.46%
##    93:   1.53%  1.05%  2.25%  2.16%  2.03%  0.46%
##    94:   1.50%  1.05%  2.25%  2.01%  2.03%  0.46%
##    95:   1.49%  1.05%  2.12%  2.11%  2.03%  0.41%
##    96:   1.47%  1.05%  2.12%  2.06%  1.98%  0.41%
##    97:   1.49%  1.05%  2.17%  2.11%  1.98%  0.46%
##    98:   1.48%  1.05%  2.12%  2.11%  1.98%  0.41%
##    99:   1.49%  1.08%  2.12%  2.16%  1.93%  0.41%
##   100:   1.48%  1.08%  2.04%  2.21%  1.93%  0.41%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   8.46%  3.35% 13.82%  9.92% 10.22%  7.64%
##     2:   7.92%  3.89% 12.24%  9.90%  9.18%  6.38%
##     3:   7.08%  4.11%  9.98%  9.80%  7.96%  5.30%
##     4:   7.08%  4.56%  9.85%  8.34%  8.67%  5.44%
##     5:   6.37%  4.20%  8.83%  7.78%  7.56%  4.72%
##     6:   6.20%  3.83%  8.86%  8.34%  6.98%  4.31%
##     7:   5.73%  3.81%  7.93%  7.31%  6.59%  4.12%
##     8:   5.24%  3.26%  7.34%  7.07%  6.09%  3.60%
##     9:   4.56%  2.66%  7.28%  6.47%  4.71%  2.69%
##    10:   4.47%  2.91%  7.28%  5.81%  4.44%  2.67%
##    11:   4.18%  2.54%  6.61%  5.54%  4.73%  2.39%
##    12:   3.90%  2.05%  6.37%  4.99%  4.45%  2.61%
##    13:   3.41%  2.08%  5.40%  4.23%  4.03%  2.06%
##    14:   3.18%  1.99%  4.78%  4.53%  3.61%  1.69%
##    15:   3.00%  1.96%  4.60%  4.33%  3.18%  1.51%
##    16:   2.97%  1.96%  4.56%  3.98%  3.18%  1.74%
##    17:   2.89%  1.87%  4.30%  4.08%  3.18%  1.60%
##    18:   2.65%  1.72%  4.08%  3.69%  2.61%  1.64%
##    19:   2.52%  1.81%  3.86%  3.44%  2.66%  1.23%
##    20:   2.45%  1.53%  4.08%  3.49%  2.29%  1.32%
##    21:   2.46%  1.53%  3.99%  3.54%  2.40%  1.32%
##    22:   2.37%  1.56%  4.12%  3.00%  2.24%  1.28%
##    23:   2.40%  1.44%  4.20%  3.34%  2.34%  1.14%
##    24:   2.34%  1.38%  3.81%  3.54%  2.24%  1.23%
##    25:   2.31%  1.47%  3.73%  3.24%  2.24%  1.28%
##    26:   2.22%  1.41%  3.38%  3.34%  2.24%  1.19%
##    27:   2.21%  1.53%  3.34%  3.19%  2.08%  1.23%
##    28:   2.08%  1.29%  3.47%  3.10%  2.08%  0.87%
##    29:   2.11%  1.35%  3.29%  3.00%  2.19%  1.14%
##    30:   2.03%  1.44%  3.03%  2.85%  2.08%  1.05%
##    31:   1.97%  1.29%  3.16%  2.70%  2.14%  0.91%
##    32:   1.98%  1.38%  2.99%  2.90%  1.93%  1.00%
##    33:   2.00%  1.35%  3.21%  2.85%  1.88%  1.05%
##    34:   1.94%  1.32%  3.03%  2.65%  1.82%  1.14%
##    35:   1.94%  1.32%  2.86%  2.85%  1.93%  1.10%
##    36:   1.88%  1.23%  2.86%  2.70%  1.93%  1.00%
##    37:   1.85%  1.23%  2.64%  2.90%  1.93%  0.91%
##    38:   1.85%  1.26%  2.64%  2.80%  1.98%  0.91%
##    39:   1.85%  1.23%  2.64%  2.85%  1.98%  0.91%
##    40:   1.86%  1.23%  2.69%  2.75%  2.03%  0.96%
##    41:   1.89%  1.23%  2.64%  2.85%  2.08%  1.00%
##    42:   1.84%  1.26%  2.69%  2.75%  1.93%  0.91%
##    43:   1.85%  1.23%  2.73%  2.65%  2.08%  0.91%
##    44:   1.83%  1.20%  2.73%  2.75%  1.93%  0.87%
##    45:   1.84%  1.11%  2.64%  2.80%  2.14%  0.96%
##    46:   1.83%  1.14%  2.69%  2.90%  1.93%  0.91%
##    47:   1.82%  1.23%  2.69%  2.65%  1.98%  0.87%
##    48:   1.76%  1.02%  2.69%  2.65%  1.98%  0.87%
##    49:   1.79%  1.05%  2.60%  2.65%  2.14%  0.96%
##    50:   1.77%  1.17%  2.64%  2.60%  1.88%  0.91%
##    51:   1.91%  1.20%  2.86%  2.75%  2.19%  0.96%
##    52:   1.81%  1.23%  2.64%  2.60%  1.93%  0.96%
##    53:   1.81%  1.17%  2.56%  2.70%  2.03%  0.96%
##    54:   1.80%  1.23%  2.51%  2.60%  1.98%  1.00%
##    55:   1.83%  1.23%  2.47%  2.70%  2.08%  1.00%
##    56:   1.77%  1.20%  2.47%  2.60%  2.03%  0.91%
##    57:   1.77%  1.20%  2.64%  2.41%  1.98%  0.96%
##    58:   1.74%  1.26%  2.47%  2.46%  1.88%  0.91%
##    59:   1.69%  1.17%  2.34%  2.51%  1.82%  0.91%
##    60:   1.72%  1.14%  2.51%  2.51%  1.88%  0.87%
##    61:   1.65%  1.14%  2.34%  2.36%  1.77%  0.91%
##    62:   1.66%  1.11%  2.34%  2.36%  1.88%  0.91%
##    63:   1.66%  1.08%  2.47%  2.46%  1.88%  0.78%
##    64:   1.64%  1.08%  2.38%  2.36%  1.98%  0.73%
##    65:   1.63%  1.02%  2.38%  2.41%  1.98%  0.73%
##    66:   1.66%  1.05%  2.38%  2.51%  1.98%  0.78%
##    67:   1.57%  0.99%  2.38%  2.21%  1.88%  0.73%
##    68:   1.65%  1.14%  2.38%  2.41%  1.93%  0.68%
##    69:   1.64%  1.14%  2.38%  2.41%  1.82%  0.73%
##    70:   1.60%  1.11%  2.25%  2.41%  1.93%  0.64%
##    71:   1.57%  1.11%  2.21%  2.21%  1.98%  0.64%
##    72:   1.58%  1.08%  2.25%  2.31%  1.98%  0.59%
##    73:   1.61%  1.08%  2.38%  2.21%  1.98%  0.73%
##    74:   1.63%  1.14%  2.30%  2.31%  1.98%  0.73%
##    75:   1.60%  1.11%  2.34%  2.31%  1.93%  0.64%
##    76:   1.60%  1.14%  2.25%  2.46%  1.93%  0.55%
##    77:   1.55%  1.02%  2.30%  2.26%  1.88%  0.59%
##    78:   1.55%  1.02%  2.25%  2.21%  1.88%  0.68%
##    79:   1.60%  1.11%  2.38%  2.21%  1.98%  0.59%
##    80:   1.55%  1.05%  2.25%  2.26%  1.93%  0.59%
##    81:   1.60%  1.11%  2.30%  2.31%  1.93%  0.64%
##    82:   1.59%  1.08%  2.34%  2.26%  1.93%  0.64%
##    83:   1.57%  1.14%  2.25%  2.31%  1.82%  0.59%
##    84:   1.54%  1.02%  2.34%  2.21%  1.82%  0.59%
##    85:   1.52%  1.05%  2.25%  2.16%  1.72%  0.68%
##    86:   1.54%  1.02%  2.30%  2.11%  1.88%  0.68%
##    87:   1.54%  0.99%  2.34%  2.16%  1.82%  0.68%
##    88:   1.57%  1.05%  2.34%  2.16%  1.88%  0.73%
##    89:   1.53%  1.05%  2.21%  2.11%  1.88%  0.68%
##    90:   1.55%  0.99%  2.34%  2.16%  1.88%  0.68%
##    91:   1.57%  1.11%  2.30%  2.11%  1.93%  0.68%
##    92:   1.55%  1.02%  2.38%  2.06%  1.93%  0.64%
##    93:   1.55%  1.08%  2.25%  2.21%  1.82%  0.64%
##    94:   1.51%  1.02%  2.30%  2.11%  1.77%  0.64%
##    95:   1.55%  1.02%  2.30%  2.26%  1.88%  0.59%
##    96:   1.54%  0.99%  2.21%  2.21%  1.93%  0.68%
##    97:   1.55%  1.02%  2.25%  2.16%  1.93%  0.68%
##    98:   1.54%  1.05%  2.25%  2.06%  1.93%  0.68%
##    99:   1.51%  1.02%  2.30%  2.01%  1.88%  0.64%
##   100:   1.55%  1.05%  2.34%  2.01%  1.98%  0.68%
## ntree      OOB      1      2      3      4      5
##     1:   9.96%  6.29% 13.09% 12.37%  9.13% 10.64%
##     2:  10.15%  6.98% 12.56% 12.79%  9.91% 10.17%
##     3:  10.30%  6.95% 12.57% 13.50% 11.03%  9.39%
##     4:   9.64%  6.62% 12.12% 11.76%  9.63%  9.69%
##     5:   9.01%  6.04% 11.07% 11.22% 10.28%  8.23%
##     6:   8.06%  5.29% 10.76%  9.58%  9.23%  7.07%
##     7:   7.29%  4.85%  9.06%  8.91%  8.16%  6.86%
##     8:   6.42%  4.09%  8.24%  8.71%  7.25%  5.16%
##     9:   5.69%  3.66%  6.94%  7.72%  6.74%  4.59%
##    10:   5.35%  3.22%  7.01%  6.96%  6.34%  4.47%
##    11:   4.94%  3.00%  6.27%  6.69%  6.38%  3.55%
##    12:   4.29%  2.48%  5.80%  5.92%  5.52%  2.84%
##    13:   3.91%  2.17%  5.06%  5.31%  5.31%  2.74%
##    14:   3.48%  2.02%  4.75%  4.88%  4.22%  2.36%
##    15:   3.40%  1.69%  4.61%  4.87%  4.11%  2.69%
##    16:   3.33%  1.78%  4.66%  4.31%  4.37%  2.46%
##    17:   3.00%  1.54%  4.03%  4.17%  3.90%  2.22%
##    18:   2.96%  1.72%  3.98%  4.07%  3.69%  2.04%
##    19:   2.67%  1.57%  3.85%  3.61%  3.37%  1.62%
##    20:   2.57%  1.42%  3.67%  3.42%  3.43%  1.62%
##    21:   2.38%  1.36%  3.31%  3.14%  3.17%  1.53%
##    22:   2.31%  1.27%  3.31%  3.14%  3.01%  1.44%
##    23:   2.23%  1.02%  3.67%  2.72%  3.17%  1.30%
##    24:   2.20%  1.11%  3.35%  2.86%  3.01%  1.30%
##    25:   2.14%  1.05%  3.26%  3.00%  2.80%  1.20%
##    26:   2.18%  1.02%  3.44%  2.76%  3.17%  1.20%
##    27:   2.03%  1.17%  2.95%  2.62%  2.86%  1.07%
##    28:   2.02%  1.08%  2.91%  2.48%  2.96%  1.25%
##    29:   1.96%  0.99%  2.82%  2.39%  3.01%  1.20%
##    30:   1.87%  0.96%  2.73%  2.39%  2.70%  1.11%
##    31:   1.84%  0.81%  2.73%  2.39%  2.75%  1.16%
##    32:   1.84%  0.87%  2.68%  2.39%  2.80%  1.07%
##    33:   1.83%  0.96%  2.68%  2.25%  2.80%  0.97%
##    34:   1.89%  0.81%  3.00%  2.30%  2.91%  1.11%
##    35:   1.82%  0.81%  2.82%  2.48%  2.70%  0.88%
##    36:   1.83%  0.72%  2.82%  2.39%  2.80%  1.07%
##    37:   1.77%  0.81%  2.77%  2.15%  2.75%  0.97%
##    38:   1.83%  0.78%  2.95%  2.34%  2.75%  0.97%
##    39:   1.77%  0.81%  2.68%  2.06%  2.86%  1.02%
##    40:   1.73%  0.78%  2.73%  2.20%  2.60%  0.93%
##    41:   1.72%  0.84%  2.33%  2.20%  2.80%  1.02%
##    42:   1.72%  0.87%  2.50%  2.11%  2.70%  0.97%
##    43:   1.67%  0.93%  2.42%  2.11%  2.44%  0.93%
##    44:   1.68%  0.96%  2.46%  2.06%  2.49%  0.88%
##    45:   1.68%  0.87%  2.50%  2.06%  2.54%  0.93%
##    46:   1.66%  0.87%  2.59%  2.06%  2.49%  0.79%
##    47:   1.68%  0.87%  2.42%  2.25%  2.44%  0.93%
##    48:   1.59%  0.69%  2.50%  2.01%  2.39%  0.88%
##    49:   1.60%  0.78%  2.50%  2.01%  2.39%  0.79%
##    50:   1.60%  0.72%  2.42%  2.20%  2.44%  0.79%
##    51:   1.61%  0.75%  2.55%  1.97%  2.39%  0.93%
##    52:   1.56%  0.75%  2.37%  1.97%  2.39%  0.83%
##    53:   1.56%  0.78%  2.37%  2.01%  2.34%  0.79%
##    54:   1.57%  0.66%  2.50%  2.06%  2.34%  0.83%
##    55:   1.55%  0.66%  2.28%  2.25%  2.34%  0.74%
##    56:   1.55%  0.75%  2.24%  2.11%  2.39%  0.79%
##    57:   1.59%  0.66%  2.37%  2.30%  2.44%  0.74%
##    58:   1.51%  0.72%  2.24%  2.15%  2.23%  0.69%
##    59:   1.53%  0.75%  2.28%  2.01%  2.39%  0.69%
##    60:   1.52%  0.78%  2.24%  2.01%  2.28%  0.74%
##    61:   1.51%  0.84%  2.33%  1.83%  2.28%  0.69%
##    62:   1.48%  0.78%  2.15%  1.87%  2.34%  0.69%
##    63:   1.49%  0.69%  2.15%  2.06%  2.28%  0.79%
##    64:   1.44%  0.72%  1.97%  1.87%  2.34%  0.74%
##    65:   1.40%  0.69%  1.88%  1.87%  2.28%  0.74%
##    66:   1.44%  0.72%  2.01%  1.83%  2.39%  0.74%
##    67:   1.43%  0.69%  1.88%  1.83%  2.39%  0.83%
##    68:   1.41%  0.72%  1.92%  1.69%  2.44%  0.74%
##    69:   1.38%  0.69%  1.70%  1.87%  2.34%  0.74%
##    70:   1.31%  0.63%  1.65%  1.73%  2.28%  0.69%
##    71:   1.34%  0.60%  1.61%  1.87%  2.34%  0.79%
##    72:   1.32%  0.66%  1.70%  1.69%  2.39%  0.65%
##    73:   1.33%  0.63%  1.74%  1.78%  2.23%  0.74%
##    74:   1.33%  0.57%  1.74%  1.97%  2.23%  0.65%
##    75:   1.32%  0.57%  1.79%  1.78%  2.34%  0.60%
##    76:   1.33%  0.57%  1.79%  1.83%  2.34%  0.65%
##    77:   1.33%  0.57%  1.83%  1.87%  2.28%  0.60%
##    78:   1.35%  0.60%  1.88%  1.83%  2.34%  0.60%
##    79:   1.32%  0.57%  1.74%  1.87%  2.34%  0.60%
##    80:   1.34%  0.60%  1.79%  1.87%  2.34%  0.60%
##    81:   1.36%  0.60%  1.83%  1.78%  2.39%  0.69%
##    82:   1.37%  0.60%  1.83%  1.83%  2.39%  0.69%
##    83:   1.35%  0.60%  1.79%  1.83%  2.39%  0.65%
##    84:   1.35%  0.60%  1.88%  1.73%  2.34%  0.69%
##    85:   1.31%  0.60%  1.65%  1.64%  2.28%  0.83%
##    86:   1.28%  0.63%  1.70%  1.59%  2.34%  0.60%
##    87:   1.32%  0.66%  1.61%  1.73%  2.34%  0.69%
##    88:   1.31%  0.60%  1.65%  1.78%  2.28%  0.69%
##    89:   1.32%  0.63%  1.70%  1.73%  2.28%  0.69%
##    90:   1.31%  0.60%  1.74%  1.64%  2.28%  0.74%
##    91:   1.35%  0.60%  1.79%  1.83%  2.34%  0.69%
##    92:   1.32%  0.63%  1.70%  1.69%  2.34%  0.69%
##    93:   1.32%  0.63%  1.61%  1.73%  2.34%  0.74%
##    94:   1.32%  0.63%  1.65%  1.73%  2.23%  0.79%
##    95:   1.29%  0.60%  1.74%  1.64%  2.18%  0.74%
##    96:   1.27%  0.57%  1.79%  1.55%  2.18%  0.74%
##    97:   1.27%  0.63%  1.65%  1.55%  2.23%  0.74%
##    98:   1.32%  0.66%  1.70%  1.64%  2.28%  0.74%
##    99:   1.30%  0.60%  1.74%  1.64%  2.23%  0.74%
##   100:   1.35%  0.66%  1.79%  1.64%  2.34%  0.79%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   8.21%  4.81% 12.00% 12.07%  8.87%  5.18%
##     2:   7.95%  5.72% 11.28% 10.57%  8.43%  5.08%
##     3:   7.51%  5.16%  9.70% 10.39%  8.35%  5.32%
##     4:   7.55%  5.53%  9.41% 10.11%  8.37%  5.48%
##     5:   6.80%  4.62%  9.24%  8.73%  7.43%  5.20%
##     6:   6.06%  4.19%  8.51%  7.33%  6.82%  4.45%
##     7:   5.52%  3.94%  7.74%  6.53%  6.84%  3.44%
##     8:   5.19%  3.82%  6.94%  6.12%  6.24%  3.62%
##     9:   4.92%  3.47%  6.76%  6.18%  5.71%  3.30%
##    10:   4.49%  3.15%  6.46%  5.73%  5.04%  2.81%
##    11:   4.11%  2.57%  5.68%  5.43%  5.08%  2.71%
##    12:   3.77%  2.23%  5.03%  4.99%  4.85%  2.65%
##    13:   3.64%  2.32%  5.38%  4.56%  4.42%  2.28%
##    14:   3.31%  1.87%  4.88%  4.18%  4.21%  2.23%
##    15:   3.33%  2.02%  4.61%  4.36%  4.26%  2.13%
##    16:   3.18%  1.81%  4.16%  4.50%  4.21%  2.04%
##    17:   2.94%  1.48%  3.80%  4.31%  3.79%  2.18%
##    18:   2.77%  1.45%  3.67%  3.98%  3.64%  1.90%
##    19:   2.60%  1.57%  3.22%  3.66%  3.38%  1.81%
##    20:   2.73%  1.63%  3.58%  3.98%  3.22%  1.90%
##    21:   2.52%  1.33%  3.44%  3.70%  3.12%  1.71%
##    22:   2.46%  1.45%  3.26%  3.65%  2.96%  1.57%
##    23:   2.38%  1.42%  3.04%  3.65%  2.80%  1.53%
##    24:   2.46%  1.60%  3.26%  3.51%  2.86%  1.57%
##    25:   2.34%  1.36%  3.35%  3.37%  2.80%  1.34%
##    26:   2.24%  1.05%  3.35%  3.42%  2.80%  1.25%
##    27:   2.22%  1.30%  3.09%  3.23%  2.80%  1.20%
##    28:   2.17%  1.17%  3.04%  3.23%  2.70%  1.30%
##    29:   2.18%  1.17%  3.09%  3.23%  2.70%  1.30%
##    30:   2.16%  1.20%  2.95%  3.19%  2.65%  1.34%
##    31:   2.09%  1.05%  2.91%  3.09%  2.60%  1.39%
##    32:   2.19%  1.05%  3.09%  3.28%  2.70%  1.48%
##    33:   2.12%  1.08%  3.04%  3.00%  2.70%  1.39%
##    34:   2.11%  1.02%  3.00%  3.14%  2.65%  1.34%
##    35:   2.07%  0.99%  2.86%  3.14%  2.60%  1.39%
##    36:   2.06%  1.08%  2.77%  3.14%  2.49%  1.34%
##    37:   2.06%  0.99%  2.73%  3.28%  2.60%  1.34%
##    38:   2.02%  0.96%  2.91%  2.81%  2.54%  1.48%
##    39:   1.91%  0.93%  2.55%  2.81%  2.49%  1.34%
##    40:   2.00%  1.05%  2.68%  2.90%  2.54%  1.34%
##    41:   1.93%  0.93%  2.64%  2.81%  2.49%  1.34%
##    42:   1.89%  0.96%  2.50%  2.72%  2.54%  1.25%
##    43:   1.88%  0.90%  2.50%  2.67%  2.60%  1.30%
##    44:   1.89%  0.99%  2.59%  2.62%  2.49%  1.30%
##    45:   1.92%  0.96%  2.59%  2.76%  2.60%  1.25%
##    46:   1.93%  0.96%  2.50%  2.90%  2.54%  1.30%
##    47:   1.86%  0.90%  2.42%  2.76%  2.54%  1.25%
##    48:   1.92%  0.99%  2.46%  2.90%  2.49%  1.30%
##    49:   1.90%  0.93%  2.46%  2.90%  2.49%  1.30%
##    50:   1.87%  0.93%  2.42%  2.86%  2.54%  1.16%
##    51:   1.90%  0.93%  2.50%  2.86%  2.60%  1.20%
##    52:   1.90%  0.90%  2.46%  2.90%  2.65%  1.20%
##    53:   1.88%  0.90%  2.46%  2.76%  2.60%  1.25%
##    54:   1.89%  0.93%  2.50%  2.81%  2.60%  1.16%
##    55:   1.83%  0.90%  2.46%  2.76%  2.44%  1.16%
##    56:   1.80%  0.87%  2.46%  2.72%  2.34%  1.16%
##    57:   1.81%  0.84%  2.37%  2.81%  2.39%  1.20%
##    58:   1.79%  0.87%  2.33%  2.81%  2.34%  1.16%
##    59:   1.80%  0.81%  2.42%  2.81%  2.44%  1.11%
##    60:   1.77%  0.78%  2.37%  2.81%  2.44%  1.02%
##    61:   1.77%  0.81%  2.33%  2.76%  2.44%  1.11%
##    62:   1.80%  0.75%  2.55%  2.76%  2.49%  1.07%
##    63:   1.78%  0.81%  2.42%  2.72%  2.49%  1.07%
##    64:   1.77%  0.78%  2.50%  2.58%  2.49%  1.07%
##    65:   1.77%  0.81%  2.46%  2.72%  2.44%  1.02%
##    66:   1.79%  0.87%  2.50%  2.62%  2.44%  1.07%
##    67:   1.83%  0.87%  2.64%  2.72%  2.49%  1.02%
##    68:   1.80%  0.84%  2.59%  2.72%  2.39%  1.02%
##    69:   1.78%  0.81%  2.68%  2.62%  2.39%  0.97%
##    70:   1.78%  0.81%  2.64%  2.48%  2.49%  1.07%
##    71:   1.75%  0.78%  2.55%  2.48%  2.54%  0.97%
##    72:   1.80%  0.87%  2.55%  2.58%  2.54%  1.02%
##    73:   1.78%  0.81%  2.64%  2.48%  2.54%  1.02%
##    74:   1.81%  0.81%  2.68%  2.58%  2.54%  1.02%
##    75:   1.80%  0.84%  2.64%  2.53%  2.54%  1.02%
##    76:   1.78%  0.81%  2.59%  2.53%  2.54%  1.02%
##    77:   1.77%  0.84%  2.55%  2.48%  2.54%  1.02%
##    78:   1.77%  0.87%  2.50%  2.48%  2.60%  0.97%
##    79:   1.82%  0.84%  2.73%  2.44%  2.65%  1.02%
##    80:   1.72%  0.75%  2.50%  2.34%  2.54%  1.02%
##    81:   1.72%  0.81%  2.42%  2.34%  2.54%  1.02%
##    82:   1.75%  0.81%  2.46%  2.48%  2.54%  1.02%
##    83:   1.79%  0.87%  2.55%  2.53%  2.54%  1.02%
##    84:   1.78%  0.84%  2.55%  2.58%  2.54%  0.97%
##    85:   1.72%  0.78%  2.46%  2.53%  2.49%  0.93%
##    86:   1.73%  0.75%  2.55%  2.58%  2.49%  0.88%
##    87:   1.72%  0.78%  2.50%  2.53%  2.49%  0.88%
##    88:   1.74%  0.75%  2.55%  2.48%  2.54%  0.97%
##    89:   1.74%  0.69%  2.59%  2.53%  2.54%  0.97%
##    90:   1.73%  0.75%  2.59%  2.48%  2.54%  0.88%
##    91:   1.73%  0.78%  2.55%  2.48%  2.54%  0.88%
##    92:   1.76%  0.72%  2.73%  2.62%  2.54%  0.79%
##    93:   1.73%  0.72%  2.68%  2.53%  2.60%  0.74%
##    94:   1.71%  0.72%  2.55%  2.53%  2.54%  0.79%
##    95:   1.71%  0.72%  2.42%  2.53%  2.65%  0.83%
##    96:   1.74%  0.75%  2.50%  2.62%  2.60%  0.83%
##    97:   1.73%  0.72%  2.68%  2.53%  2.54%  0.79%
##    98:   1.69%  0.72%  2.50%  2.53%  2.54%  0.74%
##    99:   1.67%  0.69%  2.42%  2.53%  2.54%  0.79%
##   100:   1.72%  0.75%  2.46%  2.58%  2.54%  0.83%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.45%  4.99%  9.71% 10.25%  8.11%  5.49%
##     2:   7.90%  5.14%  9.82% 10.87%  7.42%  7.62%
##     3:   7.00%  4.23% 10.17%  8.63%  7.27%  6.20%
##     4:   6.67%  3.96%  9.86%  8.10%  7.33%  5.60%
##     5:   5.97%  3.27%  8.84%  7.27%  6.88%  5.07%
##     6:   5.99%  3.31%  8.83%  7.27%  6.94%  5.06%
##     7:   5.58%  3.36%  8.48%  6.85%  6.30%  4.13%
##     8:   5.21%  3.16%  7.36%  6.60%  5.88%  4.16%
##     9:   4.69%  2.91%  7.10%  5.71%  5.18%  3.47%
##    10:   4.34%  2.65%  6.28%  5.55%  4.83%  3.32%
##    11:   3.91%  2.43%  5.86%  4.71%  4.55%  2.79%
##    12:   3.62%  2.27%  5.35%  4.28%  4.60%  2.37%
##    13:   3.42%  2.18%  5.25%  4.41%  3.96%  2.00%
##    14:   3.33%  2.02%  4.84%  4.55%  3.85%  2.13%
##    15:   3.09%  2.23%  4.34%  4.03%  3.43%  1.90%
##    16:   3.04%  1.93%  4.57%  4.45%  3.53%  1.34%
##    17:   2.80%  1.75%  4.52%  3.75%  3.38%  1.16%
##    18:   2.62%  1.81%  3.98%  3.65%  3.01%  1.11%
##    19:   2.68%  1.93%  4.12%  3.51%  3.06%  1.16%
##    20:   2.56%  1.63%  3.94%  3.47%  3.32%  1.02%
##    21:   2.55%  1.63%  3.71%  3.37%  3.27%  1.30%
##    22:   2.54%  1.60%  3.76%  3.70%  3.06%  1.11%
##    23:   2.56%  1.48%  3.62%  3.61%  3.48%  1.25%
##    24:   2.44%  1.48%  3.67%  3.33%  3.01%  1.25%
##    25:   2.32%  1.39%  3.67%  3.14%  2.91%  1.02%
##    26:   2.34%  1.54%  3.22%  3.14%  3.17%  1.16%
##    27:   2.27%  1.57%  3.31%  3.04%  2.86%  0.97%
##    28:   2.28%  1.45%  3.35%  3.04%  3.17%  0.88%
##    29:   2.26%  1.33%  3.22%  3.14%  3.06%  1.11%
##    30:   2.32%  1.45%  3.31%  3.19%  3.17%  1.02%
##    31:   2.28%  1.30%  3.22%  3.61%  2.91%  0.97%
##    32:   2.27%  1.27%  3.31%  3.42%  2.96%  0.97%
##    33:   2.24%  1.11%  3.31%  3.51%  2.80%  1.11%
##    34:   2.22%  1.17%  3.09%  3.56%  2.86%  1.07%
##    35:   2.13%  1.11%  3.22%  3.19%  2.91%  0.83%
##    36:   2.19%  1.11%  3.13%  3.42%  3.01%  0.93%
##    37:   2.17%  1.14%  3.26%  3.28%  2.91%  0.83%
##    38:   2.14%  1.14%  3.13%  3.14%  2.91%  0.97%
##    39:   2.11%  1.05%  2.95%  3.51%  2.80%  0.83%
##    40:   2.00%  0.99%  2.91%  3.14%  2.86%  0.74%
##    41:   2.06%  1.14%  2.91%  3.14%  2.91%  0.74%
##    42:   2.05%  1.08%  3.04%  3.14%  2.70%  0.83%
##    43:   2.00%  1.20%  2.77%  3.19%  2.70%  0.65%
##    44:   1.95%  1.08%  2.77%  2.95%  2.75%  0.74%
##    45:   1.91%  1.11%  2.68%  2.90%  2.65%  0.69%
##    46:   1.89%  0.96%  2.82%  3.00%  2.49%  0.69%
##    47:   1.89%  1.05%  2.77%  2.81%  2.70%  0.65%
##    48:   1.94%  1.02%  2.82%  2.90%  2.70%  0.79%
##    49:   1.89%  0.96%  2.73%  2.90%  2.65%  0.79%
##    50:   1.88%  0.96%  2.64%  2.81%  2.86%  0.69%
##    51:   2.00%  1.02%  2.86%  3.00%  2.91%  0.83%
##    52:   1.95%  1.11%  2.64%  2.90%  2.86%  0.79%
##    53:   1.93%  0.96%  2.77%  2.86%  2.86%  0.79%
##    54:   1.89%  0.96%  2.59%  2.90%  2.70%  0.83%
##    55:   1.86%  0.99%  2.55%  2.81%  2.75%  0.74%
##    56:   1.86%  0.96%  2.59%  2.76%  2.70%  0.83%
##    57:   1.88%  0.99%  2.59%  2.86%  2.70%  0.79%
##    58:   1.89%  0.96%  2.73%  2.95%  2.60%  0.79%
##    59:   1.86%  1.02%  2.68%  2.81%  2.60%  0.69%
##    60:   1.92%  0.96%  2.91%  2.95%  2.54%  0.79%
##    61:   1.89%  0.96%  2.77%  2.86%  2.54%  0.83%
##    62:   1.83%  0.90%  2.73%  2.86%  2.54%  0.69%
##    63:   1.83%  0.90%  2.68%  2.90%  2.44%  0.79%
##    64:   1.78%  0.90%  2.55%  2.81%  2.39%  0.79%
##    65:   1.77%  0.84%  2.55%  2.76%  2.44%  0.79%
##    66:   1.72%  0.84%  2.50%  2.62%  2.39%  0.74%
##    67:   1.74%  0.87%  2.46%  2.67%  2.39%  0.83%
##    68:   1.77%  0.84%  2.46%  2.81%  2.39%  0.88%
##    69:   1.77%  0.90%  2.46%  2.67%  2.44%  0.88%
##    70:   1.76%  0.90%  2.50%  2.67%  2.39%  0.83%
##    71:   1.79%  0.93%  2.55%  2.72%  2.44%  0.83%
##    72:   1.74%  0.87%  2.46%  2.67%  2.39%  0.83%
##    73:   1.74%  0.90%  2.50%  2.67%  2.34%  0.79%
##    74:   1.76%  0.87%  2.46%  2.72%  2.44%  0.83%
##    75:   1.77%  0.93%  2.42%  2.76%  2.44%  0.79%
##    76:   1.78%  0.90%  2.55%  2.81%  2.39%  0.79%
##    77:   1.74%  0.90%  2.37%  2.67%  2.44%  0.83%
##    78:   1.72%  0.90%  2.37%  2.62%  2.39%  0.83%
##    79:   1.71%  0.87%  2.33%  2.67%  2.39%  0.79%
##    80:   1.75%  0.96%  2.42%  2.67%  2.39%  0.79%
##    81:   1.73%  0.90%  2.33%  2.72%  2.39%  0.83%
##    82:   1.72%  0.93%  2.24%  2.62%  2.39%  0.88%
##    83:   1.70%  0.93%  2.24%  2.62%  2.39%  0.79%
##    84:   1.69%  0.90%  2.33%  2.53%  2.39%  0.79%
##    85:   1.68%  0.90%  2.33%  2.53%  2.34%  0.79%
##    86:   1.66%  0.90%  2.28%  2.48%  2.34%  0.74%
##    87:   1.69%  0.87%  2.46%  2.48%  2.39%  0.74%
##    88:   1.67%  0.87%  2.33%  2.44%  2.39%  0.83%
##    89:   1.65%  0.87%  2.28%  2.39%  2.39%  0.79%
##    90:   1.65%  0.84%  2.37%  2.44%  2.34%  0.74%
##    91:   1.67%  0.90%  2.46%  2.39%  2.28%  0.79%
##    92:   1.67%  0.87%  2.42%  2.48%  2.34%  0.74%
##    93:   1.68%  0.93%  2.33%  2.48%  2.28%  0.83%
##    94:   1.67%  0.90%  2.46%  2.34%  2.28%  0.83%
##    95:   1.66%  0.84%  2.46%  2.39%  2.39%  0.74%
##    96:   1.66%  0.87%  2.46%  2.34%  2.34%  0.79%
##    97:   1.69%  0.90%  2.46%  2.44%  2.34%  0.79%
##    98:   1.68%  0.87%  2.42%  2.44%  2.39%  0.79%
##    99:   1.67%  0.84%  2.42%  2.44%  2.39%  0.79%
##   100:   1.66%  0.81%  2.37%  2.44%  2.39%  0.79%
## ntree      OOB      1      2      3      4      5
##     1:  10.75%  6.69% 15.02% 13.57% 11.07%  9.56%
##     2:  10.09%  6.66% 14.72% 12.69% 10.51%  7.72%
##     3:   9.54%  6.03% 12.91% 13.14%  9.43%  8.21%
##     4:   9.44%  5.73% 12.56% 12.60% 10.19%  8.37%
##     5:   8.57%  5.52% 11.31% 11.68%  8.83%  7.33%
##     6:   7.93%  4.74% 10.37% 11.28%  8.47%  6.73%
##     7:   7.15%  4.48%  9.66%  9.26%  7.93%  5.97%
##     8:   6.58%  4.08%  9.03%  8.83%  6.95%  5.47%
##     9:   5.67%  3.41%  7.56%  7.18%  6.15%  5.41%
##    10:   5.46%  3.01%  7.82%  6.50%  6.23%  5.14%
##    11:   4.95%  2.85%  7.21%  6.16%  5.70%  4.03%
##    12:   4.42%  2.42%  6.48%  5.45%  4.91%  3.98%
##    13:   4.16%  2.39%  5.51%  5.39%  4.96%  3.64%
##    14:   3.87%  2.30%  5.28%  4.79%  4.49%  3.40%
##    15:   3.62%  2.09%  4.96%  4.94%  4.18%  2.83%
##    16:   3.34%  1.56%  4.70%  4.25%  4.23%  3.02%
##    17:   3.18%  1.83%  4.39%  4.10%  4.02%  2.41%
##    18:   3.04%  1.80%  4.66%  3.95%  3.52%  1.98%
##    19:   2.69%  1.68%  3.91%  3.21%  3.41%  1.84%
##    20:   2.61%  1.59%  3.91%  3.26%  3.21%  1.65%
##    21:   2.59%  1.53%  3.91%  2.96%  3.21%  1.94%
##    22:   2.41%  1.50%  3.56%  2.81%  3.01%  1.70%
##    23:   2.24%  1.27%  3.78%  2.57%  2.50%  1.61%
##    24:   2.18%  1.41%  3.21%  2.66%  2.65%  1.42%
##    25:   2.18%  1.33%  3.21%  2.91%  2.65%  1.32%
##    26:   2.00%  1.15%  3.12%  2.57%  2.34%  1.28%
##    27:   2.10%  1.21%  3.16%  2.86%  2.60%  1.18%
##    28:   2.17%  1.36%  3.21%  2.81%  2.60%  1.32%
##    29:   1.91%  1.24%  2.77%  2.71%  2.04%  1.18%
##    30:   1.96%  1.27%  2.86%  2.66%  2.29%  1.13%
##    31:   1.93%  1.30%  2.72%  2.57%  2.24%  1.18%
##    32:   1.80%  1.12%  2.55%  2.37%  2.19%  1.18%
##    33:   1.85%  1.21%  2.42%  2.61%  2.19%  1.23%
##    34:   1.81%  1.21%  2.42%  2.52%  1.99%  1.28%
##    35:   1.77%  1.09%  2.37%  2.71%  1.94%  1.18%
##    36:   1.72%  1.15%  2.24%  2.37%  2.14%  1.04%
##    37:   1.67%  1.06%  2.20%  2.57%  2.04%  0.90%
##    38:   1.63%  1.03%  2.15%  2.37%  1.99%  0.99%
##    39:   1.70%  1.03%  2.11%  2.81%  1.99%  0.99%
##    40:   1.62%  0.97%  2.20%  2.47%  1.94%  0.94%
##    41:   1.69%  1.03%  2.42%  2.42%  1.99%  0.99%
##    42:   1.63%  0.91%  2.28%  2.57%  1.83%  0.99%
##    43:   1.57%  0.94%  2.07%  2.27%  1.78%  1.18%
##    44:   1.55%  0.88%  1.93%  2.37%  1.94%  1.04%
##    45:   1.57%  0.91%  2.33%  2.07%  1.88%  1.04%
##    46:   1.49%  0.74%  2.33%  2.07%  1.73%  1.04%
##    47:   1.52%  0.88%  2.20%  2.17%  1.73%  0.99%
##    48:   1.49%  0.80%  2.24%  2.07%  1.78%  0.94%
##    49:   1.53%  0.85%  2.33%  2.07%  1.78%  0.99%
##    50:   1.49%  0.88%  2.37%  2.02%  1.58%  0.90%
##    51:   1.51%  0.83%  2.24%  2.17%  1.63%  1.09%
##    52:   1.38%  0.77%  2.20%  1.97%  1.43%  0.90%
##    53:   1.44%  0.91%  2.33%  1.92%  1.38%  0.90%
##    54:   1.45%  0.80%  2.24%  2.12%  1.43%  1.04%
##    55:   1.37%  0.74%  2.15%  1.97%  1.43%  0.90%
##    56:   1.41%  0.74%  2.15%  1.92%  1.63%  0.99%
##    57:   1.40%  0.71%  2.33%  1.83%  1.63%  0.90%
##    58:   1.39%  0.80%  2.24%  1.92%  1.48%  0.85%
##    59:   1.36%  0.88%  2.15%  1.68%  1.43%  0.90%
##    60:   1.29%  0.83%  2.11%  1.58%  1.48%  0.71%
##    61:   1.34%  0.74%  2.02%  1.83%  1.43%  1.04%
##    62:   1.41%  0.80%  2.11%  1.97%  1.58%  0.94%
##    63:   1.37%  0.74%  2.15%  1.87%  1.58%  0.85%
##    64:   1.27%  0.68%  1.93%  1.68%  1.48%  0.94%
##    65:   1.33%  0.77%  2.15%  1.83%  1.43%  0.80%
##    66:   1.32%  0.74%  2.11%  1.78%  1.48%  0.85%
##    67:   1.33%  0.74%  2.02%  1.92%  1.43%  0.90%
##    68:   1.29%  0.68%  2.02%  1.92%  1.32%  0.85%
##    69:   1.28%  0.71%  2.02%  1.73%  1.38%  0.90%
##    70:   1.23%  0.74%  1.89%  1.68%  1.27%  0.85%
##    71:   1.22%  0.68%  1.93%  1.73%  1.32%  0.76%
##    72:   1.24%  0.71%  1.89%  1.78%  1.32%  0.80%
##    73:   1.21%  0.68%  1.98%  1.68%  1.32%  0.71%
##    74:   1.22%  0.68%  2.07%  1.73%  1.27%  0.66%
##    75:   1.21%  0.68%  1.98%  1.68%  1.38%  0.66%
##    76:   1.28%  0.71%  2.15%  1.73%  1.38%  0.76%
##    77:   1.23%  0.68%  1.98%  1.78%  1.32%  0.71%
##    78:   1.25%  0.65%  2.07%  1.73%  1.48%  0.66%
##    79:   1.19%  0.65%  1.89%  1.73%  1.32%  0.66%
##    80:   1.26%  0.74%  2.02%  1.73%  1.48%  0.61%
##    81:   1.19%  0.62%  1.89%  1.78%  1.38%  0.61%
##    82:   1.15%  0.62%  1.89%  1.63%  1.38%  0.57%
##    83:   1.15%  0.65%  1.80%  1.68%  1.27%  0.61%
##    84:   1.15%  0.65%  1.93%  1.53%  1.27%  0.66%
##    85:   1.18%  0.65%  2.02%  1.68%  1.27%  0.57%
##    86:   1.12%  0.62%  1.80%  1.63%  1.22%  0.61%
##    87:   1.12%  0.65%  1.76%  1.58%  1.27%  0.61%
##    88:   1.12%  0.59%  1.85%  1.58%  1.32%  0.57%
##    89:   1.15%  0.59%  1.98%  1.63%  1.27%  0.57%
##    90:   1.16%  0.65%  2.02%  1.58%  1.32%  0.52%
##    91:   1.15%  0.62%  1.89%  1.63%  1.43%  0.52%
##    92:   1.16%  0.68%  1.85%  1.58%  1.38%  0.61%
##    93:   1.16%  0.59%  1.89%  1.68%  1.38%  0.61%
##    94:   1.16%  0.62%  1.80%  1.68%  1.43%  0.61%
##    95:   1.18%  0.62%  1.93%  1.63%  1.43%  0.61%
##    96:   1.09%  0.53%  1.67%  1.58%  1.38%  0.61%
##    97:   1.12%  0.56%  1.71%  1.68%  1.38%  0.61%
##    98:   1.09%  0.53%  1.76%  1.58%  1.32%  0.57%
##    99:   1.10%  0.53%  1.71%  1.63%  1.38%  0.57%
##   100:   1.13%  0.56%  1.76%  1.73%  1.38%  0.57%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.86%  4.76% 10.55%  7.96% 10.38%  7.38%
##     2:   7.42%  5.09% 10.11%  7.64%  7.74%  7.61%
##     3:   7.00%  4.91%  9.85%  8.62%  6.48%  6.08%
##     4:   6.59%  4.28%  8.91%  8.58%  6.65%  5.76%
##     5:   6.03%  4.16%  7.91%  8.76%  5.64%  4.73%
##     6:   5.86%  3.60%  7.56%  8.78%  6.38%  4.34%
##     7:   5.53%  4.00%  7.42%  7.75%  5.71%  3.60%
##     8:   4.83%  3.36%  6.24%  7.53%  4.83%  3.06%
##     9:   4.78%  2.90%  6.28%  7.65%  5.12%  3.08%
##    10:   4.34%  2.69%  5.76%  6.70%  4.69%  2.87%
##    11:   3.92%  2.32%  5.48%  5.87%  4.05%  2.81%
##    12:   3.62%  2.23%  4.81%  5.65%  3.79%  2.47%
##    13:   3.41%  1.98%  4.93%  5.19%  3.58%  2.18%
##    14:   3.23%  1.86%  4.40%  4.89%  3.83%  2.04%
##    15:   3.09%  1.86%  4.13%  4.64%  3.47%  2.13%
##    16:   3.04%  1.71%  4.35%  4.59%  3.47%  1.89%
##    17:   2.95%  1.74%  4.09%  4.34%  3.31%  1.98%
##    18:   2.88%  1.56%  3.95%  4.54%  3.16%  1.98%
##    19:   2.73%  1.71%  3.65%  4.05%  2.75%  2.13%
##    20:   2.67%  1.65%  3.78%  3.90%  2.75%  1.84%
##    21:   2.51%  1.59%  3.38%  3.80%  2.80%  1.51%
##    22:   2.48%  1.59%  3.30%  3.75%  2.60%  1.70%
##    23:   2.52%  1.56%  3.51%  3.85%  2.70%  1.56%
##    24:   2.45%  1.68%  3.47%  3.40%  2.50%  1.61%
##    25:   2.33%  1.33%  3.56%  3.45%  2.39%  1.46%
##    26:   2.36%  1.41%  3.51%  3.70%  2.29%  1.42%
##    27:   2.25%  1.36%  3.21%  3.55%  2.29%  1.37%
##    28:   2.21%  1.15%  3.34%  3.45%  2.29%  1.42%
##    29:   2.27%  1.21%  3.38%  3.85%  2.14%  1.37%
##    30:   2.25%  1.30%  3.43%  3.55%  2.09%  1.42%
##    31:   2.22%  1.27%  3.38%  3.45%  2.09%  1.46%
##    32:   2.15%  1.36%  3.08%  3.21%  2.24%  1.32%
##    33:   2.15%  1.36%  3.16%  3.21%  2.04%  1.42%
##    34:   2.16%  1.50%  3.08%  3.26%  1.99%  1.32%
##    35:   2.08%  1.30%  3.34%  3.11%  1.78%  1.28%
##    36:   2.04%  1.27%  2.94%  3.26%  1.83%  1.32%
##    37:   2.11%  1.41%  3.12%  3.16%  1.94%  1.32%
##    38:   2.03%  1.27%  3.12%  3.11%  1.78%  1.28%
##    39:   2.03%  1.30%  3.21%  3.06%  1.73%  1.23%
##    40:   2.04%  1.24%  3.08%  3.06%  1.99%  1.28%
##    41:   2.11%  1.30%  3.16%  3.31%  2.09%  1.18%
##    42:   2.04%  1.24%  3.16%  3.01%  2.09%  1.13%
##    43:   2.16%  1.33%  3.16%  3.16%  2.34%  1.28%
##    44:   2.12%  1.21%  3.25%  3.21%  2.14%  1.32%
##    45:   2.12%  1.24%  3.30%  3.11%  2.19%  1.28%
##    46:   2.10%  1.36%  3.30%  2.96%  2.09%  1.18%
##    47:   2.05%  1.18%  3.30%  2.91%  1.94%  1.37%
##    48:   1.99%  1.18%  3.16%  2.86%  1.83%  1.32%
##    49:   2.01%  1.12%  3.25%  2.91%  1.99%  1.28%
##    50:   1.97%  1.12%  3.21%  2.71%  1.94%  1.32%
##    51:   2.00%  1.15%  3.21%  2.96%  1.94%  1.18%
##    52:   2.02%  1.18%  3.25%  2.86%  2.04%  1.23%
##    53:   2.03%  1.09%  3.43%  2.86%  1.99%  1.28%
##    54:   2.00%  1.06%  3.21%  2.91%  2.04%  1.28%
##    55:   1.91%  1.09%  3.21%  2.66%  1.78%  1.23%
##    56:   1.94%  1.03%  3.25%  2.81%  1.94%  1.13%
##    57:   1.96%  1.06%  3.25%  2.91%  1.88%  1.18%
##    58:   2.00%  1.03%  3.47%  2.96%  1.88%  1.13%
##    59:   1.94%  0.94%  3.30%  2.86%  1.88%  1.28%
##    60:   1.90%  0.97%  3.16%  2.81%  1.83%  1.23%
##    61:   1.87%  0.94%  3.16%  2.81%  1.68%  1.23%
##    62:   1.89%  0.94%  3.25%  2.91%  1.78%  1.09%
##    63:   1.87%  0.94%  3.21%  2.57%  1.88%  1.23%
##    64:   1.84%  0.91%  3.08%  2.61%  1.94%  1.18%
##    65:   1.88%  0.94%  3.03%  2.91%  1.88%  1.13%
##    66:   1.89%  0.94%  3.03%  2.86%  1.94%  1.18%
##    67:   1.89%  0.97%  3.08%  2.86%  1.83%  1.23%
##    68:   1.92%  1.00%  3.25%  2.81%  1.88%  1.13%
##    69:   1.89%  1.03%  3.08%  2.96%  1.78%  1.09%
##    70:   1.86%  1.00%  3.12%  2.76%  1.73%  1.13%
##    71:   1.87%  1.06%  3.12%  2.71%  1.68%  1.18%
##    72:   1.83%  1.00%  2.99%  2.71%  1.78%  1.13%
##    73:   1.83%  1.03%  2.99%  2.76%  1.73%  1.09%
##    74:   1.80%  0.97%  3.03%  2.76%  1.63%  1.04%
##    75:   1.80%  1.06%  2.94%  2.71%  1.68%  0.99%
##    76:   1.83%  0.97%  3.03%  2.86%  1.68%  1.04%
##    77:   1.83%  1.03%  3.03%  2.86%  1.73%  0.94%
##    78:   1.78%  1.06%  2.86%  2.76%  1.68%  0.94%
##    79:   1.83%  1.03%  3.08%  2.66%  1.78%  0.99%
##    80:   1.77%  0.91%  3.03%  2.66%  1.73%  0.94%
##    81:   1.77%  0.94%  2.94%  2.71%  1.63%  1.04%
##    82:   1.77%  1.00%  2.94%  2.66%  1.63%  0.99%
##    83:   1.82%  1.06%  3.03%  2.61%  1.68%  1.09%
##    84:   1.84%  1.06%  3.12%  2.76%  1.63%  1.04%
##    85:   1.79%  0.94%  3.12%  2.66%  1.68%  0.99%
##    86:   1.81%  0.94%  3.08%  2.71%  1.73%  1.04%
##    87:   1.79%  0.94%  3.16%  2.66%  1.63%  0.99%
##    88:   1.77%  0.88%  3.08%  2.66%  1.63%  1.09%
##    89:   1.78%  0.94%  3.03%  2.61%  1.68%  1.09%
##    90:   1.84%  1.09%  3.12%  2.66%  1.63%  1.09%
##    91:   1.83%  1.00%  3.08%  2.71%  1.73%  1.04%
##    92:   1.81%  1.06%  2.90%  2.76%  1.73%  0.99%
##    93:   1.77%  1.00%  2.94%  2.61%  1.68%  1.04%
##    94:   1.80%  1.00%  2.90%  2.81%  1.68%  1.04%
##    95:   1.79%  0.97%  2.94%  2.61%  1.73%  1.13%
##    96:   1.77%  1.03%  2.99%  2.61%  1.68%  0.94%
##    97:   1.77%  1.06%  3.08%  2.42%  1.73%  0.94%
##    98:   1.74%  1.12%  2.90%  2.42%  1.68%  0.90%
##    99:   1.71%  0.94%  2.90%  2.42%  1.73%  0.94%
##   100:   1.74%  0.94%  2.99%  2.57%  1.68%  0.94%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.79%  4.72% 11.06% 10.37%  8.59%  5.61%
##     2:   7.15%  4.28%  9.14% 10.01%  8.17%  5.82%
##     3:   7.03%  4.10%  8.69% 10.43%  7.52%  6.09%
##     4:   6.85%  4.20%  9.21%  9.28%  7.41%  5.65%
##     5:   6.58%  4.26%  8.59%  9.38%  6.79%  5.20%
##     6:   5.83%  3.58%  7.78%  8.28%  5.94%  4.90%
##     7:   5.56%  3.44%  7.69%  7.63%  6.16%  4.14%
##     8:   5.08%  3.08%  6.62%  7.49%  5.92%  3.54%
##     9:   4.96%  3.44%  6.16%  7.18%  5.43%  3.55%
##    10:   4.56%  2.92%  6.07%  6.47%  5.33%  3.01%
##    11:   4.22%  2.67%  5.74%  5.96%  4.90%  2.81%
##    12:   4.08%  2.48%  5.95%  5.74%  4.83%  2.33%
##    13:   3.75%  2.30%  5.50%  5.53%  4.10%  2.13%
##    14:   3.56%  2.30%  5.15%  5.18%  4.15%  1.75%
##    15:   3.23%  1.86%  4.84%  4.79%  3.83%  1.65%
##    16:   3.17%  2.00%  4.44%  4.34%  3.88%  1.89%
##    17:   3.08%  1.92%  4.31%  4.54%  3.62%  1.70%
##    18:   3.04%  1.77%  4.17%  4.49%  3.77%  1.80%
##    19:   2.92%  1.80%  3.95%  4.44%  3.41%  1.70%
##    20:   2.83%  1.53%  4.00%  4.54%  3.21%  1.65%
##    21:   2.73%  1.56%  3.95%  4.05%  3.06%  1.75%
##    22:   2.62%  1.77%  3.43%  4.05%  2.85%  1.51%
##    23:   2.60%  1.65%  3.65%  4.05%  2.80%  1.42%
##    24:   2.54%  1.65%  3.34%  4.09%  2.70%  1.46%
##    25:   2.44%  1.53%  3.34%  3.70%  2.70%  1.46%
##    26:   2.43%  1.59%  3.34%  3.55%  2.70%  1.46%
##    27:   2.39%  1.41%  3.08%  3.80%  2.70%  1.61%
##    28:   2.39%  1.44%  3.38%  3.70%  2.55%  1.46%
##    29:   2.40%  1.44%  3.43%  3.60%  2.70%  1.42%
##    30:   2.39%  1.47%  3.21%  3.65%  2.70%  1.46%
##    31:   2.38%  1.50%  3.38%  3.60%  2.60%  1.32%
##    32:   2.31%  1.50%  3.34%  3.35%  2.60%  1.23%
##    33:   2.32%  1.47%  3.43%  3.60%  2.45%  1.13%
##    34:   2.29%  1.50%  3.47%  3.45%  2.29%  1.18%
##    35:   2.17%  1.36%  3.47%  3.21%  2.24%  1.04%
##    36:   2.19%  1.39%  3.25%  3.45%  2.24%  1.09%
##    37:   2.22%  1.47%  3.30%  3.40%  2.34%  1.04%
##    38:   2.14%  1.39%  3.34%  3.16%  2.14%  1.09%
##    39:   2.22%  1.41%  3.34%  3.35%  2.29%  1.13%
##    40:   2.15%  1.39%  3.16%  3.35%  2.29%  0.99%
##    41:   2.07%  1.30%  2.99%  3.40%  2.09%  1.04%
##    42:   2.11%  1.27%  3.03%  3.45%  2.19%  1.09%
##    43:   2.06%  1.18%  3.03%  3.45%  2.04%  1.09%
##    44:   2.06%  1.24%  3.03%  3.45%  2.09%  0.99%
##    45:   2.11%  1.30%  3.16%  3.31%  2.14%  1.09%
##    46:   2.12%  1.30%  3.16%  3.40%  2.24%  0.99%
##    47:   2.10%  1.27%  3.12%  3.35%  2.24%  0.99%
##    48:   2.06%  1.21%  3.03%  3.50%  2.14%  0.94%
##    49:   2.08%  1.21%  3.21%  3.40%  2.14%  0.94%
##    50:   2.00%  1.21%  3.03%  3.21%  2.14%  0.85%
##    51:   2.04%  1.18%  3.12%  3.31%  2.29%  0.80%
##    52:   2.00%  1.24%  3.08%  3.11%  2.14%  0.85%
##    53:   1.94%  1.18%  2.81%  3.06%  2.24%  0.85%
##    54:   1.94%  1.18%  2.90%  2.96%  2.14%  0.94%
##    55:   1.95%  1.12%  2.90%  3.21%  2.09%  0.94%
##    56:   1.93%  1.03%  2.94%  3.16%  2.04%  0.99%
##    57:   1.90%  1.09%  2.90%  3.01%  2.04%  0.94%
##    58:   1.89%  1.06%  2.94%  2.91%  1.99%  1.04%
##    59:   1.84%  1.09%  2.81%  2.81%  1.99%  0.94%
##    60:   1.90%  1.12%  2.86%  2.96%  2.09%  0.94%
##    61:   1.89%  1.18%  2.99%  2.96%  1.88%  0.85%
##    62:   1.86%  1.12%  2.94%  2.96%  1.83%  0.85%
##    63:   1.87%  1.15%  2.90%  2.86%  1.94%  0.90%
##    64:   1.88%  1.15%  2.90%  2.96%  1.99%  0.80%
##    65:   1.80%  1.09%  2.77%  2.81%  1.88%  0.85%
##    66:   1.86%  1.12%  2.86%  2.86%  1.94%  0.94%
##    67:   1.89%  1.09%  2.90%  2.91%  2.04%  0.94%
##    68:   1.86%  1.15%  2.90%  2.76%  1.94%  0.94%
##    69:   1.83%  1.00%  3.03%  2.86%  1.88%  0.85%
##    70:   1.83%  1.06%  2.94%  2.71%  1.94%  0.90%
##    71:   1.80%  1.09%  2.81%  2.86%  1.78%  0.85%
##    72:   1.79%  1.00%  2.86%  2.86%  1.78%  0.90%
##    73:   1.80%  1.00%  2.90%  2.91%  1.83%  0.80%
##    74:   1.79%  0.97%  2.94%  2.96%  1.78%  0.76%
##    75:   1.80%  1.00%  2.94%  2.96%  1.78%  0.76%
##    76:   1.77%  1.00%  2.86%  2.86%  1.83%  0.76%
##    77:   1.72%  0.94%  2.81%  2.71%  1.83%  0.71%
##    78:   1.81%  1.03%  2.86%  2.91%  1.94%  0.76%
##    79:   1.76%  1.00%  2.90%  2.71%  1.88%  0.71%
##    80:   1.76%  1.06%  2.81%  2.71%  1.83%  0.76%
##    81:   1.77%  0.97%  2.90%  2.81%  1.83%  0.80%
##    82:   1.75%  1.00%  2.90%  2.71%  1.78%  0.76%
##    83:   1.72%  0.97%  2.81%  2.76%  1.73%  0.71%
##    84:   1.75%  0.97%  2.81%  2.86%  1.83%  0.71%
##    85:   1.72%  0.94%  2.77%  2.86%  1.78%  0.66%
##    86:   1.77%  1.06%  2.94%  2.71%  1.73%  0.76%
##    87:   1.78%  1.03%  2.86%  2.81%  1.78%  0.85%
##    88:   1.73%  0.91%  2.86%  2.81%  1.78%  0.76%
##    89:   1.71%  0.91%  2.77%  2.81%  1.73%  0.76%
##    90:   1.69%  0.85%  2.77%  2.76%  1.78%  0.76%
##    91:   1.73%  1.00%  2.86%  2.81%  1.73%  0.66%
##    92:   1.67%  1.00%  2.77%  2.66%  1.68%  0.61%
##    93:   1.68%  0.94%  2.68%  2.81%  1.73%  0.66%
##    94:   1.68%  0.91%  2.68%  2.76%  1.78%  0.71%
##    95:   1.69%  0.97%  2.72%  2.76%  1.73%  0.66%
##    96:   1.70%  0.97%  2.72%  2.76%  1.73%  0.71%
##    97:   1.71%  0.94%  2.77%  2.76%  1.78%  0.71%
##    98:   1.70%  0.91%  2.77%  2.81%  1.68%  0.76%
##    99:   1.68%  0.91%  2.77%  2.81%  1.63%  0.71%
##   100:   1.69%  0.91%  2.68%  2.86%  1.63%  0.80%
## ntree      OOB      1      2      3      4      5
##     1:  11.12%  7.45% 16.17% 14.86%  8.46%  9.85%
##     2:  10.89%  7.27% 15.50% 13.01% 10.46%  9.75%
##     3:   9.99%  6.53% 13.01% 12.57% 10.67%  8.92%
##     4:   9.61%  6.38% 12.76% 12.24% 10.33%  8.08%
##     5:   8.78%  5.65% 12.30% 11.22%  9.38%  7.00%
##     6:   7.88%  5.39% 10.34%  9.74%  8.73%  6.64%
##     7:   7.04%  4.79%  9.30%  8.88%  8.32%  5.29%
##     8:   6.32%  4.08%  8.44%  8.09%  7.25%  5.02%
##     9:   5.95%  3.71%  7.95%  7.85%  6.83%  4.70%
##    10:   5.31%  3.30%  7.47%  7.12%  5.56%  4.16%
##    11:   5.01%  3.53%  6.92%  6.36%  5.60%  3.45%
##    12:   4.47%  3.04%  6.61%  5.89%  4.90%  2.67%
##    13:   3.92%  2.89%  5.52%  5.54%  3.89%  2.30%
##    14:   3.79%  2.73%  5.30%  5.33%  3.84%  2.34%
##    15:   3.64%  2.61%  5.09%  4.79%  3.88%  2.38%
##    16:   3.42%  2.22%  5.04%  4.59%  3.78%  2.15%
##    17:   3.19%  2.34%  4.31%  4.39%  3.36%  2.06%
##    18:   3.19%  2.19%  4.44%  4.44%  3.83%  1.65%
##    19:   2.97%  1.95%  4.39%  4.14%  3.57%  1.42%
##    20:   2.79%  1.83%  4.05%  4.09%  3.31%  1.28%
##    21:   2.68%  1.71%  3.79%  3.89%  3.15%  1.42%
##    22:   2.56%  1.59%  3.49%  4.09%  2.83%  1.37%
##    23:   2.51%  1.56%  3.23%  4.09%  2.94%  1.33%
##    24:   2.43%  1.50%  3.53%  3.79%  2.73%  1.14%
##    25:   2.35%  1.50%  3.44%  3.79%  2.57%  0.96%
##    26:   2.23%  1.59%  2.88%  3.79%  2.31%  1.01%
##    27:   2.13%  1.53%  2.80%  3.55%  2.26%  0.92%
##    28:   2.06%  1.38%  2.84%  3.40%  2.15%  0.96%
##    29:   2.00%  1.29%  2.62%  3.50%  2.15%  0.87%
##    30:   2.00%  1.38%  2.67%  3.20%  2.20%  0.96%
##    31:   1.82%  1.26%  2.62%  2.86%  1.94%  0.73%
##    32:   1.88%  1.35%  2.62%  2.81%  2.05%  0.87%
##    33:   1.86%  1.20%  2.80%  2.91%  1.89%  0.87%
##    34:   1.78%  1.17%  2.50%  2.91%  1.94%  0.78%
##    35:   1.80%  1.02%  2.62%  2.96%  1.99%  0.87%
##    36:   1.67%  1.05%  2.45%  2.76%  1.78%  0.69%
##    37:   1.68%  1.02%  2.28%  2.61%  2.15%  0.78%
##    38:   1.72%  0.99%  2.41%  2.86%  1.99%  0.78%
##    39:   1.66%  0.99%  2.28%  2.76%  1.94%  0.73%
##    40:   1.66%  0.99%  2.45%  2.71%  1.89%  0.69%
##    41:   1.56%  0.90%  2.07%  2.66%  1.89%  0.73%
##    42:   1.59%  0.96%  2.28%  2.56%  1.89%  0.64%
##    43:   1.55%  0.93%  2.24%  2.51%  1.78%  0.64%
##    44:   1.54%  0.90%  2.15%  2.37%  2.05%  0.64%
##    45:   1.51%  0.99%  1.94%  2.51%  1.89%  0.59%
##    46:   1.47%  0.87%  1.89%  2.51%  1.89%  0.59%
##    47:   1.50%  0.90%  2.02%  2.51%  1.84%  0.64%
##    48:   1.46%  0.93%  1.89%  2.41%  1.84%  0.59%
##    49:   1.54%  0.96%  2.11%  2.51%  1.89%  0.59%
##    50:   1.48%  0.93%  2.02%  2.37%  1.73%  0.69%
##    51:   1.46%  0.93%  1.94%  2.32%  1.89%  0.59%
##    52:   1.44%  0.99%  1.94%  2.27%  1.63%  0.64%
##    53:   1.40%  0.93%  1.89%  2.07%  1.78%  0.64%
##    54:   1.39%  0.84%  1.98%  2.07%  1.73%  0.69%
##    55:   1.44%  0.87%  2.15%  2.12%  1.78%  0.64%
##    56:   1.45%  0.87%  2.07%  2.27%  1.78%  0.64%
##    57:   1.42%  0.84%  1.94%  2.27%  1.78%  0.64%
##    58:   1.42%  0.75%  2.02%  2.41%  1.73%  0.59%
##    59:   1.39%  0.72%  2.11%  2.22%  1.73%  0.59%
##    60:   1.39%  0.72%  2.02%  2.22%  1.84%  0.59%
##    61:   1.38%  0.75%  2.07%  2.12%  1.84%  0.55%
##    62:   1.41%  0.72%  2.11%  2.27%  1.84%  0.55%
##    63:   1.36%  0.72%  1.94%  2.17%  1.84%  0.55%
##    64:   1.38%  0.81%  2.02%  2.17%  1.78%  0.50%
##    65:   1.36%  0.75%  1.98%  2.12%  1.84%  0.50%
##    66:   1.39%  0.81%  1.98%  2.12%  1.78%  0.64%
##    67:   1.38%  0.72%  2.02%  2.17%  1.78%  0.59%
##    68:   1.34%  0.69%  1.94%  2.17%  1.78%  0.55%
##    69:   1.27%  0.66%  1.94%  1.97%  1.63%  0.55%
##    70:   1.27%  0.69%  1.81%  1.97%  1.73%  0.55%
##    71:   1.29%  0.81%  1.81%  1.92%  1.68%  0.55%
##    72:   1.32%  0.81%  1.98%  2.02%  1.57%  0.55%
##    73:   1.31%  0.78%  1.94%  1.97%  1.63%  0.55%
##    74:   1.34%  0.81%  1.89%  1.97%  1.84%  0.55%
##    75:   1.28%  0.72%  1.94%  1.87%  1.73%  0.50%
##    76:   1.30%  0.72%  1.98%  1.87%  1.73%  0.55%
##    77:   1.26%  0.72%  1.89%  1.77%  1.68%  0.55%
##    78:   1.27%  0.66%  1.98%  1.72%  1.78%  0.55%
##    79:   1.21%  0.60%  1.85%  1.77%  1.63%  0.55%
##    80:   1.23%  0.60%  1.94%  1.82%  1.68%  0.50%
##    81:   1.21%  0.66%  1.76%  1.72%  1.78%  0.50%
##    82:   1.24%  0.66%  1.89%  1.72%  1.78%  0.50%
##    83:   1.21%  0.60%  1.85%  1.82%  1.68%  0.50%
##    84:   1.23%  0.63%  1.85%  1.87%  1.68%  0.50%
##    85:   1.26%  0.69%  1.89%  1.82%  1.73%  0.50%
##    86:   1.22%  0.63%  1.85%  1.82%  1.68%  0.50%
##    87:   1.26%  0.63%  1.89%  1.87%  1.78%  0.50%
##    88:   1.22%  0.63%  1.85%  1.82%  1.68%  0.50%
##    89:   1.22%  0.63%  1.76%  1.87%  1.73%  0.50%
##    90:   1.21%  0.63%  1.76%  1.92%  1.68%  0.46%
##    91:   1.24%  0.69%  1.72%  1.82%  1.78%  0.55%
##    92:   1.25%  0.66%  1.76%  1.87%  1.78%  0.55%
##    93:   1.22%  0.63%  1.76%  1.87%  1.73%  0.50%
##    94:   1.20%  0.60%  1.64%  1.82%  1.78%  0.55%
##    95:   1.23%  0.57%  1.72%  1.92%  1.84%  0.55%
##    96:   1.23%  0.60%  1.81%  1.82%  1.78%  0.55%
##    97:   1.21%  0.63%  1.76%  1.72%  1.78%  0.50%
##    98:   1.26%  0.69%  1.81%  1.82%  1.78%  0.55%
##    99:   1.21%  0.72%  1.68%  1.82%  1.68%  0.46%
##   100:   1.21%  0.69%  1.72%  1.72%  1.73%  0.50%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   6.76%  4.44% 10.06%  9.96%  5.87%  4.52%
##     2:   7.49%  4.93% 10.71%  9.89%  8.20%  5.21%
##     3:   7.35%  4.85% 10.96%  9.55%  7.55%  5.15%
##     4:   7.41%  4.89% 10.34%  9.38%  7.72%  6.00%
##     5:   7.02%  4.51%  9.61%  9.34%  7.19%  5.80%
##     6:   6.51%  3.98%  9.65%  8.34%  6.45%  5.40%
##     7:   6.10%  4.04%  9.04%  8.14%  5.46%  4.79%
##     8:   5.45%  3.80%  7.95%  6.92%  5.34%  4.07%
##     9:   5.02%  3.25%  6.82%  7.17%  5.01%  3.81%
##    10:   4.63%  3.09%  6.70%  6.43%  4.46%  3.29%
##    11:   4.09%  3.02%  5.63%  5.85%  4.07%  2.45%
##    12:   3.89%  2.68%  5.71%  5.39%  4.26%  2.07%
##    13:   3.57%  2.32%  5.18%  5.09%  3.99%  1.98%
##    14:   3.42%  2.22%  4.96%  4.93%  3.78%  1.88%
##    15:   3.38%  2.28%  5.04%  4.68%  3.68%  1.83%
##    16:   3.16%  1.92%  4.69%  4.54%  3.68%  1.70%
##    17:   3.06%  1.80%  4.48%  4.34%  3.41%  1.97%
##    18:   2.99%  1.71%  4.60%  4.14%  3.57%  1.65%
##    19:   2.72%  1.68%  4.04%  3.75%  3.41%  1.33%
##    20:   2.75%  1.56%  4.26%  3.60%  3.31%  1.69%
##    21:   2.79%  1.86%  4.04%  3.90%  3.36%  1.37%
##    22:   2.67%  1.74%  4.04%  3.79%  2.94%  1.37%
##    23:   2.68%  1.56%  4.17%  3.75%  3.04%  1.51%
##    24:   2.68%  1.83%  3.79%  3.75%  2.99%  1.56%
##    25:   2.55%  1.50%  3.79%  3.65%  2.94%  1.46%
##    26:   2.45%  1.41%  3.83%  3.35%  2.73%  1.51%
##    27:   2.48%  1.56%  3.57%  3.40%  2.99%  1.42%
##    28:   2.36%  1.50%  3.31%  3.55%  2.68%  1.28%
##    29:   2.34%  1.56%  3.44%  3.35%  2.57%  1.24%
##    30:   2.25%  1.53%  3.31%  3.10%  2.68%  1.05%
##    31:   2.25%  1.50%  3.40%  3.10%  2.41%  1.24%
##    32:   2.22%  1.53%  3.27%  3.01%  2.62%  1.10%
##    33:   2.22%  1.41%  3.27%  3.20%  2.57%  1.14%
##    34:   2.22%  1.44%  3.31%  3.20%  2.47%  1.10%
##    35:   2.20%  1.41%  3.31%  2.91%  2.62%  1.19%
##    36:   2.28%  1.44%  3.23%  3.35%  2.89%  1.05%
##    37:   2.22%  1.47%  3.14%  3.06%  2.78%  1.14%
##    38:   2.12%  1.41%  3.14%  2.91%  2.41%  1.14%
##    39:   2.13%  1.41%  3.18%  2.91%  2.41%  1.14%
##    40:   2.16%  1.35%  3.18%  2.91%  2.68%  1.14%
##    41:   2.13%  1.35%  3.14%  3.01%  2.57%  1.05%
##    42:   2.10%  1.35%  3.06%  2.96%  2.52%  1.05%
##    43:   2.08%  1.29%  3.06%  2.96%  2.57%  1.01%
##    44:   2.06%  1.20%  3.14%  2.96%  2.47%  1.01%
##    45:   2.10%  1.14%  3.01%  3.10%  2.73%  1.10%
##    46:   2.02%  1.14%  2.93%  2.91%  2.62%  1.05%
##    47:   1.98%  1.11%  2.84%  2.76%  2.73%  1.01%
##    48:   2.00%  1.14%  2.93%  2.86%  2.52%  1.05%
##    49:   2.00%  1.11%  2.93%  2.86%  2.57%  1.05%
##    50:   1.92%  1.02%  2.80%  2.71%  2.57%  1.05%
##    51:   1.94%  1.05%  2.88%  2.76%  2.52%  1.05%
##    52:   1.93%  0.99%  2.93%  2.76%  2.47%  1.05%
##    53:   1.94%  1.08%  2.80%  2.91%  2.57%  0.92%
##    54:   1.96%  1.11%  2.84%  2.86%  2.52%  1.01%
##    55:   1.89%  1.02%  2.80%  2.76%  2.52%  0.92%
##    56:   1.90%  0.99%  2.84%  2.76%  2.52%  0.96%
##    57:   1.84%  0.87%  2.75%  2.86%  2.47%  0.87%
##    58:   1.89%  0.93%  2.93%  2.71%  2.57%  0.92%
##    59:   1.85%  1.02%  2.88%  2.61%  2.36%  0.87%
##    60:   1.89%  0.99%  2.84%  2.61%  2.68%  0.87%
##    61:   1.89%  1.02%  2.88%  2.56%  2.62%  0.87%
##    62:   1.94%  0.96%  2.97%  2.76%  2.73%  0.92%
##    63:   1.94%  0.99%  2.97%  2.81%  2.57%  0.92%
##    64:   1.93%  1.02%  2.88%  2.76%  2.62%  0.92%
##    65:   1.89%  0.99%  2.80%  2.66%  2.68%  0.87%
##    66:   1.91%  1.02%  2.80%  2.76%  2.62%  0.92%
##    67:   1.91%  0.99%  2.84%  2.86%  2.57%  0.87%
##    68:   1.89%  0.99%  2.80%  2.86%  2.52%  0.82%
##    69:   1.87%  0.99%  2.67%  2.86%  2.52%  0.87%
##    70:   1.87%  0.90%  2.75%  2.81%  2.68%  0.82%
##    71:   1.86%  0.93%  2.67%  2.81%  2.62%  0.87%
##    72:   1.90%  0.99%  2.67%  2.81%  2.68%  0.96%
##    73:   1.86%  0.99%  2.71%  2.81%  2.52%  0.82%
##    74:   1.83%  1.05%  2.54%  2.76%  2.52%  0.82%
##    75:   1.83%  0.99%  2.58%  2.71%  2.47%  0.92%
##    76:   1.83%  0.99%  2.54%  2.76%  2.57%  0.87%
##    77:   1.83%  1.02%  2.41%  2.66%  2.73%  0.87%
##    78:   1.82%  1.05%  2.28%  2.81%  2.57%  0.92%
##    79:   1.87%  1.05%  2.50%  2.81%  2.68%  0.87%
##    80:   1.89%  1.02%  2.62%  2.86%  2.68%  0.87%
##    81:   1.84%  0.99%  2.58%  2.81%  2.57%  0.82%
##    82:   1.82%  1.05%  2.58%  2.61%  2.47%  0.87%
##    83:   1.81%  0.99%  2.58%  2.81%  2.41%  0.78%
##    84:   1.82%  0.99%  2.62%  2.71%  2.41%  0.87%
##    85:   1.79%  0.96%  2.58%  2.76%  2.36%  0.82%
##    86:   1.77%  0.99%  2.54%  2.66%  2.36%  0.82%
##    87:   1.79%  0.99%  2.62%  2.71%  2.31%  0.82%
##    88:   1.79%  0.99%  2.67%  2.76%  2.26%  0.78%
##    89:   1.76%  0.99%  2.54%  2.71%  2.36%  0.69%
##    90:   1.76%  0.96%  2.54%  2.71%  2.31%  0.78%
##    91:   1.77%  0.96%  2.54%  2.66%  2.41%  0.78%
##    92:   1.75%  0.96%  2.54%  2.66%  2.31%  0.78%
##    93:   1.74%  0.99%  2.54%  2.56%  2.36%  0.73%
##    94:   1.75%  0.96%  2.58%  2.61%  2.36%  0.73%
##    95:   1.76%  0.93%  2.58%  2.56%  2.57%  0.69%
##    96:   1.74%  1.02%  2.45%  2.51%  2.47%  0.73%
##    97:   1.72%  0.99%  2.37%  2.56%  2.36%  0.82%
##    98:   1.75%  0.99%  2.37%  2.56%  2.52%  0.82%
##    99:   1.75%  0.96%  2.41%  2.66%  2.41%  0.82%
##   100:   1.76%  0.96%  2.41%  2.66%  2.52%  0.78%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.95%  3.81% 15.09%  9.09%  8.01%  5.78%
##     2:   7.95%  4.83% 11.41%  9.54%  8.99%  6.73%
##     3:   7.86%  4.40% 10.97%  9.77%  9.43%  6.81%
##     4:   7.68%  4.34% 11.25% 10.08%  8.29%  6.29%
##     5:   6.96%  4.27%  9.49%  9.43%  7.69%  5.50%
##     6:   6.47%  3.99%  9.12%  8.17%  7.40%  5.07%
##     7:   6.09%  3.58%  8.93%  8.00%  7.01%  4.35%
##     8:   5.70%  3.27%  8.66%  7.63%  6.33%  3.93%
##     9:   5.38%  3.21%  7.48%  7.59%  6.11%  3.76%
##    10:   5.01%  2.95%  7.22%  6.99%  5.39%  3.65%
##    11:   4.58%  2.57%  6.56%  6.51%  5.38%  3.04%
##    12:   4.32%  2.38%  6.28%  6.04%  4.74%  3.22%
##    13:   3.91%  2.26%  5.27%  5.89%  4.42%  2.71%
##    14:   3.77%  2.23%  5.52%  5.09%  4.52%  2.38%
##    15:   3.56%  1.95%  4.91%  5.23%  4.25%  2.43%
##    16:   3.40%  2.19%  4.44%  4.93%  3.99%  2.20%
##    17:   3.23%  2.07%  4.31%  4.69%  3.83%  1.97%
##    18:   3.22%  2.04%  4.44%  4.44%  3.93%  1.97%
##    19:   3.04%  1.86%  4.18%  4.19%  3.57%  2.11%
##    20:   2.81%  1.80%  3.44%  3.89%  3.52%  2.06%
##    21:   2.75%  1.71%  3.92%  3.60%  3.36%  1.78%
##    22:   2.56%  1.62%  3.49%  3.50%  3.10%  1.69%
##    23:   2.69%  1.74%  3.61%  3.60%  3.31%  1.78%
##    24:   2.62%  1.68%  3.44%  3.50%  3.20%  1.83%
##    25:   2.56%  1.62%  3.66%  3.30%  3.10%  1.65%
##    26:   2.53%  1.59%  3.61%  3.25%  3.36%  1.42%
##    27:   2.45%  1.65%  3.31%  3.10%  2.99%  1.65%
##    28:   2.37%  1.62%  3.23%  3.15%  2.78%  1.51%
##    29:   2.36%  1.44%  3.27%  3.25%  2.83%  1.56%
##    30:   2.36%  1.47%  3.53%  3.15%  2.73%  1.42%
##    31:   2.37%  1.62%  3.40%  2.96%  2.78%  1.51%
##    32:   2.33%  1.50%  3.49%  3.01%  2.78%  1.33%
##    33:   2.24%  1.32%  3.23%  2.91%  2.83%  1.46%
##    34:   2.18%  1.32%  3.18%  2.81%  2.73%  1.37%
##    35:   2.23%  1.47%  3.18%  2.86%  2.83%  1.28%
##    36:   2.19%  1.38%  3.18%  2.66%  2.99%  1.24%
##    37:   2.17%  1.32%  3.23%  2.66%  2.99%  1.19%
##    38:   2.11%  1.20%  3.23%  2.51%  2.89%  1.28%
##    39:   2.15%  1.35%  3.27%  2.51%  2.94%  1.14%
##    40:   2.16%  1.29%  3.23%  2.66%  2.99%  1.14%
##    41:   2.03%  1.23%  3.10%  2.51%  2.73%  1.05%
##    42:   2.04%  1.14%  3.06%  2.66%  2.89%  1.01%
##    43:   2.06%  1.20%  3.14%  2.71%  2.89%  0.92%
##    44:   2.00%  1.14%  3.01%  2.61%  2.73%  1.01%
##    45:   2.00%  1.11%  3.23%  2.41%  2.73%  1.01%
##    46:   2.00%  1.14%  3.10%  2.61%  2.68%  0.96%
##    47:   2.00%  1.14%  3.01%  2.61%  2.73%  1.05%
##    48:   2.01%  1.20%  3.06%  2.46%  2.78%  1.05%
##    49:   1.98%  1.17%  3.01%  2.51%  2.78%  0.92%
##    50:   2.00%  1.26%  2.97%  2.51%  2.83%  0.92%
##    51:   2.00%  1.17%  2.97%  2.61%  2.83%  0.96%
##    52:   1.96%  1.14%  2.88%  2.66%  2.73%  0.92%
##    53:   2.00%  1.23%  2.97%  2.61%  2.73%  0.92%
##    54:   1.94%  1.23%  2.93%  2.46%  2.57%  0.96%
##    55:   1.94%  1.20%  2.71%  2.66%  2.68%  0.96%
##    56:   2.00%  1.20%  3.01%  2.61%  2.73%  0.92%
##    57:   1.97%  1.20%  3.01%  2.61%  2.57%  0.92%
##    58:   1.94%  1.20%  2.97%  2.51%  2.62%  0.82%
##    59:   1.90%  1.20%  2.97%  2.41%  2.52%  0.82%
##    60:   1.95%  1.26%  2.88%  2.66%  2.52%  0.87%
##    61:   1.96%  1.20%  2.93%  2.76%  2.57%  0.82%
##    62:   1.95%  1.17%  2.93%  2.71%  2.57%  0.87%
##    63:   1.87%  1.08%  2.84%  2.46%  2.57%  0.87%
##    64:   1.89%  1.17%  2.93%  2.51%  2.52%  0.78%
##    65:   1.89%  1.20%  2.93%  2.41%  2.52%  0.78%
##    66:   1.92%  1.17%  2.97%  2.61%  2.47%  0.82%
##    67:   1.92%  1.26%  2.80%  2.51%  2.62%  0.82%
##    68:   1.89%  1.20%  2.88%  2.46%  2.52%  0.78%
##    69:   1.91%  1.26%  2.93%  2.46%  2.52%  0.78%
##    70:   1.89%  1.14%  2.84%  2.56%  2.52%  0.82%
##    71:   1.86%  1.17%  2.84%  2.51%  2.47%  0.73%
##    72:   1.82%  1.11%  2.75%  2.56%  2.41%  0.69%
##    73:   1.83%  1.05%  2.80%  2.56%  2.41%  0.82%
##    74:   1.80%  1.11%  2.75%  2.41%  2.41%  0.73%
##    75:   1.83%  1.11%  2.80%  2.41%  2.41%  0.82%
##    76:   1.89%  1.11%  2.97%  2.56%  2.52%  0.73%
##    77:   1.83%  1.02%  2.88%  2.37%  2.47%  0.87%
##    78:   1.83%  1.08%  2.93%  2.37%  2.47%  0.73%
##    79:   1.77%  1.05%  2.75%  2.27%  2.47%  0.78%
##    80:   1.80%  1.08%  2.80%  2.41%  2.47%  0.69%
##    81:   1.86%  1.11%  2.88%  2.51%  2.62%  0.64%
##    82:   1.83%  1.05%  2.88%  2.46%  2.57%  0.69%
##    83:   1.83%  1.08%  2.75%  2.46%  2.57%  0.78%
##    84:   1.86%  1.14%  2.84%  2.41%  2.62%  0.73%
##    85:   1.87%  1.08%  2.97%  2.41%  2.57%  0.78%
##    86:   1.81%  1.05%  2.80%  2.41%  2.57%  0.69%
##    87:   1.83%  1.08%  2.80%  2.46%  2.62%  0.69%
##    88:   1.87%  1.05%  2.84%  2.46%  2.73%  0.78%
##    89:   1.83%  1.08%  2.80%  2.41%  2.57%  0.73%
##    90:   1.81%  0.99%  2.75%  2.46%  2.68%  0.69%
##    91:   1.78%  0.99%  2.71%  2.56%  2.52%  0.64%
##    92:   1.78%  0.99%  2.71%  2.46%  2.57%  0.69%
##    93:   1.76%  0.99%  2.62%  2.51%  2.52%  0.64%
##    94:   1.79%  0.99%  2.71%  2.51%  2.57%  0.69%
##    95:   1.82%  1.08%  2.71%  2.46%  2.62%  0.69%
##    96:   1.81%  1.05%  2.71%  2.56%  2.52%  0.69%
##    97:   1.82%  1.05%  2.71%  2.56%  2.57%  0.69%
##    98:   1.83%  1.08%  2.67%  2.56%  2.62%  0.69%
##    99:   1.79%  1.08%  2.58%  2.46%  2.68%  0.64%
##   100:   1.83%  1.11%  2.62%  2.56%  2.68%  0.64%
## ntree      OOB      1      2      3      4      5
##     1:  10.48%  7.58% 13.09% 12.52% 11.67%  9.14%
##     2:  10.05%  7.37% 12.55% 12.93% 10.70%  8.26%
##     3:   9.91%  7.24% 13.55% 12.74%  9.83%  7.65%
##     4:   9.42%  6.03% 12.77% 12.43% 10.35%  7.57%
##     5:   8.61%  6.29% 10.88% 10.63% 10.41%  6.41%
##     6:   7.69%  5.48% 10.25%  9.73%  8.44%  5.89%
##     7:   7.37%  5.33%  9.76%  8.49%  8.54%  5.97%
##     8:   6.17%  4.19%  8.28%  7.29%  7.41%  4.91%
##     9:   5.64%  4.04%  7.51%  6.62%  6.76%  4.27%
##    10:   5.37%  3.59%  8.04%  5.89%  6.44%  3.92%
##    11:   4.83%  3.31%  6.82%  5.73%  5.72%  3.50%
##    12:   4.52%  3.55%  6.06%  4.98%  5.11%  3.45%
##    13:   4.12%  3.12%  5.27%  4.87%  4.84%  3.12%
##    14:   3.72%  2.58%  4.90%  4.71%  4.68%  2.47%
##    15:   3.41%  2.17%  4.42%  4.86%  4.09%  2.33%
##    16:   3.10%  2.20%  4.20%  4.16%  3.77%  1.78%
##    17:   2.97%  2.08%  3.81%  3.92%  4.03%  1.69%
##    18:   2.68%  1.78%  3.68%  3.62%  3.39%  1.55%
##    19:   2.68%  1.81%  3.81%  3.52%  3.34%  1.51%
##    20:   2.65%  1.93%  3.63%  3.47%  3.44%  1.28%
##    21:   2.47%  1.63%  3.50%  3.38%  3.28%  1.14%
##    22:   2.50%  1.81%  3.28%  3.28%  3.44%  1.19%
##    23:   2.28%  1.51%  3.24%  2.98%  3.12%  1.10%
##    24:   2.19%  1.39%  3.19%  2.59%  3.12%  1.19%
##    25:   2.22%  1.51%  3.33%  2.74%  2.81%  1.14%
##    26:   2.03%  1.36%  2.98%  2.50%  2.81%  0.96%
##    27:   2.01%  1.42%  2.84%  2.74%  2.65%  0.82%
##    28:   1.90%  1.36%  2.89%  2.40%  2.54%  0.69%
##    29:   1.83%  1.25%  2.80%  2.54%  2.38%  0.59%
##    30:   1.83%  1.28%  2.80%  2.30%  2.49%  0.64%
##    31:   1.81%  1.25%  2.76%  2.40%  2.38%  0.64%
##    32:   1.72%  1.19%  2.80%  2.25%  2.12%  0.59%
##    33:   1.73%  1.10%  2.89%  2.15%  2.38%  0.55%
##    34:   1.69%  1.04%  2.84%  1.96%  2.38%  0.64%
##    35:   1.69%  0.98%  2.76%  2.35%  2.22%  0.59%
##    36:   1.66%  0.95%  2.76%  2.15%  2.33%  0.55%
##    37:   1.68%  1.01%  2.71%  2.15%  2.38%  0.59%
##    38:   1.55%  0.83%  2.67%  1.96%  2.28%  0.50%
##    39:   1.56%  0.89%  2.67%  1.91%  2.28%  0.50%
##    40:   1.53%  0.92%  2.49%  2.05%  2.12%  0.46%
##    41:   1.54%  0.95%  2.58%  1.86%  2.28%  0.41%
##    42:   1.54%  0.89%  2.54%  1.81%  2.33%  0.55%
##    43:   1.51%  0.80%  2.32%  2.15%  2.33%  0.46%
##    44:   1.58%  0.83%  2.67%  2.10%  2.33%  0.46%
##    45:   1.57%  0.92%  2.45%  2.01%  2.49%  0.46%
##    46:   1.54%  0.89%  2.36%  2.05%  2.38%  0.46%
##    47:   1.55%  0.95%  2.41%  1.96%  2.44%  0.41%
##    48:   1.52%  0.89%  2.32%  2.05%  2.38%  0.41%
##    49:   1.49%  0.89%  2.28%  1.86%  2.44%  0.46%
##    50:   1.57%  0.89%  2.36%  2.15%  2.44%  0.50%
##    51:   1.49%  0.89%  2.41%  1.76%  2.44%  0.41%
##    52:   1.49%  0.83%  2.41%  1.91%  2.38%  0.37%
##    53:   1.43%  0.74%  2.28%  1.96%  2.22%  0.41%
##    54:   1.47%  0.89%  2.19%  2.10%  2.28%  0.32%
##    55:   1.43%  0.83%  2.32%  1.81%  2.22%  0.37%
##    56:   1.43%  0.77%  2.32%  1.96%  2.17%  0.37%
##    57:   1.36%  0.80%  2.23%  1.71%  2.07%  0.37%
##    58:   1.33%  0.71%  2.23%  1.61%  2.07%  0.46%
##    59:   1.36%  0.71%  2.23%  1.71%  2.17%  0.41%
##    60:   1.37%  0.71%  2.32%  1.66%  2.07%  0.50%
##    61:   1.31%  0.80%  2.10%  1.52%  2.07%  0.41%
##    62:   1.38%  0.80%  2.28%  1.57%  2.07%  0.55%
##    63:   1.34%  0.68%  2.32%  1.47%  2.12%  0.55%
##    64:   1.34%  0.71%  2.23%  1.52%  2.17%  0.50%
##    65:   1.34%  0.65%  2.23%  1.52%  2.17%  0.59%
##    66:   1.35%  0.68%  2.14%  1.71%  2.07%  0.59%
##    67:   1.35%  0.68%  2.19%  1.76%  2.07%  0.50%
##    68:   1.30%  0.68%  2.14%  1.57%  2.12%  0.41%
##    69:   1.32%  0.71%  2.10%  1.66%  2.01%  0.50%
##    70:   1.31%  0.62%  2.23%  1.52%  2.07%  0.55%
##    71:   1.31%  0.62%  2.19%  1.57%  2.12%  0.50%
##    72:   1.30%  0.65%  2.14%  1.37%  2.22%  0.55%
##    73:   1.31%  0.62%  2.23%  1.47%  2.07%  0.59%
##    74:   1.31%  0.65%  2.19%  1.42%  2.17%  0.55%
##    75:   1.29%  0.71%  2.06%  1.42%  2.17%  0.50%
##    76:   1.30%  0.65%  2.23%  1.47%  2.12%  0.46%
##    77:   1.32%  0.68%  2.23%  1.47%  2.12%  0.50%
##    78:   1.32%  0.65%  2.23%  1.52%  2.17%  0.46%
##    79:   1.27%  0.59%  2.10%  1.42%  2.17%  0.55%
##    80:   1.31%  0.62%  2.10%  1.57%  2.22%  0.50%
##    81:   1.29%  0.59%  2.14%  1.57%  2.12%  0.50%
##    82:   1.32%  0.59%  2.28%  1.57%  2.12%  0.50%
##    83:   1.30%  0.62%  2.10%  1.52%  2.22%  0.50%
##    84:   1.27%  0.56%  2.10%  1.57%  2.12%  0.50%
##    85:   1.27%  0.53%  2.10%  1.61%  2.12%  0.50%
##    86:   1.28%  0.59%  2.14%  1.52%  2.12%  0.50%
##    87:   1.33%  0.56%  2.28%  1.57%  2.28%  0.50%
##    88:   1.35%  0.65%  2.28%  1.61%  2.17%  0.50%
##    89:   1.34%  0.65%  2.28%  1.57%  2.17%  0.50%
##    90:   1.34%  0.62%  2.23%  1.61%  2.22%  0.50%
##    91:   1.38%  0.65%  2.28%  1.71%  2.22%  0.55%
##    92:   1.31%  0.56%  2.19%  1.61%  2.17%  0.50%
##    93:   1.34%  0.56%  2.14%  1.66%  2.28%  0.59%
##    94:   1.33%  0.56%  2.19%  1.66%  2.17%  0.59%
##    95:   1.32%  0.47%  2.28%  1.71%  2.07%  0.59%
##    96:   1.30%  0.50%  2.19%  1.76%  2.01%  0.55%
##    97:   1.27%  0.44%  2.14%  1.66%  2.07%  0.55%
##    98:   1.32%  0.53%  2.19%  1.76%  2.07%  0.55%
##    99:   1.30%  0.47%  2.28%  1.61%  2.07%  0.59%
##   100:   1.32%  0.56%  2.19%  1.66%  2.07%  0.59%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.03%  5.03%  8.70%  9.32%  7.77%  5.62%
##     2:   7.30%  5.18%  9.79%  8.93%  7.85%  6.09%
##     3:   7.04%  4.74%  9.22%  9.31%  8.10%  5.34%
##     4:   6.46%  4.13%  8.62%  8.45%  8.05%  4.61%
##     5:   6.04%  3.74%  8.07%  8.26%  7.62%  4.08%
##     6:   5.33%  3.29%  7.35%  7.06%  6.21%  4.03%
##     7:   5.12%  3.00%  6.86%  7.41%  6.32%  3.41%
##     8:   4.75%  2.84%  6.38%  6.78%  5.73%  3.26%
##     9:   4.49%  2.83%  5.65%  6.95%  5.65%  2.54%
##    10:   4.20%  2.69%  5.53%  6.32%  4.86%  2.58%
##    11:   4.10%  2.98%  5.46%  5.66%  4.97%  2.20%
##    12:   3.81%  2.47%  5.18%  5.70%  4.36%  2.20%
##    13:   3.50%  2.32%  4.91%  5.34%  3.72%  1.92%
##    14:   3.31%  2.44%  4.56%  4.80%  3.72%  1.60%
##    15:   3.20%  2.29%  4.51%  4.50%  3.77%  1.51%
##    16:   3.05%  2.26%  4.68%  4.01%  3.34%  1.42%
##    17:   2.86%  2.14%  4.38%  3.62%  2.97%  1.60%
##    18:   2.84%  2.08%  4.11%  3.96%  3.02%  1.46%
##    19:   2.74%  2.14%  3.89%  3.77%  3.07%  1.23%
##    20:   2.69%  1.96%  4.07%  3.57%  3.02%  1.28%
##    21:   2.50%  1.93%  3.72%  3.18%  2.97%  1.05%
##    22:   2.34%  1.78%  3.24%  3.08%  2.75%  1.23%
##    23:   2.38%  1.72%  3.76%  2.84%  2.97%  1.01%
##    24:   2.42%  1.87%  3.68%  2.89%  2.97%  1.05%
##    25:   2.49%  1.84%  3.46%  3.18%  3.28%  1.14%
##    26:   2.43%  1.72%  3.33%  3.23%  3.18%  1.19%
##    27:   2.30%  1.75%  3.15%  2.89%  3.07%  1.05%
##    28:   2.30%  1.78%  3.15%  3.08%  2.81%  1.05%
##    29:   2.39%  1.75%  3.46%  3.08%  3.07%  1.05%
##    30:   2.23%  1.69%  3.19%  2.74%  2.86%  1.05%
##    31:   2.09%  1.57%  3.19%  2.59%  2.54%  0.87%
##    32:   2.12%  1.54%  3.37%  2.54%  2.60%  0.91%
##    33:   2.17%  1.51%  3.63%  2.50%  2.54%  1.05%
##    34:   2.13%  1.54%  3.24%  2.84%  2.60%  0.82%
##    35:   2.12%  1.45%  3.24%  2.59%  2.75%  1.01%
##    36:   2.21%  1.63%  3.28%  2.64%  2.91%  0.96%
##    37:   2.11%  1.63%  3.02%  2.54%  2.70%  1.01%
##    38:   2.06%  1.60%  2.80%  2.54%  2.70%  0.96%
##    39:   2.06%  1.60%  2.84%  2.25%  2.91%  1.05%
##    40:   2.03%  1.60%  3.15%  2.15%  2.65%  0.87%
##    41:   2.03%  1.48%  3.02%  2.35%  2.65%  1.01%
##    42:   2.11%  1.54%  3.15%  2.25%  2.97%  1.01%
##    43:   2.05%  1.45%  3.02%  2.35%  2.81%  1.01%
##    44:   2.02%  1.45%  3.06%  2.30%  2.65%  1.01%
##    45:   2.10%  1.51%  3.15%  2.40%  2.75%  1.05%
##    46:   2.06%  1.45%  3.11%  2.30%  2.81%  1.05%
##    47:   2.11%  1.45%  3.19%  2.45%  2.86%  1.05%
##    48:   2.11%  1.48%  3.15%  2.35%  2.86%  1.14%
##    49:   2.06%  1.45%  3.11%  2.35%  2.70%  1.10%
##    50:   2.08%  1.42%  3.11%  2.35%  2.91%  1.05%
##    51:   2.00%  1.45%  2.93%  2.35%  2.70%  0.96%
##    52:   2.00%  1.39%  2.93%  2.35%  2.81%  0.96%
##    53:   2.04%  1.39%  3.11%  2.45%  2.65%  1.01%
##    54:   1.98%  1.42%  2.89%  2.40%  2.49%  1.05%
##    55:   1.97%  1.45%  2.84%  2.35%  2.60%  0.96%
##    56:   1.96%  1.42%  2.89%  2.45%  2.49%  0.91%
##    57:   1.95%  1.42%  2.80%  2.25%  2.70%  0.96%
##    58:   1.89%  1.42%  2.76%  2.10%  2.49%  0.96%
##    59:   1.89%  1.33%  2.84%  2.15%  2.49%  1.01%
##    60:   1.91%  1.39%  2.80%  2.20%  2.54%  0.96%
##    61:   1.85%  1.31%  2.89%  2.10%  2.49%  0.82%
##    62:   1.88%  1.28%  2.89%  2.30%  2.54%  0.78%
##    63:   1.89%  1.36%  2.89%  2.30%  2.54%  0.73%
##    64:   1.88%  1.28%  2.93%  2.30%  2.54%  0.73%
##    65:   1.90%  1.42%  2.84%  2.35%  2.54%  0.69%
##    66:   1.92%  1.45%  3.06%  2.25%  2.44%  0.69%
##    67:   1.93%  1.51%  2.89%  2.50%  2.38%  0.64%
##    68:   1.93%  1.39%  3.02%  2.45%  2.49%  0.64%
##    69:   1.89%  1.39%  2.93%  2.25%  2.54%  0.69%
##    70:   1.90%  1.39%  2.93%  2.30%  2.54%  0.69%
##    71:   1.89%  1.39%  2.98%  2.15%  2.60%  0.69%
##    72:   1.89%  1.39%  2.98%  2.15%  2.60%  0.64%
##    73:   1.86%  1.42%  2.71%  2.30%  2.44%  0.73%
##    74:   1.86%  1.33%  2.89%  2.25%  2.49%  0.69%
##    75:   1.84%  1.39%  2.80%  2.05%  2.60%  0.69%
##    76:   1.84%  1.36%  2.80%  2.30%  2.44%  0.64%
##    77:   1.86%  1.36%  2.84%  2.30%  2.49%  0.64%
##    78:   1.84%  1.33%  2.84%  2.30%  2.44%  0.64%
##    79:   1.86%  1.33%  2.80%  2.40%  2.49%  0.64%
##    80:   1.79%  1.33%  2.80%  2.10%  2.38%  0.64%
##    81:   1.80%  1.28%  2.71%  2.20%  2.60%  0.59%
##    82:   1.86%  1.36%  2.89%  2.15%  2.60%  0.64%
##    83:   1.82%  1.31%  2.76%  2.20%  2.54%  0.64%
##    84:   1.89%  1.33%  2.89%  2.30%  2.60%  0.69%
##    85:   1.89%  1.36%  2.89%  2.20%  2.65%  0.69%
##    86:   1.89%  1.36%  2.98%  2.20%  2.60%  0.69%
##    87:   1.83%  1.33%  2.89%  2.15%  2.54%  0.59%
##    88:   1.88%  1.33%  2.93%  2.30%  2.54%  0.64%
##    89:   1.92%  1.36%  3.02%  2.40%  2.54%  0.64%
##    90:   1.84%  1.31%  2.76%  2.35%  2.54%  0.64%
##    91:   1.83%  1.28%  2.80%  2.20%  2.54%  0.73%
##    92:   1.84%  1.28%  2.84%  2.25%  2.54%  0.69%
##    93:   1.81%  1.25%  2.80%  2.15%  2.54%  0.69%
##    94:   1.82%  1.28%  2.80%  2.25%  2.44%  0.69%
##    95:   1.89%  1.31%  2.89%  2.40%  2.54%  0.69%
##    96:   1.81%  1.25%  2.80%  2.25%  2.44%  0.69%
##    97:   1.83%  1.28%  2.76%  2.30%  2.49%  0.73%
##    98:   1.86%  1.36%  2.84%  2.25%  2.44%  0.73%
##    99:   1.84%  1.33%  2.71%  2.30%  2.49%  0.73%
##   100:   1.85%  1.33%  2.89%  2.25%  2.44%  0.69%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.65%  5.10% 11.50%  9.74%  7.32%  5.84%
##     2:   7.31%  5.70% 10.25%  8.68%  7.36%  5.31%
##     3:   7.15%  5.14%  9.91%  8.80%  7.54%  5.45%
##     4:   6.30%  4.45%  8.27%  8.30%  7.05%  4.61%
##     5:   5.85%  4.06%  8.55%  7.16%  6.15%  4.27%
##     6:   5.51%  3.80%  8.34%  7.58%  5.34%  3.41%
##     7:   5.21%  3.46%  7.28%  7.50%  5.30%  3.52%
##     8:   4.87%  3.07%  7.30%  7.49%  4.71%  2.81%
##     9:   4.46%  2.89%  6.97%  6.49%  4.50%  2.32%
##    10:   4.35%  2.72%  6.50%  6.40%  4.73%  2.35%
##    11:   3.95%  2.63%  6.09%  5.21%  4.27%  2.30%
##    12:   3.71%  2.32%  5.28%  5.44%  4.21%  2.16%
##    13:   3.45%  2.35%  5.18%  4.56%  3.62%  2.15%
##    14:   3.29%  2.29%  4.78%  4.55%  3.50%  1.92%
##    15:   3.10%  2.14%  4.30%  4.21%  3.66%  1.83%
##    16:   3.05%  2.23%  4.52%  4.06%  3.18%  1.74%
##    17:   2.94%  1.96%  4.38%  3.86%  3.34%  1.74%
##    18:   2.79%  1.93%  4.16%  3.57%  3.28%  1.51%
##    19:   2.80%  2.05%  3.98%  3.67%  3.28%  1.51%
##    20:   2.62%  1.84%  3.77%  3.67%  2.97%  1.33%
##    21:   2.62%  1.84%  3.68%  3.33%  3.39%  1.42%
##    22:   2.49%  1.69%  3.59%  3.52%  3.18%  1.01%
##    23:   2.43%  1.75%  3.46%  3.57%  2.91%  0.91%
##    24:   2.38%  1.51%  3.41%  3.57%  3.07%  0.91%
##    25:   2.37%  1.42%  3.59%  3.47%  3.07%  0.91%
##    26:   2.29%  1.54%  3.41%  3.28%  2.86%  0.87%
##    27:   2.33%  1.48%  3.59%  3.47%  2.65%  0.96%
##    28:   2.21%  1.60%  3.33%  2.79%  2.81%  0.91%
##    29:   2.20%  1.48%  3.28%  2.89%  2.97%  0.87%
##    30:   2.16%  1.54%  3.37%  2.79%  2.65%  0.82%
##    31:   2.15%  1.39%  3.33%  2.89%  2.60%  1.01%
##    32:   2.11%  1.36%  3.33%  2.64%  2.70%  0.96%
##    33:   2.09%  1.48%  3.33%  2.45%  2.70%  0.87%
##    34:   2.07%  1.42%  3.24%  2.69%  2.60%  0.82%
##    35:   2.10%  1.45%  3.24%  2.74%  2.75%  0.73%
##    36:   2.03%  1.45%  3.02%  2.54%  2.70%  0.82%
##    37:   2.03%  1.39%  3.11%  2.50%  2.75%  0.82%
##    38:   2.10%  1.48%  3.19%  2.50%  2.86%  0.87%
##    39:   2.03%  1.42%  3.11%  2.45%  2.75%  0.82%
##    40:   2.07%  1.45%  3.15%  2.64%  2.75%  0.78%
##    41:   2.04%  1.33%  3.19%  2.50%  2.91%  0.73%
##    42:   2.04%  1.36%  3.11%  2.50%  3.02%  0.69%
##    43:   2.09%  1.48%  3.11%  2.64%  2.91%  0.73%
##    44:   2.00%  1.51%  2.84%  2.45%  2.91%  0.64%
##    45:   2.00%  1.48%  3.06%  2.35%  2.81%  0.64%
##    46:   1.98%  1.39%  3.02%  2.45%  2.86%  0.59%
##    47:   1.93%  1.42%  2.89%  2.30%  2.70%  0.69%
##    48:   1.93%  1.39%  2.76%  2.54%  2.70%  0.64%
##    49:   1.90%  1.39%  2.93%  2.30%  2.70%  0.55%
##    50:   1.98%  1.39%  3.02%  2.50%  2.70%  0.69%
##    51:   1.97%  1.42%  2.93%  2.45%  2.75%  0.69%
##    52:   1.96%  1.42%  2.89%  2.45%  2.81%  0.64%
##    53:   1.90%  1.39%  2.98%  2.15%  2.75%  0.59%
##    54:   1.97%  1.42%  2.80%  2.54%  2.81%  0.69%
##    55:   1.93%  1.42%  2.80%  2.40%  2.70%  0.69%
##    56:   1.91%  1.45%  2.80%  2.35%  2.54%  0.73%
##    57:   1.93%  1.45%  2.80%  2.25%  2.65%  0.82%
##    58:   1.87%  1.45%  2.76%  2.25%  2.49%  0.69%
##    59:   1.86%  1.39%  2.80%  2.20%  2.49%  0.73%
##    60:   1.84%  1.39%  2.84%  2.15%  2.44%  0.69%
##    61:   1.83%  1.36%  2.71%  2.30%  2.49%  0.64%
##    62:   1.89%  1.36%  2.98%  2.50%  2.33%  0.64%
##    63:   1.91%  1.39%  2.93%  2.40%  2.44%  0.73%
##    64:   1.88%  1.36%  2.84%  2.30%  2.49%  0.73%
##    65:   1.85%  1.36%  2.84%  2.35%  2.38%  0.64%
##    66:   1.85%  1.33%  2.84%  2.35%  2.49%  0.59%
##    67:   1.93%  1.36%  3.02%  2.40%  2.54%  0.69%
##    68:   1.89%  1.33%  2.98%  2.20%  2.54%  0.73%
##    69:   1.88%  1.33%  2.84%  2.25%  2.60%  0.73%
##    70:   1.84%  1.33%  2.89%  2.25%  2.44%  0.64%
##    71:   1.90%  1.33%  2.98%  2.35%  2.54%  0.69%
##    72:   1.85%  1.31%  2.84%  2.40%  2.44%  0.64%
##    73:   1.86%  1.31%  2.80%  2.40%  2.49%  0.69%
##    74:   1.84%  1.25%  2.71%  2.45%  2.54%  0.69%
##    75:   1.84%  1.28%  2.80%  2.35%  2.44%  0.73%
##    76:   1.85%  1.33%  2.71%  2.45%  2.49%  0.64%
##    77:   1.82%  1.36%  2.71%  2.40%  2.33%  0.59%
##    78:   1.88%  1.36%  2.84%  2.50%  2.38%  0.64%
##    79:   1.83%  1.33%  2.71%  2.40%  2.38%  0.64%
##    80:   1.80%  1.33%  2.63%  2.35%  2.33%  0.69%
##    81:   1.82%  1.31%  2.67%  2.35%  2.38%  0.73%
##    82:   1.82%  1.36%  2.63%  2.35%  2.33%  0.73%
##    83:   1.82%  1.33%  2.63%  2.40%  2.33%  0.73%
##    84:   1.83%  1.39%  2.80%  2.30%  2.33%  0.64%
##    85:   1.79%  1.31%  2.76%  2.15%  2.33%  0.73%
##    86:   1.79%  1.28%  2.76%  2.25%  2.28%  0.73%
##    87:   1.76%  1.28%  2.71%  2.10%  2.33%  0.69%
##    88:   1.78%  1.25%  2.80%  2.20%  2.44%  0.59%
##    89:   1.79%  1.28%  2.76%  2.20%  2.44%  0.64%
##    90:   1.78%  1.25%  2.84%  2.15%  2.44%  0.59%
##    91:   1.76%  1.25%  2.71%  2.05%  2.54%  0.59%
##    92:   1.70%  1.25%  2.49%  2.05%  2.49%  0.55%
##    93:   1.72%  1.19%  2.63%  2.10%  2.49%  0.59%
##    94:   1.67%  1.22%  2.58%  1.96%  2.38%  0.55%
##    95:   1.73%  1.25%  2.58%  2.05%  2.54%  0.59%
##    96:   1.73%  1.22%  2.58%  2.05%  2.60%  0.59%
##    97:   1.74%  1.22%  2.63%  2.15%  2.49%  0.59%
##    98:   1.76%  1.22%  2.63%  2.15%  2.60%  0.59%
##    99:   1.72%  1.22%  2.63%  2.01%  2.49%  0.59%
##   100:   1.72%  1.28%  2.63%  1.96%  2.49%  0.59%
## ntree      OOB      1      2      3      4      5
##     1:  10.89%  6.93% 13.61% 12.50% 12.93% 11.25%
##     2:  10.57%  6.51% 13.12% 12.59% 13.08% 10.22%
##     3:  10.15%  6.04% 13.70% 12.68% 10.64% 10.09%
##     4:   9.75%  5.92% 13.49% 12.26% 10.11%  9.26%
##     5:   9.01%  5.23% 12.07% 11.53% 10.19%  8.44%
##     6:   8.48%  5.19% 11.85% 10.56%  9.42%  7.43%
##     7:   7.90%  4.57% 11.23%  9.96%  8.87%  6.97%
##     8:   6.99%  4.32%  9.94%  8.36%  7.61%  6.34%
##     9:   6.31%  3.87%  9.04%  7.89%  6.51%  5.71%
##    10:   5.82%  3.54%  8.71%  7.30%  6.01%  4.92%
##    11:   5.12%  3.18%  7.26%  6.53%  5.62%  4.25%
##    12:   4.86%  3.08%  6.83%  5.87%  5.66%  4.01%
##    13:   4.50%  2.60%  6.37%  5.51%  5.54%  3.72%
##    14:   4.36%  2.69%  6.27%  5.66%  4.72%  3.48%
##    15:   3.95%  2.10%  6.09%  5.21%  4.51%  3.02%
##    16:   3.76%  2.51%  5.72%  4.42%  4.35%  2.60%
##    17:   3.69%  2.42%  5.49%  4.52%  4.20%  2.60%
##    18:   3.56%  2.36%  5.35%  4.42%  3.99%  2.41%
##    19:   3.35%  1.87%  5.31%  3.98%  3.99%  2.51%
##    20:   3.15%  2.07%  4.76%  3.88%  3.48%  2.23%
##    21:   3.02%  1.75%  4.45%  4.07%  3.63%  2.04%
##    22:   2.89%  1.87%  4.26%  3.63%  3.63%  1.72%
##    23:   2.87%  1.81%  4.26%  3.68%  3.43%  1.86%
##    24:   2.72%  1.78%  4.17%  3.14%  3.38%  1.72%
##    25:   2.52%  1.40%  4.08%  3.04%  3.17%  1.62%
##    26:   2.48%  1.49%  3.99%  2.85%  3.02%  1.67%
##    27:   2.41%  1.49%  3.81%  2.90%  3.02%  1.44%
##    28:   2.39%  1.46%  4.13%  2.70%  2.71%  1.48%
##    29:   2.34%  1.20%  3.99%  2.85%  2.81%  1.53%
##    30:   2.34%  1.37%  3.77%  2.80%  2.97%  1.39%
##    31:   2.21%  1.26%  3.68%  2.90%  2.51%  1.30%
##    32:   2.16%  1.11%  3.63%  2.65%  2.71%  1.35%
##    33:   2.25%  1.14%  3.86%  2.75%  2.87%  1.35%
##    34:   2.11%  1.08%  3.68%  2.65%  2.66%  1.11%
##    35:   2.07%  1.20%  3.40%  2.60%  2.71%  1.02%
##    36:   2.11%  1.31%  3.45%  2.60%  2.66%  1.02%
##    37:   2.12%  1.26%  3.54%  2.60%  2.76%  1.02%
##    38:   2.05%  1.23%  3.22%  2.60%  2.56%  1.16%
##    39:   1.97%  1.11%  3.36%  2.36%  2.61%  0.97%
##    40:   1.89%  1.02%  3.18%  2.41%  2.46%  0.97%
##    41:   1.89%  1.08%  3.13%  2.45%  2.35%  0.97%
##    42:   1.86%  1.14%  2.90%  2.31%  2.56%  0.88%
##    43:   1.84%  1.11%  3.04%  2.11%  2.46%  0.97%
##    44:   1.83%  1.11%  3.04%  2.16%  2.41%  0.88%
##    45:   1.82%  1.08%  3.04%  2.11%  2.46%  0.88%
##    46:   1.80%  1.05%  2.95%  2.21%  2.51%  0.79%
##    47:   1.76%  1.02%  2.90%  2.11%  2.51%  0.74%
##    48:   1.77%  1.02%  2.86%  2.26%  2.35%  0.84%
##    49:   1.70%  1.05%  2.72%  1.96%  2.46%  0.74%
##    50:   1.76%  1.20%  2.77%  2.01%  2.46%  0.74%
##    51:   1.71%  1.05%  2.77%  2.11%  2.35%  0.70%
##    52:   1.77%  1.20%  2.72%  2.01%  2.51%  0.84%
##    53:   1.73%  1.11%  2.81%  1.96%  2.41%  0.79%
##    54:   1.73%  1.08%  2.86%  2.01%  2.35%  0.79%
##    55:   1.67%  1.02%  2.77%  1.96%  2.30%  0.74%
##    56:   1.66%  1.05%  2.72%  1.91%  2.20%  0.84%
##    57:   1.66%  0.99%  2.72%  2.01%  2.25%  0.74%
##    58:   1.62%  0.96%  2.77%  1.87%  2.25%  0.70%
##    59:   1.67%  1.02%  2.81%  2.06%  2.20%  0.70%
##    60:   1.63%  0.93%  2.72%  1.96%  2.25%  0.74%
##    61:   1.62%  0.96%  2.81%  1.96%  2.05%  0.74%
##    62:   1.61%  1.02%  2.59%  2.01%  2.05%  0.79%
##    63:   1.70%  1.02%  2.72%  2.16%  2.25%  0.79%
##    64:   1.67%  1.14%  2.59%  2.11%  2.15%  0.74%
##    65:   1.60%  1.08%  2.45%  2.01%  2.20%  0.65%
##    66:   1.61%  1.02%  2.40%  2.11%  2.30%  0.65%
##    67:   1.66%  0.96%  2.72%  2.16%  2.25%  0.70%
##    68:   1.63%  1.08%  2.50%  2.11%  2.15%  0.70%
##    69:   1.64%  0.96%  2.63%  2.16%  2.25%  0.65%
##    70:   1.67%  1.08%  2.63%  2.16%  2.20%  0.70%
##    71:   1.65%  1.05%  2.63%  2.01%  2.20%  0.74%
##    72:   1.63%  1.05%  2.63%  1.91%  2.20%  0.74%
##    73:   1.64%  1.05%  2.54%  2.06%  2.20%  0.74%
##    74:   1.61%  0.96%  2.59%  1.96%  2.30%  0.70%
##    75:   1.59%  0.93%  2.59%  1.91%  2.25%  0.70%
##    76:   1.56%  0.90%  2.45%  1.91%  2.30%  0.70%
##    77:   1.56%  0.88%  2.50%  1.96%  2.25%  0.70%
##    78:   1.59%  0.82%  2.59%  2.06%  2.30%  0.70%
##    79:   1.57%  0.88%  2.59%  1.91%  2.25%  0.70%
##    80:   1.56%  0.90%  2.63%  1.91%  2.15%  0.65%
##    81:   1.54%  0.82%  2.68%  1.91%  2.10%  0.65%
##    82:   1.54%  0.88%  2.59%  1.82%  2.20%  0.65%
##    83:   1.57%  0.88%  2.63%  1.91%  2.25%  0.65%
##    84:   1.55%  0.88%  2.50%  2.01%  2.25%  0.56%
##    85:   1.53%  0.85%  2.45%  1.96%  2.25%  0.60%
##    86:   1.54%  0.90%  2.45%  1.96%  2.20%  0.60%
##    87:   1.55%  0.88%  2.54%  2.01%  2.20%  0.60%
##    88:   1.53%  0.90%  2.45%  2.06%  2.15%  0.51%
##    89:   1.54%  0.93%  2.36%  2.06%  2.20%  0.56%
##    90:   1.53%  0.85%  2.54%  2.01%  2.15%  0.56%
##    91:   1.51%  0.90%  2.45%  1.91%  2.20%  0.51%
##    92:   1.51%  0.93%  2.40%  1.96%  2.10%  0.56%
##    93:   1.48%  0.88%  2.31%  1.96%  2.10%  0.56%
##    94:   1.47%  0.88%  2.31%  1.91%  2.05%  0.60%
##    95:   1.45%  0.93%  2.22%  1.96%  2.05%  0.46%
##    96:   1.44%  0.90%  2.22%  1.87%  2.05%  0.51%
##    97:   1.45%  0.88%  2.40%  1.91%  2.00%  0.46%
##    98:   1.45%  0.88%  2.31%  1.96%  1.94%  0.56%
##    99:   1.45%  0.82%  2.40%  1.96%  2.00%  0.51%
##   100:   1.48%  0.82%  2.40%  2.06%  2.05%  0.51%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.56%  4.09% 10.47% 10.27%  8.98%  6.44%
##     2:   8.10%  5.44% 10.71% 10.07%  9.35%  6.85%
##     3:   7.74%  5.20% 10.95%  9.43%  8.65%  6.21%
##     4:   7.25%  4.49% 10.42%  8.75%  8.28%  6.15%
##     5:   6.72%  4.08% 10.02%  7.33%  8.30%  5.59%
##     6:   6.43%  3.61%  9.93%  7.11%  8.19%  5.12%
##     7:   6.04%  3.47%  9.49%  7.11%  7.58%  4.15%
##     8:   5.69%  3.33%  8.78%  6.32%  7.60%  3.92%
##     9:   5.23%  2.76%  8.05%  6.37%  6.49%  4.02%
##    10:   4.70%  2.60%  7.09%  5.65%  5.83%  3.67%
##    11:   4.27%  2.35%  6.86%  4.60%  5.14%  3.60%
##    12:   4.17%  2.31%  6.33%  4.59%  5.64%  3.17%
##    13:   3.98%  2.02%  6.36%  4.97%  5.07%  2.75%
##    14:   3.83%  1.99%  6.04%  4.67%  4.91%  2.74%
##    15:   3.63%  1.84%  5.26%  4.42%  5.02%  2.79%
##    16:   3.66%  1.87%  5.76%  4.28%  5.12%  2.46%
##    17:   3.56%  1.78%  5.58%  4.18%  5.17%  2.27%
##    18:   3.55%  1.87%  5.67%  4.13%  4.71%  2.46%
##    19:   3.26%  1.61%  5.40%  3.54%  4.61%  2.23%
##    20:   3.12%  1.58%  5.13%  3.54%  4.20%  2.13%
##    21:   3.12%  1.58%  5.22%  3.63%  3.99%  2.13%
##    22:   3.03%  1.43%  5.13%  3.53%  3.94%  2.13%
##    23:   3.01%  1.52%  4.63%  3.58%  4.15%  2.18%
##    24:   3.01%  1.43%  4.99%  3.34%  4.20%  2.09%
##    25:   2.96%  1.37%  4.95%  3.34%  4.09%  2.04%
##    26:   2.89%  1.40%  4.63%  3.14%  4.20%  2.04%
##    27:   2.88%  1.43%  4.49%  3.44%  4.04%  1.95%
##    28:   2.85%  1.34%  4.67%  3.44%  3.94%  1.86%
##    29:   2.88%  1.46%  4.95%  3.19%  3.74%  1.95%
##    30:   2.84%  1.52%  4.63%  3.09%  4.04%  1.81%
##    31:   2.73%  1.40%  4.40%  2.99%  4.04%  1.72%
##    32:   2.65%  1.52%  3.99%  3.04%  3.68%  1.76%
##    33:   2.62%  1.28%  4.22%  2.90%  3.84%  1.76%
##    34:   2.63%  1.17%  4.45%  3.04%  3.84%  1.62%
##    35:   2.53%  1.23%  4.13%  2.60%  3.84%  1.72%
##    36:   2.52%  1.14%  3.95%  2.90%  3.74%  1.81%
##    37:   2.51%  1.31%  3.81%  2.85%  3.89%  1.53%
##    38:   2.54%  1.26%  4.22%  2.75%  3.74%  1.58%
##    39:   2.62%  1.31%  4.22%  2.90%  3.68%  1.81%
##    40:   2.51%  1.26%  3.95%  2.65%  3.79%  1.72%
##    41:   2.43%  1.20%  3.95%  2.65%  3.63%  1.53%
##    42:   2.44%  1.23%  3.81%  2.55%  3.74%  1.67%
##    43:   2.34%  1.05%  4.13%  2.41%  3.38%  1.53%
##    44:   2.33%  1.20%  3.99%  2.26%  3.28%  1.62%
##    45:   2.31%  1.17%  3.81%  2.21%  3.53%  1.58%
##    46:   2.24%  1.08%  3.90%  2.31%  3.22%  1.44%
##    47:   2.27%  1.08%  3.81%  2.21%  3.58%  1.44%
##    48:   2.28%  1.02%  3.90%  2.11%  3.53%  1.67%
##    49:   2.22%  1.08%  3.77%  2.16%  3.38%  1.48%
##    50:   2.22%  1.05%  3.63%  2.21%  3.33%  1.62%
##    51:   2.28%  1.05%  3.77%  2.26%  3.48%  1.62%
##    52:   2.24%  1.08%  3.72%  2.31%  3.28%  1.58%
##    53:   2.24%  1.08%  3.72%  2.21%  3.48%  1.48%
##    54:   2.26%  1.08%  3.63%  2.36%  3.33%  1.67%
##    55:   2.24%  1.02%  3.54%  2.21%  3.58%  1.67%
##    56:   2.25%  0.99%  3.72%  2.31%  3.48%  1.58%
##    57:   2.25%  0.99%  3.81%  2.16%  3.48%  1.62%
##    58:   2.20%  1.02%  3.58%  2.21%  3.48%  1.48%
##    59:   2.22%  1.05%  3.58%  2.21%  3.53%  1.53%
##    60:   2.23%  1.08%  3.58%  2.21%  3.43%  1.62%
##    61:   2.19%  1.05%  3.49%  2.11%  3.53%  1.53%
##    62:   2.20%  1.08%  3.45%  2.06%  3.58%  1.58%
##    63:   2.23%  1.14%  3.63%  2.06%  3.53%  1.53%
##    64:   2.19%  1.05%  3.49%  2.16%  3.43%  1.58%
##    65:   2.23%  1.08%  3.58%  2.16%  3.53%  1.58%
##    66:   2.24%  1.08%  3.68%  2.11%  3.48%  1.62%
##    67:   2.22%  1.05%  3.68%  2.16%  3.48%  1.48%
##    68:   2.24%  1.05%  3.72%  2.06%  3.53%  1.62%
##    69:   2.18%  1.08%  3.49%  2.01%  3.43%  1.62%
##    70:   2.22%  1.08%  3.45%  2.31%  3.48%  1.58%
##    71:   2.20%  1.05%  3.45%  2.16%  3.53%  1.58%
##    72:   2.22%  1.11%  3.49%  2.06%  3.53%  1.62%
##    73:   2.21%  1.17%  3.49%  2.11%  3.28%  1.67%
##    74:   2.20%  1.14%  3.40%  2.11%  3.43%  1.62%
##    75:   2.11%  1.08%  3.36%  2.11%  3.22%  1.48%
##    76:   2.11%  1.11%  3.31%  2.06%  3.22%  1.48%
##    77:   2.11%  1.08%  3.27%  2.06%  3.28%  1.53%
##    78:   2.17%  1.08%  3.49%  2.11%  3.28%  1.58%
##    79:   2.16%  1.14%  3.36%  2.16%  3.28%  1.53%
##    80:   2.16%  1.17%  3.40%  2.11%  3.28%  1.48%
##    81:   2.14%  1.05%  3.40%  2.21%  3.33%  1.44%
##    82:   2.14%  1.11%  3.36%  2.21%  3.28%  1.44%
##    83:   2.16%  1.08%  3.40%  2.21%  3.33%  1.48%
##    84:   2.16%  1.14%  3.45%  2.11%  3.22%  1.53%
##    85:   2.15%  1.20%  3.22%  2.21%  3.22%  1.53%
##    86:   2.10%  1.17%  3.13%  2.16%  3.22%  1.44%
##    87:   2.13%  1.20%  3.31%  2.16%  3.12%  1.48%
##    88:   2.09%  1.14%  3.22%  2.16%  3.12%  1.44%
##    89:   2.10%  1.11%  3.27%  2.16%  3.17%  1.44%
##    90:   2.07%  1.08%  3.22%  2.11%  3.12%  1.48%
##    91:   2.10%  1.05%  3.36%  2.21%  3.12%  1.44%
##    92:   2.12%  1.11%  3.40%  2.21%  3.12%  1.44%
##    93:   2.08%  1.05%  3.45%  2.16%  3.07%  1.35%
##    94:   2.12%  1.14%  3.45%  2.21%  3.12%  1.35%
##    95:   2.11%  1.08%  3.36%  2.21%  3.17%  1.39%
##    96:   2.11%  1.11%  3.36%  2.31%  3.12%  1.35%
##    97:   2.11%  1.05%  3.31%  2.31%  3.12%  1.44%
##    98:   2.12%  1.14%  3.27%  2.26%  3.22%  1.39%
##    99:   2.13%  1.14%  3.36%  2.36%  3.02%  1.44%
##   100:   2.14%  1.08%  3.40%  2.31%  3.17%  1.44%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.44%  5.79% 10.54%  8.63%  8.06%  5.10%
##     2:   7.88%  5.26% 11.92%  9.02%  8.16%  6.54%
##     3:   7.38%  4.55% 11.17%  8.05%  8.46%  6.29%
##     4:   7.10%  4.03% 10.71%  8.42%  8.13%  6.12%
##     5:   6.44%  3.89% 10.08%  7.15%  7.22%  5.43%
##     6:   6.40%  3.64%  9.70%  7.61%  6.90%  5.82%
##     7:   5.81%  3.67%  8.53%  6.90%  6.27%  5.00%
##     8:   5.35%  3.37%  7.72%  6.24%  6.03%  4.60%
##     9:   5.12%  3.45%  7.31%  6.58%  5.03%  4.23%
##    10:   5.03%  3.24%  7.22%  6.01%  5.64%  4.17%
##    11:   4.35%  2.71%  6.23%  5.40%  5.20%  3.27%
##    12:   4.19%  2.78%  5.75%  4.74%  5.34%  3.26%
##    13:   3.94%  2.55%  5.56%  4.29%  5.28%  2.93%
##    14:   3.87%  2.25%  5.41%  4.43%  5.43%  2.93%
##    15:   3.93%  2.54%  5.59%  4.23%  5.28%  2.97%
##    16:   3.53%  2.10%  5.36%  3.73%  4.87%  2.55%
##    17:   3.46%  2.16%  4.72%  3.93%  4.61%  2.74%
##    18:   3.31%  2.01%  4.76%  3.73%  4.40%  2.51%
##    19:   3.14%  1.93%  4.40%  3.53%  4.35%  2.32%
##    20:   2.98%  1.69%  4.31%  3.53%  3.84%  2.37%
##    21:   2.90%  1.72%  4.26%  3.29%  3.79%  2.23%
##    22:   2.84%  1.90%  3.86%  3.49%  3.63%  2.00%
##    23:   2.83%  1.87%  4.22%  3.19%  3.58%  1.90%
##    24:   2.85%  1.93%  3.99%  3.24%  3.84%  1.90%
##    25:   2.79%  1.78%  4.04%  3.09%  3.74%  1.95%
##    26:   2.79%  1.90%  3.95%  3.49%  3.22%  1.95%
##    27:   2.79%  1.81%  4.13%  3.49%  3.28%  1.90%
##    28:   2.74%  1.90%  3.99%  3.19%  3.33%  1.86%
##    29:   2.64%  1.69%  3.99%  3.14%  3.17%  1.81%
##    30:   2.58%  1.63%  3.81%  3.14%  3.17%  1.76%
##    31:   2.49%  1.55%  3.63%  3.04%  3.17%  1.67%
##    32:   2.59%  1.61%  3.63%  3.24%  3.33%  1.81%
##    33:   2.51%  1.49%  3.58%  3.14%  3.28%  1.76%
##    34:   2.47%  1.46%  3.63%  3.19%  3.22%  1.53%
##    35:   2.47%  1.46%  3.54%  3.29%  3.28%  1.48%
##    36:   2.37%  1.26%  3.54%  3.09%  3.17%  1.53%
##    37:   2.35%  1.31%  3.49%  3.04%  3.02%  1.58%
##    38:   2.42%  1.43%  3.54%  2.99%  3.07%  1.72%
##    39:   2.37%  1.37%  3.31%  3.04%  3.07%  1.72%
##    40:   2.34%  1.28%  3.45%  3.09%  2.92%  1.67%
##    41:   2.38%  1.31%  3.68%  2.99%  2.92%  1.67%
##    42:   2.37%  1.31%  3.77%  2.95%  2.87%  1.62%
##    43:   2.36%  1.17%  3.77%  2.90%  3.02%  1.72%
##    44:   2.32%  1.26%  3.49%  2.90%  3.12%  1.53%
##    45:   2.32%  1.26%  3.72%  2.90%  2.76%  1.62%
##    46:   2.33%  1.28%  3.68%  2.90%  3.07%  1.39%
##    47:   2.28%  1.20%  3.72%  2.80%  2.92%  1.48%
##    48:   2.28%  1.11%  3.72%  2.90%  2.92%  1.48%
##    49:   2.33%  1.17%  3.86%  2.90%  3.02%  1.44%
##    50:   2.25%  1.14%  3.72%  2.70%  3.07%  1.35%
##    51:   2.37%  1.23%  3.86%  2.85%  3.28%  1.39%
##    52:   2.28%  1.14%  3.77%  2.80%  3.07%  1.39%
##    53:   2.18%  1.14%  3.49%  2.55%  3.07%  1.35%
##    54:   2.22%  1.11%  3.72%  2.60%  3.02%  1.39%
##    55:   2.22%  1.08%  3.68%  2.70%  2.97%  1.39%
##    56:   2.25%  1.11%  3.68%  2.75%  3.12%  1.35%
##    57:   2.18%  1.14%  3.58%  2.60%  2.97%  1.30%
##    58:   2.20%  1.20%  3.54%  2.60%  3.02%  1.30%
##    59:   2.20%  1.14%  3.49%  2.55%  3.17%  1.35%
##    60:   2.23%  1.08%  3.58%  2.70%  3.22%  1.35%
##    61:   2.19%  1.05%  3.63%  2.55%  3.17%  1.30%
##    62:   2.22%  1.11%  3.58%  2.65%  3.17%  1.30%
##    63:   2.22%  1.11%  3.58%  2.60%  3.22%  1.35%
##    64:   2.21%  1.11%  3.58%  2.60%  3.22%  1.25%
##    65:   2.18%  1.11%  3.58%  2.50%  3.17%  1.25%
##    66:   2.15%  1.17%  3.45%  2.41%  3.17%  1.21%
##    67:   2.14%  1.05%  3.45%  2.45%  3.22%  1.25%
##    68:   2.12%  1.08%  3.36%  2.50%  3.07%  1.30%
##    69:   2.13%  1.17%  3.40%  2.36%  3.07%  1.30%
##    70:   2.11%  1.08%  3.49%  2.41%  3.02%  1.25%
##    71:   2.14%  1.11%  3.63%  2.31%  3.07%  1.25%
##    72:   2.13%  1.11%  3.49%  2.45%  3.02%  1.25%
##    73:   2.06%  1.08%  3.18%  2.36%  3.07%  1.30%
##    74:   2.11%  0.99%  3.36%  2.50%  3.22%  1.25%
##    75:   2.11%  0.99%  3.36%  2.50%  3.17%  1.25%
##    76:   2.06%  1.02%  3.31%  2.41%  3.02%  1.21%
##    77:   2.07%  1.02%  3.27%  2.41%  3.22%  1.16%
##    78:   2.10%  1.05%  3.22%  2.45%  3.22%  1.25%
##    79:   2.11%  1.05%  3.22%  2.55%  3.22%  1.25%
##    80:   2.08%  1.05%  3.36%  2.41%  3.12%  1.16%
##    81:   2.12%  1.11%  3.22%  2.60%  3.12%  1.25%
##    82:   2.10%  1.05%  3.31%  2.41%  3.17%  1.25%
##    83:   2.10%  1.02%  3.40%  2.41%  3.07%  1.30%
##    84:   2.13%  1.11%  3.45%  2.36%  3.12%  1.30%
##    85:   2.12%  1.02%  3.49%  2.41%  3.07%  1.35%
##    86:   2.11%  1.05%  3.45%  2.31%  3.07%  1.39%
##    87:   2.14%  1.08%  3.58%  2.41%  3.07%  1.25%
##    88:   2.09%  1.08%  3.40%  2.36%  3.02%  1.25%
##    89:   2.08%  1.05%  3.40%  2.31%  3.02%  1.30%
##    90:   2.07%  1.11%  3.31%  2.26%  3.02%  1.30%
##    91:   2.05%  1.02%  3.36%  2.26%  3.02%  1.25%
##    92:   2.02%  1.08%  3.27%  2.11%  2.97%  1.30%
##    93:   2.06%  1.05%  3.36%  2.31%  2.97%  1.30%
##    94:   2.08%  1.11%  3.36%  2.26%  3.07%  1.25%
##    95:   2.07%  1.11%  3.36%  2.31%  2.92%  1.30%
##    96:   2.01%  1.08%  3.27%  2.26%  2.81%  1.25%
##    97:   2.00%  1.14%  3.27%  2.06%  2.92%  1.21%
##    98:   2.02%  1.11%  3.31%  2.11%  2.97%  1.21%
##    99:   2.06%  1.11%  3.40%  2.16%  3.02%  1.21%
##   100:   2.06%  1.23%  3.31%  2.11%  2.97%  1.25%
## ntree      OOB      1      2      3      4      5
##     1:   9.67%  5.78% 11.11% 13.75%  9.44% 10.22%
##     2:   9.14%  5.76% 11.40% 11.76%  9.62%  9.05%
##     3:   9.10%  5.88% 11.50% 11.32% 10.84%  8.02%
##     4:   8.94%  5.40% 11.98% 11.72% 10.22%  7.45%
##     5:   8.05%  5.45% 10.42%  9.73%  9.50%  6.70%
##     6:   7.62%  4.85%  9.93%  9.17%  9.52%  6.32%
##     7:   7.05%  4.42%  9.06%  8.81%  9.44%  5.26%
##     8:   6.44%  4.08%  8.00%  7.61%  9.13%  5.00%
##     9:   5.61%  3.68%  7.19%  6.85%  7.38%  4.19%
##    10:   5.09%  3.17%  6.62%  6.41%  6.89%  3.61%
##    11:   4.54%  2.89%  6.18%  5.29%  6.15%  3.23%
##    12:   4.01%  2.56%  5.54%  5.02%  5.00%  2.80%
##    13:   3.78%  2.56%  5.28%  4.62%  4.83%  2.33%
##    14:   3.35%  2.38%  4.99%  3.92%  4.17%  1.81%
##    15:   3.11%  2.23%  4.27%  3.77%  4.22%  1.63%
##    16:   2.85%  2.02%  4.31%  3.37%  3.52%  1.49%
##    17:   2.75%  1.99%  3.64%  3.66%  3.63%  1.35%
##    18:   2.66%  1.81%  3.85%  3.22%  3.63%  1.30%
##    19:   2.58%  1.87%  3.68%  3.07%  3.41%  1.30%
##    20:   2.34%  1.84%  3.31%  2.77%  2.71%  1.35%
##    21:   2.34%  1.81%  3.26%  2.67%  2.92%  1.35%
##    22:   2.26%  1.58%  3.22%  2.77%  2.92%  1.21%
##    23:   2.06%  1.49%  2.93%  2.57%  2.87%  0.84%
##    24:   2.06%  1.49%  2.85%  2.57%  2.71%  1.02%
##    25:   2.15%  1.63%  2.89%  3.02%  2.60%  0.93%
##    26:   2.02%  1.28%  2.93%  2.72%  2.60%  1.02%
##    27:   1.94%  1.22%  2.55%  2.67%  2.71%  1.02%
##    28:   1.86%  1.40%  2.59%  2.43%  2.33%  0.84%
##    29:   1.82%  1.22%  2.64%  2.48%  2.11%  0.97%
##    30:   1.74%  1.13%  2.55%  2.48%  2.17%  0.74%
##    31:   1.81%  1.25%  2.59%  2.28%  2.38%  0.88%
##    32:   1.72%  1.13%  2.30%  2.33%  2.27%  0.97%
##    33:   1.77%  1.07%  2.59%  2.33%  2.38%  0.88%
##    34:   1.77%  1.10%  2.51%  2.38%  2.33%  0.93%
##    35:   1.67%  1.19%  2.18%  2.43%  2.22%  0.70%
##    36:   1.67%  1.10%  2.43%  2.13%  2.27%  0.79%
##    37:   1.70%  1.07%  2.38%  2.33%  2.27%  0.84%
##    38:   1.59%  1.01%  2.30%  2.18%  2.06%  0.74%
##    39:   1.56%  0.95%  2.26%  2.23%  2.11%  0.65%
##    40:   1.62%  1.07%  2.26%  2.33%  2.06%  0.74%
##    41:   1.53%  1.04%  2.22%  2.08%  1.89%  0.70%
##    42:   1.55%  0.95%  2.22%  2.13%  2.00%  0.79%
##    43:   1.55%  1.01%  2.22%  2.03%  2.06%  0.74%
##    44:   1.52%  0.89%  2.30%  2.08%  1.95%  0.74%
##    45:   1.53%  0.92%  2.26%  1.98%  2.00%  0.84%
##    46:   1.47%  0.77%  2.22%  2.03%  1.95%  0.79%
##    47:   1.49%  0.86%  2.26%  1.98%  1.84%  0.88%
##    48:   1.45%  0.89%  2.26%  1.83%  1.84%  0.74%
##    49:   1.40%  0.86%  2.18%  1.78%  1.84%  0.65%
##    50:   1.43%  0.89%  2.09%  1.93%  1.84%  0.70%
##    51:   1.47%  0.89%  2.13%  1.93%  2.00%  0.74%
##    52:   1.40%  0.80%  2.05%  1.93%  1.95%  0.65%
##    53:   1.35%  0.83%  2.01%  1.88%  1.68%  0.65%
##    54:   1.44%  0.92%  2.18%  2.03%  1.73%  0.65%
##    55:   1.43%  0.86%  2.09%  2.08%  1.79%  0.65%
##    56:   1.42%  0.86%  2.09%  1.98%  1.79%  0.70%
##    57:   1.36%  0.86%  2.05%  1.88%  1.68%  0.60%
##    58:   1.39%  0.83%  2.09%  1.93%  1.79%  0.65%
##    59:   1.35%  0.83%  2.09%  1.88%  1.62%  0.60%
##    60:   1.35%  0.77%  2.09%  2.03%  1.57%  0.60%
##    61:   1.31%  0.80%  2.05%  1.78%  1.62%  0.56%
##    62:   1.33%  0.74%  2.09%  1.83%  1.68%  0.65%
##    63:   1.29%  0.68%  2.01%  1.93%  1.73%  0.46%
##    64:   1.32%  0.71%  2.09%  2.03%  1.62%  0.46%
##    65:   1.30%  0.71%  2.05%  1.93%  1.68%  0.46%
##    66:   1.34%  0.77%  2.22%  1.98%  1.62%  0.42%
##    67:   1.32%  0.83%  2.09%  1.83%  1.57%  0.51%
##    68:   1.30%  0.74%  2.13%  1.88%  1.57%  0.46%
##    69:   1.28%  0.65%  2.13%  1.88%  1.57%  0.51%
##    70:   1.30%  0.80%  1.97%  1.88%  1.62%  0.51%
##    71:   1.30%  0.77%  2.01%  1.93%  1.52%  0.56%
##    72:   1.27%  0.74%  1.97%  1.93%  1.57%  0.46%
##    73:   1.26%  0.74%  1.84%  1.93%  1.57%  0.51%
##    74:   1.26%  0.77%  1.88%  1.88%  1.52%  0.51%
##    75:   1.24%  0.65%  1.97%  1.93%  1.52%  0.46%
##    76:   1.29%  0.83%  1.88%  2.03%  1.52%  0.46%
##    77:   1.21%  0.59%  1.92%  1.93%  1.46%  0.46%
##    78:   1.21%  0.68%  1.88%  1.93%  1.41%  0.42%
##    79:   1.22%  0.65%  1.88%  1.98%  1.46%  0.46%
##    80:   1.22%  0.65%  1.88%  1.93%  1.52%  0.46%
##    81:   1.24%  0.68%  1.97%  1.98%  1.35%  0.51%
##    82:   1.21%  0.65%  1.84%  1.93%  1.41%  0.51%
##    83:   1.20%  0.65%  1.80%  1.88%  1.46%  0.51%
##    84:   1.20%  0.62%  1.84%  1.88%  1.46%  0.51%
##    85:   1.18%  0.56%  1.80%  1.88%  1.57%  0.46%
##    86:   1.21%  0.59%  1.88%  1.83%  1.52%  0.56%
##    87:   1.16%  0.59%  1.76%  1.83%  1.46%  0.51%
##    88:   1.16%  0.59%  1.76%  1.88%  1.46%  0.46%
##    89:   1.20%  0.62%  1.84%  1.88%  1.46%  0.51%
##    90:   1.21%  0.62%  1.88%  2.03%  1.46%  0.42%
##    91:   1.15%  0.62%  1.72%  1.93%  1.41%  0.42%
##    92:   1.13%  0.51%  1.80%  1.93%  1.35%  0.42%
##    93:   1.21%  0.62%  1.92%  2.03%  1.41%  0.42%
##    94:   1.18%  0.59%  1.92%  1.88%  1.41%  0.42%
##    95:   1.17%  0.56%  1.97%  1.88%  1.35%  0.42%
##    96:   1.13%  0.56%  1.72%  1.93%  1.35%  0.42%
##    97:   1.15%  0.62%  1.76%  1.93%  1.35%  0.42%
##    98:   1.19%  0.62%  1.88%  1.93%  1.41%  0.42%
##    99:   1.15%  0.56%  1.80%  1.98%  1.35%  0.37%
##   100:   1.14%  0.53%  1.80%  2.03%  1.30%  0.37%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.82%  5.58% 10.79% 11.34%  7.20%  5.38%
##     2:   6.79%  4.24%  9.34%  9.76%  7.40%  4.82%
##     3:   7.18%  4.81% 10.03%  9.80%  7.93%  4.75%
##     4:   6.38%  4.07%  8.81%  9.25%  6.85%  4.30%
##     5:   6.32%  3.78%  8.85%  8.64%  7.88%  4.00%
##     6:   6.04%  3.93%  7.89%  8.18%  7.85%  3.75%
##     7:   5.37%  4.02%  6.78%  7.21%  6.69%  3.10%
##     8:   5.00%  3.20%  6.35%  6.46%  6.89%  3.35%
##     9:   4.40%  3.22%  5.36%  6.15%  5.75%  2.37%
##    10:   4.03%  2.61%  4.92%  5.88%  5.10%  2.63%
##    11:   3.97%  2.81%  4.64%  5.66%  5.24%  2.38%
##    12:   3.71%  2.59%  4.75%  4.94%  4.52%  2.47%
##    13:   3.52%  2.41%  4.40%  5.07%  4.40%  2.05%
##    14:   3.30%  2.32%  4.19%  4.46%  4.34%  1.86%
##    15:   2.97%  1.93%  3.72%  4.36%  3.90%  1.63%
##    16:   2.75%  1.72%  3.68%  3.81%  3.47%  1.72%
##    17:   2.73%  1.60%  3.60%  3.67%  3.58%  1.91%
##    18:   2.53%  1.46%  3.43%  3.47%  3.41%  1.58%
##    19:   2.47%  1.46%  3.39%  3.47%  3.19%  1.49%
##    20:   2.52%  1.52%  3.43%  3.47%  3.30%  1.53%
##    21:   2.39%  1.34%  3.05%  3.47%  3.30%  1.53%
##    22:   2.29%  1.31%  2.93%  3.47%  3.19%  1.25%
##    23:   2.36%  1.40%  3.10%  3.27%  3.14%  1.53%
##    24:   2.32%  1.37%  2.97%  3.27%  3.25%  1.39%
##    25:   2.27%  1.31%  3.26%  3.02%  3.14%  1.21%
##    26:   2.07%  1.16%  2.93%  2.92%  2.92%  1.02%
##    27:   2.06%  1.25%  2.97%  2.57%  2.98%  1.02%
##    28:   2.08%  1.34%  2.80%  2.87%  2.87%  1.02%
##    29:   2.05%  1.22%  2.93%  2.77%  2.87%  0.97%
##    30:   1.93%  1.13%  2.72%  2.43%  2.76%  1.11%
##    31:   1.94%  1.16%  2.51%  2.48%  2.98%  1.11%
##    32:   1.98%  1.22%  2.59%  2.77%  2.82%  1.02%
##    33:   1.98%  1.13%  2.59%  2.77%  2.87%  1.11%
##    34:   2.04%  1.25%  2.51%  2.97%  2.98%  1.07%
##    35:   1.95%  1.16%  2.59%  2.72%  2.87%  0.97%
##    36:   2.00%  1.25%  2.55%  2.77%  2.92%  1.02%
##    37:   1.92%  1.19%  2.51%  2.62%  2.82%  0.97%
##    38:   1.90%  1.10%  2.38%  2.72%  2.82%  1.07%
##    39:   1.83%  1.13%  2.22%  2.57%  2.76%  0.97%
##    40:   1.83%  1.19%  2.30%  2.52%  2.60%  0.97%
##    41:   1.77%  1.10%  2.26%  2.38%  2.54%  1.02%
##    42:   1.76%  1.10%  2.13%  2.52%  2.54%  0.97%
##    43:   1.79%  1.13%  2.30%  2.28%  2.65%  1.07%
##    44:   1.77%  1.07%  2.22%  2.28%  2.82%  1.02%
##    45:   1.76%  1.10%  2.18%  2.18%  2.82%  1.02%
##    46:   1.77%  1.10%  2.26%  2.28%  2.65%  1.02%
##    47:   1.72%  1.01%  2.26%  2.28%  2.54%  0.97%
##    48:   1.68%  1.07%  2.13%  2.18%  2.60%  0.88%
##    49:   1.63%  0.98%  2.01%  2.13%  2.60%  0.93%
##    50:   1.73%  1.10%  2.05%  2.33%  2.71%  0.97%
##    51:   1.70%  1.04%  2.18%  2.18%  2.54%  1.02%
##    52:   1.74%  1.01%  2.22%  2.43%  2.60%  0.97%
##    53:   1.70%  1.01%  2.22%  2.33%  2.49%  0.93%
##    54:   1.66%  0.95%  2.05%  2.33%  2.49%  0.97%
##    55:   1.71%  1.04%  2.05%  2.33%  2.60%  1.02%
##    56:   1.63%  0.95%  2.05%  2.18%  2.54%  0.93%
##    57:   1.68%  0.95%  2.18%  2.23%  2.60%  0.97%
##    58:   1.67%  0.98%  2.18%  2.18%  2.49%  1.02%
##    59:   1.70%  0.98%  2.18%  2.38%  2.49%  0.97%
##    60:   1.66%  0.89%  2.13%  2.28%  2.54%  1.02%
##    61:   1.64%  0.92%  2.09%  2.13%  2.49%  1.07%
##    62:   1.66%  0.89%  2.13%  2.23%  2.60%  0.97%
##    63:   1.64%  0.92%  2.09%  2.28%  2.44%  0.97%
##    64:   1.63%  0.95%  2.09%  2.08%  2.44%  1.07%
##    65:   1.60%  0.92%  2.13%  2.03%  2.44%  0.97%
##    66:   1.60%  0.95%  2.09%  2.08%  2.38%  0.93%
##    67:   1.63%  0.92%  2.22%  2.08%  2.49%  0.93%
##    68:   1.69%  0.98%  2.18%  2.13%  2.60%  1.07%
##    69:   1.66%  0.92%  2.13%  2.33%  2.44%  1.02%
##    70:   1.68%  0.95%  2.22%  2.18%  2.54%  1.02%
##    71:   1.69%  0.95%  2.30%  2.18%  2.54%  0.97%
##    72:   1.67%  0.92%  2.22%  2.18%  2.60%  0.97%
##    73:   1.66%  0.92%  2.13%  2.13%  2.71%  0.97%
##    74:   1.66%  0.92%  2.09%  2.23%  2.71%  0.88%
##    75:   1.66%  0.98%  2.13%  2.13%  2.60%  0.93%
##    76:   1.60%  0.95%  2.09%  2.08%  2.44%  0.93%
##    77:   1.60%  0.98%  2.09%  1.98%  2.49%  0.93%
##    78:   1.61%  0.95%  2.13%  2.03%  2.49%  0.93%
##    79:   1.60%  0.92%  2.18%  2.03%  2.44%  0.93%
##    80:   1.60%  0.95%  2.13%  2.13%  2.38%  0.88%
##    81:   1.57%  0.92%  2.09%  1.98%  2.38%  0.93%
##    82:   1.55%  0.92%  2.05%  1.93%  2.38%  0.88%
##    83:   1.55%  0.89%  2.05%  1.98%  2.33%  0.93%
##    84:   1.53%  0.89%  2.01%  2.03%  2.33%  0.84%
##    85:   1.52%  0.86%  2.05%  2.03%  2.27%  0.84%
##    86:   1.57%  0.89%  2.09%  2.08%  2.44%  0.84%
##    87:   1.57%  0.86%  2.13%  2.03%  2.49%  0.84%
##    88:   1.55%  0.86%  2.05%  1.98%  2.49%  0.84%
##    89:   1.52%  0.89%  1.97%  1.98%  2.44%  0.79%
##    90:   1.55%  0.92%  1.97%  2.08%  2.44%  0.79%
##    91:   1.54%  0.95%  2.01%  1.98%  2.33%  0.84%
##    92:   1.55%  0.92%  2.13%  1.88%  2.49%  0.74%
##    93:   1.54%  0.92%  1.97%  2.03%  2.49%  0.74%
##    94:   1.52%  0.89%  1.97%  1.93%  2.49%  0.79%
##    95:   1.54%  0.89%  2.05%  1.93%  2.49%  0.79%
##    96:   1.49%  0.86%  1.84%  1.98%  2.49%  0.79%
##    97:   1.50%  0.86%  1.97%  1.88%  2.49%  0.79%
##    98:   1.50%  0.89%  1.97%  1.88%  2.44%  0.79%
##    99:   1.52%  0.92%  1.97%  1.98%  2.38%  0.79%
##   100:   1.57%  0.95%  2.09%  1.98%  2.44%  0.84%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.70%  3.59% 10.58% 10.51% 11.62%  4.95%
##     2:   7.18%  3.18%  9.94% 10.32% 10.41%  4.79%
##     3:   6.78%  4.05%  8.71%  9.47%  8.64%  4.84%
##     4:   6.41%  3.90%  8.03%  8.92%  8.17%  4.73%
##     5:   5.93%  3.90%  7.46%  7.94%  7.35%  4.36%
##     6:   5.51%  3.65%  7.29%  7.13%  6.67%  3.98%
##     7:   5.34%  3.37%  6.97%  6.85%  6.95%  3.83%
##     8:   4.99%  2.90%  6.31%  6.77%  6.67%  3.68%
##     9:   4.70%  2.90%  6.34%  5.99%  5.92%  3.45%
##    10:   4.28%  2.70%  5.54%  5.55%  5.73%  2.91%
##    11:   3.74%  2.54%  4.59%  4.98%  4.96%  2.48%
##    12:   3.72%  2.15%  4.83%  5.27%  4.99%  2.42%
##    13:   3.45%  2.23%  4.20%  4.37%  5.09%  2.23%
##    14:   3.27%  2.08%  4.36%  3.97%  4.44%  2.28%
##    15:   3.21%  1.82%  4.19%  4.51%  4.39%  2.09%
##    16:   2.96%  1.61%  3.60%  4.46%  4.17%  1.90%
##    17:   2.66%  1.58%  3.35%  3.67%  3.68%  1.76%
##    18:   2.41%  1.40%  3.02%  3.37%  3.52%  1.49%
##    19:   2.46%  1.46%  3.18%  3.22%  3.47%  1.62%
##    20:   2.34%  1.40%  3.10%  2.62%  3.74%  1.49%
##    21:   2.34%  1.40%  2.97%  3.07%  3.52%  1.44%
##    22:   2.26%  1.31%  3.01%  2.77%  3.47%  1.39%
##    23:   2.23%  1.34%  3.01%  2.77%  3.19%  1.44%
##    24:   2.06%  1.22%  2.93%  2.38%  3.09%  1.25%
##    25:   1.99%  1.10%  2.80%  2.33%  3.03%  1.25%
##    26:   1.97%  1.01%  2.68%  2.38%  3.14%  1.30%
##    27:   1.95%  1.10%  2.51%  2.38%  3.19%  1.21%
##    28:   1.98%  1.07%  2.59%  2.48%  2.98%  1.39%
##    29:   2.04%  1.10%  2.76%  2.57%  3.03%  1.35%
##    30:   2.00%  1.04%  2.68%  2.57%  3.09%  1.25%
##    31:   2.00%  1.22%  2.51%  2.62%  2.87%  1.30%
##    32:   1.89%  1.10%  2.55%  2.38%  2.71%  1.25%
##    33:   1.90%  1.22%  2.43%  2.43%  2.82%  1.11%
##    34:   1.83%  1.31%  2.26%  2.33%  2.60%  1.02%
##    35:   1.76%  1.01%  2.38%  2.48%  2.44%  0.97%
##    36:   1.76%  1.07%  2.26%  2.43%  2.54%  0.97%
##    37:   1.69%  1.01%  2.22%  2.28%  2.44%  0.97%
##    38:   1.73%  1.16%  2.22%  2.08%  2.65%  0.97%
##    39:   1.69%  1.04%  2.38%  2.03%  2.33%  1.07%
##    40:   1.65%  0.92%  2.34%  2.23%  2.33%  0.88%
##    41:   1.75%  0.95%  2.34%  2.33%  2.49%  1.16%
##    42:   1.64%  0.89%  2.13%  2.23%  2.38%  1.07%
##    43:   1.67%  0.95%  2.34%  2.03%  2.44%  1.07%
##    44:   1.64%  0.95%  2.26%  2.18%  2.17%  1.07%
##    45:   1.68%  0.98%  2.34%  2.28%  2.38%  0.88%
##    46:   1.60%  0.98%  2.22%  2.13%  2.27%  0.84%
##    47:   1.57%  0.95%  2.18%  2.03%  2.17%  0.93%
##    48:   1.62%  1.07%  2.22%  1.98%  2.27%  0.93%
##    49:   1.65%  1.04%  2.22%  1.98%  2.44%  0.97%
##    50:   1.57%  0.92%  2.22%  1.83%  2.44%  0.88%
##    51:   1.55%  1.01%  2.18%  1.88%  2.22%  0.84%
##    52:   1.56%  1.04%  2.13%  1.83%  2.27%  0.88%
##    53:   1.48%  0.89%  2.13%  1.63%  2.22%  0.88%
##    54:   1.53%  1.04%  2.09%  1.68%  2.22%  0.93%
##    55:   1.51%  0.95%  2.13%  1.58%  2.33%  0.93%
##    56:   1.51%  0.95%  2.09%  1.73%  2.27%  0.88%
##    57:   1.55%  0.98%  2.09%  1.93%  2.27%  0.88%
##    58:   1.60%  0.98%  2.22%  1.98%  2.33%  0.93%
##    59:   1.60%  0.98%  2.26%  2.03%  2.22%  0.93%
##    60:   1.61%  1.01%  2.34%  1.88%  2.33%  0.88%
##    61:   1.62%  1.07%  2.26%  1.88%  2.27%  0.97%
##    62:   1.62%  1.04%  2.30%  1.88%  2.33%  0.93%
##    63:   1.63%  1.04%  2.30%  2.03%  2.33%  0.84%
##    64:   1.64%  1.04%  2.38%  1.88%  2.44%  0.84%
##    65:   1.65%  1.07%  2.30%  1.93%  2.33%  0.97%
##    66:   1.64%  1.07%  2.26%  1.98%  2.38%  0.88%
##    67:   1.69%  1.10%  2.26%  1.98%  2.54%  0.97%
##    68:   1.62%  1.07%  2.13%  1.88%  2.38%  1.02%
##    69:   1.62%  1.07%  2.13%  1.93%  2.44%  0.93%
##    70:   1.61%  1.07%  2.01%  2.03%  2.44%  0.93%
##    71:   1.59%  1.04%  2.01%  1.93%  2.44%  0.93%
##    72:   1.63%  1.10%  2.01%  1.93%  2.54%  0.97%
##    73:   1.61%  1.04%  1.97%  2.08%  2.44%  0.97%
##    74:   1.60%  1.07%  1.97%  1.98%  2.38%  0.97%
##    75:   1.55%  0.92%  2.09%  1.83%  2.38%  0.97%
##    76:   1.55%  0.92%  2.01%  1.88%  2.44%  0.93%
##    77:   1.52%  0.83%  2.05%  1.83%  2.38%  0.97%
##    78:   1.51%  0.86%  2.01%  1.78%  2.38%  0.97%
##    79:   1.52%  0.95%  2.01%  1.78%  2.33%  0.93%
##    80:   1.55%  0.98%  2.05%  1.93%  2.22%  0.97%
##    81:   1.49%  0.92%  1.97%  1.83%  2.17%  0.97%
##    82:   1.49%  0.95%  1.88%  1.78%  2.22%  0.97%
##    83:   1.50%  0.89%  1.97%  1.78%  2.22%  1.07%
##    84:   1.48%  0.89%  2.01%  1.68%  2.22%  0.97%
##    85:   1.46%  0.86%  1.92%  1.68%  2.22%  1.02%
##    86:   1.49%  0.86%  2.01%  1.73%  2.22%  1.07%
##    87:   1.47%  0.86%  1.97%  1.68%  2.22%  1.02%
##    88:   1.50%  0.89%  1.97%  1.78%  2.27%  1.02%
##    89:   1.42%  0.77%  1.92%  1.73%  2.11%  0.97%
##    90:   1.46%  0.80%  1.92%  1.78%  2.22%  1.02%
##    91:   1.41%  0.80%  1.88%  1.68%  2.22%  0.88%
##    92:   1.44%  0.80%  1.97%  1.73%  2.22%  0.88%
##    93:   1.39%  0.77%  1.76%  1.68%  2.22%  0.97%
##    94:   1.38%  0.77%  1.72%  1.63%  2.22%  0.97%
##    95:   1.46%  0.83%  1.72%  1.78%  2.27%  1.16%
##    96:   1.39%  0.80%  1.80%  1.63%  2.17%  0.97%
##    97:   1.40%  0.77%  1.80%  1.63%  2.17%  1.07%
##    98:   1.41%  0.77%  1.84%  1.78%  2.17%  0.93%
##    99:   1.44%  0.80%  1.80%  1.88%  2.17%  1.02%
##   100:   1.44%  0.83%  1.76%  1.88%  2.22%  0.97%
## ntree      OOB      1      2      3      4      5
##     1:   9.39%  7.10% 13.34% 12.03%  6.85%  8.33%
##     2:   9.83%  7.02% 13.08% 13.27%  8.42%  8.48%
##     3:   9.40%  6.16% 12.19% 12.82%  8.94%  8.42%
##     4:   9.32%  6.52% 12.75% 12.09%  9.06%  7.56%
##     5:   8.48%  5.84% 11.61% 11.57%  7.93%  6.77%
##     6:   7.89%  4.96% 10.85% 11.28%  7.34%  6.50%
##     7:   7.30%  5.02% 10.78%  9.34%  6.55%  5.80%
##     8:   6.34%  4.57%  9.30%  8.46%  6.04%  4.13%
##     9:   6.02%  3.56%  8.77%  8.45%  5.64%  4.86%
##    10:   5.29%  3.30%  8.10%  7.26%  4.78%  3.90%
##    11:   4.88%  3.17%  7.49%  6.51%  4.61%  3.41%
##    12:   4.41%  3.10%  6.73%  5.50%  4.24%  3.07%
##    13:   4.06%  2.88%  6.11%  4.90%  3.68%  3.21%
##    14:   3.85%  2.52%  6.15%  4.70%  3.57%  2.87%
##    15:   3.46%  2.46%  5.40%  4.26%  3.46%  2.15%
##    16:   3.05%  2.37%  4.52%  3.68%  3.16%  1.82%
##    17:   2.79%  1.97%  4.08%  3.49%  3.05%  1.77%
##    18:   2.69%  1.86%  4.38%  3.06%  2.95%  1.58%
##    19:   2.54%  1.83%  3.94%  2.72%  3.16%  1.39%
##    20:   2.44%  1.65%  3.59%  2.87%  2.95%  1.53%
##    21:   2.46%  1.82%  3.72%  2.82%  2.75%  1.48%
##    22:   2.28%  1.65%  3.42%  2.77%  2.70%  1.15%
##    23:   2.24%  1.50%  3.68%  2.72%  2.54%  1.10%
##    24:   2.13%  1.56%  3.15%  2.20%  2.85%  1.19%
##    25:   2.17%  1.35%  3.29%  2.63%  2.54%  1.43%
##    26:   2.16%  1.47%  3.29%  2.49%  2.54%  1.34%
##    27:   2.02%  1.41%  3.29%  2.10%  2.34%  1.24%
##    28:   2.06%  1.38%  3.46%  2.25%  2.29%  1.19%
##    29:   1.96%  1.26%  3.24%  2.15%  2.34%  1.15%
##    30:   2.00%  1.32%  3.20%  2.20%  2.34%  1.24%
##    31:   1.83%  1.26%  2.89%  2.01%  2.34%  0.91%
##    32:   1.88%  1.38%  3.15%  1.86%  2.24%  0.96%
##    33:   1.78%  1.23%  2.76%  1.96%  2.24%  1.00%
##    34:   1.85%  1.26%  2.89%  2.15%  2.34%  0.91%
##    35:   1.73%  1.20%  2.67%  1.72%  2.19%  1.15%
##    36:   1.72%  1.26%  2.50%  1.77%  2.29%  1.00%
##    37:   1.69%  1.20%  2.54%  1.82%  2.34%  0.81%
##    38:   1.66%  1.20%  2.63%  1.77%  2.19%  0.76%
##    39:   1.68%  1.14%  2.63%  1.86%  2.24%  0.81%
##    40:   1.60%  1.14%  2.50%  1.82%  2.09%  0.72%
##    41:   1.62%  1.26%  2.45%  1.58%  2.19%  0.81%
##    42:   1.62%  1.23%  2.58%  1.63%  2.04%  0.81%
##    43:   1.55%  1.11%  2.54%  1.63%  2.04%  0.67%
##    44:   1.49%  1.14%  2.37%  1.53%  1.93%  0.67%
##    45:   1.53%  1.11%  2.37%  1.63%  1.98%  0.76%
##    46:   1.50%  1.02%  2.37%  1.58%  2.04%  0.76%
##    47:   1.57%  1.11%  2.45%  1.72%  2.04%  0.76%
##    48:   1.50%  1.05%  2.19%  1.67%  2.24%  0.62%
##    49:   1.53%  1.20%  2.15%  1.48%  2.39%  0.62%
##    50:   1.49%  1.08%  2.32%  1.43%  2.14%  0.72%
##    51:   1.44%  0.99%  2.32%  1.29%  2.14%  0.67%
##    52:   1.44%  0.99%  2.37%  1.34%  2.09%  0.67%
##    53:   1.42%  0.93%  2.37%  1.24%  2.04%  0.76%
##    54:   1.47%  1.02%  2.32%  1.53%  1.98%  0.72%
##    55:   1.43%  1.02%  2.23%  1.20%  2.09%  0.81%
##    56:   1.42%  1.02%  2.15%  1.34%  2.09%  0.72%
##    57:   1.45%  1.02%  2.28%  1.15%  2.24%  0.81%
##    58:   1.45%  0.93%  2.23%  1.29%  2.24%  0.86%
##    59:   1.49%  0.87%  2.54%  1.29%  2.29%  0.81%
##    60:   1.41%  0.90%  2.28%  1.24%  2.19%  0.72%
##    61:   1.49%  0.99%  2.41%  1.34%  2.24%  0.72%
##    62:   1.52%  0.93%  2.45%  1.39%  2.34%  0.81%
##    63:   1.49%  0.96%  2.45%  1.24%  2.34%  0.76%
##    64:   1.51%  0.93%  2.54%  1.24%  2.39%  0.76%
##    65:   1.41%  0.90%  2.32%  1.20%  2.29%  0.62%
##    66:   1.48%  0.96%  2.41%  1.34%  2.29%  0.67%
##    67:   1.44%  0.93%  2.28%  1.34%  2.34%  0.62%
##    68:   1.43%  0.90%  2.32%  1.29%  2.24%  0.67%
##    69:   1.39%  0.84%  2.23%  1.29%  2.24%  0.67%
##    70:   1.47%  0.99%  2.23%  1.34%  2.34%  0.72%
##    71:   1.46%  0.90%  2.32%  1.39%  2.29%  0.72%
##    72:   1.41%  0.90%  2.23%  1.24%  2.29%  0.67%
##    73:   1.44%  0.87%  2.32%  1.24%  2.34%  0.72%
##    74:   1.46%  0.99%  2.32%  1.29%  2.29%  0.67%
##    75:   1.43%  0.90%  2.37%  1.24%  2.29%  0.62%
##    76:   1.40%  0.90%  2.23%  1.24%  2.24%  0.67%
##    77:   1.37%  0.90%  2.19%  1.20%  2.19%  0.62%
##    78:   1.40%  0.87%  2.32%  1.24%  2.24%  0.62%
##    79:   1.42%  1.02%  2.19%  1.29%  2.24%  0.57%
##    80:   1.41%  0.90%  2.28%  1.34%  2.24%  0.57%
##    81:   1.41%  0.84%  2.32%  1.34%  2.24%  0.62%
##    82:   1.38%  0.81%  2.32%  1.29%  2.19%  0.57%
##    83:   1.36%  0.84%  2.23%  1.29%  2.14%  0.57%
##    84:   1.39%  0.84%  2.32%  1.29%  2.19%  0.62%
##    85:   1.39%  0.90%  2.23%  1.34%  2.14%  0.62%
##    86:   1.40%  0.81%  2.32%  1.39%  2.19%  0.62%
##    87:   1.38%  0.78%  2.28%  1.34%  2.19%  0.62%
##    88:   1.39%  0.81%  2.28%  1.34%  2.19%  0.67%
##    89:   1.38%  0.75%  2.37%  1.34%  2.14%  0.62%
##    90:   1.38%  0.75%  2.41%  1.29%  2.19%  0.62%
##    91:   1.42%  0.81%  2.41%  1.34%  2.24%  0.62%
##    92:   1.40%  0.81%  2.37%  1.34%  2.19%  0.62%
##    93:   1.40%  0.78%  2.41%  1.39%  2.14%  0.62%
##    94:   1.43%  0.87%  2.37%  1.39%  2.19%  0.62%
##    95:   1.44%  0.90%  2.41%  1.39%  2.19%  0.62%
##    96:   1.39%  0.78%  2.28%  1.34%  2.29%  0.62%
##    97:   1.42%  0.84%  2.37%  1.39%  2.19%  0.62%
##    98:   1.42%  0.84%  2.32%  1.39%  2.24%  0.62%
##    99:   1.38%  0.84%  2.23%  1.39%  2.14%  0.62%
##   100:   1.41%  0.84%  2.41%  1.34%  2.14%  0.62%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.64%  5.51%  9.59%  9.08%  8.25%  6.84%
##     2:   7.96%  5.57%  9.98%  9.55%  8.01%  7.81%
##     3:   8.03%  6.60%  9.85%  9.99%  8.18%  6.22%
##     4:   7.25%  5.71%  8.69%  9.06%  7.62%  5.92%
##     5:   6.62%  4.95%  7.90%  8.22%  7.03%  5.88%
##     6:   6.20%  5.14%  7.43%  7.17%  6.85%  4.93%
##     7:   5.81%  4.93%  6.57%  6.83%  6.49%  4.70%
##     8:   5.29%  3.98%  6.64%  6.44%  5.92%  4.17%
##     9:   4.95%  3.74%  6.57%  5.99%  5.53%  3.54%
##    10:   4.66%  3.48%  5.88%  5.61%  5.91%  3.09%
##    11:   4.34%  3.22%  5.87%  5.15%  4.97%  3.08%
##    12:   4.24%  3.10%  5.73%  5.19%  4.80%  2.98%
##    13:   3.94%  2.76%  5.32%  4.89%  4.54%  2.78%
##    14:   3.70%  2.76%  4.70%  4.74%  4.43%  2.39%
##    15:   3.42%  2.64%  4.52%  3.73%  4.33%  2.30%
##    16:   3.26%  2.43%  4.38%  3.54%  4.12%  2.29%
##    17:   3.14%  2.51%  3.81%  3.63%  4.07%  2.05%
##    18:   2.85%  1.89%  3.33%  3.63%  3.82%  2.20%
##    19:   2.64%  1.77%  3.37%  2.96%  3.51%  2.10%
##    20:   2.66%  1.65%  3.64%  3.01%  3.61%  1.96%
##    21:   2.58%  1.79%  3.55%  2.72%  3.31%  1.96%
##    22:   2.56%  1.73%  3.50%  3.01%  3.41%  1.62%
##    23:   2.51%  1.82%  3.42%  2.92%  3.00%  1.77%
##    24:   2.44%  1.79%  3.37%  2.58%  2.95%  1.82%
##    25:   2.42%  1.65%  3.29%  2.72%  3.21%  1.67%
##    26:   2.45%  1.76%  3.46%  2.49%  3.10%  1.82%
##    27:   2.31%  1.62%  3.20%  2.44%  3.05%  1.62%
##    28:   2.34%  1.65%  3.29%  2.72%  2.85%  1.58%
##    29:   2.39%  1.59%  3.33%  2.82%  3.00%  1.67%
##    30:   2.38%  1.62%  3.37%  2.58%  3.16%  1.58%
##    31:   2.35%  1.68%  3.29%  2.58%  3.05%  1.53%
##    32:   2.42%  1.85%  3.37%  2.68%  2.90%  1.58%
##    33:   2.34%  1.65%  3.42%  2.53%  2.95%  1.48%
##    34:   2.33%  1.71%  3.29%  2.58%  2.95%  1.43%
##    35:   2.32%  1.62%  3.29%  2.58%  2.95%  1.53%
##    36:   2.34%  1.71%  3.46%  2.49%  2.90%  1.48%
##    37:   2.31%  1.76%  3.37%  2.44%  2.85%  1.39%
##    38:   2.32%  1.71%  3.42%  2.34%  2.95%  1.48%
##    39:   2.28%  1.65%  3.33%  2.49%  2.90%  1.39%
##    40:   2.30%  1.68%  3.50%  2.44%  2.85%  1.34%
##    41:   2.24%  1.56%  3.42%  2.39%  2.80%  1.39%
##    42:   2.23%  1.50%  3.33%  2.44%  2.85%  1.43%
##    43:   2.17%  1.44%  3.15%  2.39%  2.85%  1.43%
##    44:   2.19%  1.38%  3.24%  2.29%  2.90%  1.58%
##    45:   2.15%  1.32%  3.42%  2.15%  2.80%  1.48%
##    46:   2.15%  1.50%  3.29%  2.15%  2.65%  1.48%
##    47:   2.13%  1.38%  3.20%  2.20%  2.80%  1.48%
##    48:   2.20%  1.56%  3.46%  2.10%  2.75%  1.43%
##    49:   2.11%  1.38%  3.24%  2.06%  2.80%  1.43%
##    50:   2.11%  1.47%  3.11%  2.20%  2.75%  1.39%
##    51:   2.15%  1.53%  3.24%  2.15%  2.65%  1.48%
##    52:   2.10%  1.44%  2.98%  2.29%  2.70%  1.43%
##    53:   2.06%  1.35%  2.93%  2.25%  2.75%  1.43%
##    54:   2.06%  1.38%  2.93%  2.20%  2.75%  1.39%
##    55:   2.00%  1.47%  2.80%  1.86%  2.65%  1.48%
##    56:   1.97%  1.35%  2.85%  1.96%  2.65%  1.39%
##    57:   1.93%  1.38%  2.63%  1.91%  2.60%  1.43%
##    58:   1.94%  1.38%  2.58%  2.01%  2.65%  1.39%
##    59:   1.92%  1.32%  2.76%  1.86%  2.54%  1.43%
##    60:   1.94%  1.29%  2.76%  2.06%  2.54%  1.43%
##    61:   1.91%  1.29%  2.72%  2.06%  2.49%  1.34%
##    62:   1.91%  1.23%  2.76%  1.96%  2.49%  1.48%
##    63:   1.91%  1.29%  2.72%  1.96%  2.54%  1.39%
##    64:   1.90%  1.26%  2.80%  1.96%  2.49%  1.34%
##    65:   1.92%  1.29%  2.85%  1.96%  2.49%  1.34%
##    66:   1.95%  1.29%  2.85%  2.01%  2.60%  1.39%
##    67:   1.89%  1.29%  2.67%  1.91%  2.54%  1.39%
##    68:   1.91%  1.29%  2.72%  1.77%  2.65%  1.48%
##    69:   1.87%  1.29%  2.67%  1.72%  2.54%  1.43%
##    70:   1.89%  1.32%  2.63%  1.77%  2.54%  1.48%
##    71:   1.88%  1.38%  2.50%  1.82%  2.54%  1.43%
##    72:   1.88%  1.38%  2.50%  1.86%  2.49%  1.43%
##    73:   1.90%  1.41%  2.63%  1.86%  2.49%  1.39%
##    74:   1.93%  1.47%  2.67%  1.72%  2.54%  1.48%
##    75:   1.92%  1.41%  2.54%  1.86%  2.60%  1.48%
##    76:   1.94%  1.44%  2.67%  1.91%  2.49%  1.48%
##    77:   1.95%  1.41%  2.72%  2.01%  2.54%  1.39%
##    78:   1.92%  1.41%  2.63%  1.96%  2.44%  1.43%
##    79:   1.89%  1.38%  2.76%  1.82%  2.44%  1.34%
##    80:   1.94%  1.44%  2.76%  1.91%  2.49%  1.39%
##    81:   1.90%  1.38%  2.67%  1.86%  2.49%  1.39%
##    82:   1.91%  1.35%  2.76%  1.91%  2.49%  1.34%
##    83:   1.91%  1.38%  2.58%  1.91%  2.54%  1.43%
##    84:   1.89%  1.41%  2.58%  1.82%  2.60%  1.34%
##    85:   1.90%  1.35%  2.63%  1.91%  2.54%  1.39%
##    86:   1.93%  1.35%  2.72%  1.96%  2.54%  1.39%
##    87:   1.89%  1.35%  2.58%  1.91%  2.54%  1.39%
##    88:   1.92%  1.38%  2.67%  1.91%  2.49%  1.43%
##    89:   1.87%  1.29%  2.54%  1.96%  2.54%  1.34%
##    90:   1.89%  1.32%  2.54%  2.01%  2.54%  1.39%
##    91:   1.87%  1.23%  2.67%  1.86%  2.49%  1.43%
##    92:   1.88%  1.29%  2.63%  1.91%  2.54%  1.34%
##    93:   1.94%  1.26%  2.80%  1.96%  2.54%  1.53%
##    94:   1.95%  1.35%  2.85%  1.91%  2.54%  1.43%
##    95:   1.89%  1.26%  2.76%  1.82%  2.49%  1.48%
##    96:   1.88%  1.26%  2.80%  1.77%  2.54%  1.34%
##    97:   1.89%  1.26%  2.76%  1.82%  2.54%  1.39%
##    98:   1.90%  1.32%  2.72%  1.82%  2.60%  1.39%
##    99:   1.94%  1.29%  2.80%  1.96%  2.60%  1.43%
##   100:   1.94%  1.35%  2.72%  1.91%  2.60%  1.43%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   8.58%  6.02% 10.41% 11.95%  9.33%  6.60%
##     2:   8.26%  5.15% 10.90% 10.17%  9.14%  7.56%
##     3:   7.86%  5.10%  9.92% 10.31%  8.54%  6.93%
##     4:   7.63%  4.91%  9.93%  9.38%  8.52%  6.90%
##     5:   7.14%  4.47%  9.64%  8.55%  8.01%  6.44%
##     6:   6.53%  4.08%  9.12%  8.08%  7.29%  5.32%
##     7:   5.96%  3.98%  7.95%  7.40%  6.59%  4.92%
##     8:   5.52%  3.75%  7.59%  6.73%  6.24%  4.22%
##     9:   5.03%  3.46%  7.23%  5.91%  5.58%  3.70%
##    10:   4.79%  3.57%  6.71%  5.41%  5.44%  3.38%
##    11:   4.48%  3.10%  6.39%  5.29%  4.75%  3.51%
##    12:   4.15%  2.85%  5.46%  4.56%  5.10%  3.50%
##    13:   3.87%  2.82%  5.36%  4.17%  4.44%  3.06%
##    14:   3.77%  2.40%  5.53%  3.98%  4.74%  2.92%
##    15:   3.48%  2.43%  4.96%  3.88%  4.02%  2.63%
##    16:   3.42%  2.43%  4.95%  3.68%  4.02%  2.49%
##    17:   3.25%  2.33%  4.69%  3.40%  3.87%  2.44%
##    18:   3.07%  2.09%  4.43%  3.35%  3.77%  2.20%
##    19:   2.96%  2.21%  4.03%  3.16%  3.61%  2.15%
##    20:   2.93%  1.91%  3.72%  3.35%  4.12%  2.15%
##    21:   2.95%  1.97%  4.07%  3.40%  3.77%  2.05%
##    22:   2.79%  1.82%  3.77%  2.77%  4.02%  2.10%
##    23:   2.79%  1.79%  3.77%  3.06%  3.87%  2.01%
##    24:   2.69%  1.73%  3.59%  3.01%  3.72%  1.96%
##    25:   2.63%  1.65%  3.46%  2.97%  3.61%  2.05%
##    26:   2.51%  1.62%  3.11%  2.92%  3.46%  1.96%
##    27:   2.46%  1.53%  3.15%  2.87%  3.56%  1.77%
##    28:   2.45%  1.47%  3.42%  2.77%  3.31%  1.86%
##    29:   2.36%  1.56%  3.20%  2.63%  3.26%  1.62%
##    30:   2.33%  1.44%  2.98%  2.92%  3.16%  1.67%
##    31:   2.28%  1.35%  3.02%  2.82%  3.05%  1.67%
##    32:   2.31%  1.50%  3.15%  2.72%  3.05%  1.58%
##    33:   2.31%  1.38%  3.07%  2.72%  3.36%  1.58%
##    34:   2.22%  1.32%  2.98%  2.68%  3.05%  1.58%
##    35:   2.16%  1.38%  2.76%  2.49%  2.95%  1.67%
##    36:   2.15%  1.35%  2.80%  2.49%  3.10%  1.48%
##    37:   2.17%  1.26%  2.89%  2.53%  3.10%  1.62%
##    38:   2.10%  1.26%  2.98%  2.25%  2.85%  1.62%
##    39:   2.17%  1.29%  3.11%  2.34%  2.90%  1.72%
##    40:   2.11%  1.29%  2.93%  2.25%  2.80%  1.72%
##    41:   2.11%  1.26%  3.02%  2.15%  2.85%  1.72%
##    42:   2.07%  1.29%  3.02%  2.10%  2.70%  1.67%
##    43:   2.00%  1.26%  2.89%  2.15%  2.65%  1.48%
##    44:   1.96%  1.29%  2.80%  1.96%  2.54%  1.58%
##    45:   1.99%  1.23%  2.98%  2.01%  2.54%  1.58%
##    46:   2.00%  1.29%  2.85%  2.06%  2.65%  1.58%
##    47:   1.96%  1.26%  2.72%  2.06%  2.65%  1.53%
##    48:   1.94%  1.29%  2.54%  1.91%  2.75%  1.58%
##    49:   1.95%  1.14%  2.54%  2.20%  2.75%  1.62%
##    50:   1.90%  1.14%  2.45%  2.10%  2.70%  1.58%
##    51:   1.89%  1.20%  2.32%  2.01%  2.80%  1.58%
##    52:   1.92%  1.14%  2.45%  2.01%  2.80%  1.67%
##    53:   1.94%  1.14%  2.58%  2.06%  2.75%  1.62%
##    54:   1.90%  1.11%  2.58%  1.91%  2.85%  1.53%
##    55:   1.92%  1.17%  2.50%  2.10%  2.75%  1.53%
##    56:   1.92%  1.14%  2.58%  2.01%  2.75%  1.58%
##    57:   1.89%  1.11%  2.54%  2.06%  2.65%  1.53%
##    58:   1.91%  1.11%  2.63%  2.10%  2.70%  1.48%
##    59:   1.92%  1.17%  2.54%  2.25%  2.70%  1.39%
##    60:   1.94%  1.14%  2.72%  2.20%  2.70%  1.43%
##    61:   1.94%  1.08%  2.80%  2.10%  2.65%  1.53%
##    62:   1.90%  1.11%  2.58%  2.15%  2.65%  1.48%
##    63:   1.86%  1.05%  2.63%  2.01%  2.60%  1.48%
##    64:   1.83%  1.05%  2.54%  2.06%  2.54%  1.43%
##    65:   1.82%  1.02%  2.58%  1.96%  2.65%  1.34%
##    66:   1.86%  1.08%  2.67%  1.91%  2.75%  1.34%
##    67:   1.85%  0.99%  2.67%  1.91%  2.60%  1.58%
##    68:   1.90%  1.08%  2.76%  1.91%  2.75%  1.48%
##    69:   1.89%  0.99%  2.85%  2.01%  2.65%  1.48%
##    70:   1.89%  0.99%  2.76%  2.01%  2.70%  1.53%
##    71:   1.89%  0.99%  2.76%  2.01%  2.70%  1.53%
##    72:   1.89%  1.08%  2.67%  2.01%  2.65%  1.48%
##    73:   1.85%  1.02%  2.58%  2.01%  2.70%  1.43%
##    74:   1.89%  0.99%  2.72%  2.06%  2.80%  1.43%
##    75:   1.83%  0.96%  2.67%  2.01%  2.65%  1.39%
##    76:   1.86%  0.99%  2.67%  2.01%  2.65%  1.48%
##    77:   1.87%  1.08%  2.58%  2.06%  2.65%  1.43%
##    78:   1.88%  1.17%  2.54%  2.01%  2.65%  1.43%
##    79:   1.87%  1.11%  2.54%  2.01%  2.65%  1.48%
##    80:   1.83%  1.02%  2.45%  2.01%  2.75%  1.43%
##    81:   1.80%  1.02%  2.37%  1.96%  2.65%  1.48%
##    82:   1.81%  1.08%  2.37%  1.96%  2.65%  1.43%
##    83:   1.84%  1.11%  2.41%  1.96%  2.65%  1.53%
##    84:   1.83%  1.11%  2.41%  2.01%  2.65%  1.43%
##    85:   1.79%  1.05%  2.32%  1.91%  2.60%  1.53%
##    86:   1.76%  1.05%  2.15%  1.96%  2.60%  1.48%
##    87:   1.79%  1.08%  2.32%  1.96%  2.60%  1.43%
##    88:   1.82%  1.08%  2.45%  2.01%  2.54%  1.43%
##    89:   1.83%  1.05%  2.45%  2.06%  2.60%  1.48%
##    90:   1.83%  1.02%  2.50%  2.10%  2.60%  1.43%
##    91:   1.83%  1.08%  2.41%  2.06%  2.60%  1.48%
##    92:   1.82%  1.11%  2.37%  2.01%  2.60%  1.43%
##    93:   1.82%  1.11%  2.45%  1.96%  2.54%  1.43%
##    94:   1.77%  1.11%  2.32%  1.91%  2.49%  1.43%
##    95:   1.81%  1.11%  2.32%  2.01%  2.54%  1.48%
##    96:   1.79%  1.11%  2.32%  1.91%  2.54%  1.48%
##    97:   1.79%  1.08%  2.41%  1.96%  2.44%  1.48%
##    98:   1.78%  1.08%  2.37%  1.91%  2.49%  1.48%
##    99:   1.77%  1.08%  2.32%  1.91%  2.54%  1.43%
##   100:   1.75%  1.08%  2.32%  1.86%  2.44%  1.43%
## ntree      OOB      1      2      3      4      5
##     1:  10.76%  6.98% 15.15% 11.99% 11.99%  9.53%
##     2:  10.95%  6.48% 14.26% 13.74% 12.10% 10.46%
##     3:  10.30%  5.55% 14.10% 12.00% 12.26% 10.15%
##     4:   9.80%  6.08% 13.97% 10.62% 12.21%  8.25%
##     5:   9.43%  5.40% 14.24% 10.69% 11.11%  7.99%
##     6:   8.85%  5.38% 12.90% 10.47% 10.67%  6.88%
##     7:   8.00%  4.97% 11.47%  9.12% 10.04%  6.25%
##     8:   7.49%  4.62% 10.37%  9.14%  8.82%  6.19%
##     9:   6.43%  4.11%  9.00%  7.71%  7.67%  5.04%
##    10:   6.01%  3.85%  8.17%  7.41%  6.80%  5.05%
##    11:   5.67%  3.71%  8.05%  7.10%  6.21%  4.40%
##    12:   4.98%  3.19%  6.70%  6.05%  5.48%  4.48%
##    13:   4.39%  2.85%  6.27%  5.04%  5.11%  3.57%
##    14:   4.11%  2.81%  5.86%  4.47%  4.69%  3.48%
##    15:   3.94%  2.81%  5.53%  4.56%  4.64%  2.84%
##    16:   3.49%  2.29%  5.11%  4.04%  4.06%  2.66%
##    17:   3.29%  2.14%  4.84%  3.67%  3.95%  2.52%
##    18:   3.18%  2.14%  4.61%  3.52%  3.85%  2.43%
##    19:   3.24%  2.29%  4.98%  3.62%  3.75%  2.16%
##    20:   2.96%  2.05%  4.47%  3.20%  3.64%  1.98%
##    21:   2.85%  2.23%  4.29%  2.87%  3.33%  1.94%
##    22:   2.76%  1.87%  4.75%  2.91%  3.12%  1.67%
##    23:   2.62%  1.84%  4.11%  2.82%  3.28%  1.58%
##    24:   2.54%  1.81%  3.93%  2.82%  2.97%  1.62%
##    25:   2.45%  1.72%  4.16%  2.63%  2.81%  1.40%
##    26:   2.51%  1.48%  4.20%  2.91%  3.02%  1.53%
##    27:   2.31%  1.48%  3.93%  2.54%  2.81%  1.31%
##    28:   2.22%  1.36%  4.02%  2.16%  2.65%  1.44%
##    29:   2.11%  1.30%  3.70%  2.25%  2.71%  1.13%
##    30:   2.03%  1.21%  3.42%  2.11%  2.60%  1.31%
##    31:   2.06%  1.39%  3.20%  2.35%  2.55%  1.22%
##    32:   1.96%  1.27%  3.01%  2.40%  2.65%  0.95%
##    33:   1.93%  1.24%  2.83%  2.35%  2.65%  1.04%
##    34:   1.96%  1.24%  3.15%  2.40%  2.55%  0.95%
##    35:   1.89%  1.06%  3.01%  2.30%  2.71%  0.90%
##    36:   1.84%  1.09%  2.92%  1.97%  2.71%  1.04%
##    37:   1.90%  1.12%  3.06%  2.11%  2.71%  1.04%
##    38:   1.87%  1.09%  3.06%  2.21%  2.55%  0.95%
##    39:   1.88%  1.12%  2.97%  2.07%  2.81%  0.95%
##    40:   1.75%  1.12%  2.79%  1.78%  2.65%  0.86%
##    41:   1.80%  1.12%  2.79%  1.88%  2.81%  0.90%
##    42:   1.79%  1.09%  2.97%  1.74%  2.81%  0.86%
##    43:   1.74%  1.09%  2.79%  1.64%  2.76%  0.90%
##    44:   1.80%  1.12%  2.88%  1.74%  2.81%  0.95%
##    45:   1.72%  1.12%  2.65%  1.55%  2.81%  0.90%
##    46:   1.77%  1.12%  2.88%  1.78%  2.81%  0.77%
##    47:   1.77%  1.03%  2.97%  1.78%  2.81%  0.77%
##    48:   1.72%  0.99%  2.88%  1.74%  2.76%  0.77%
##    49:   1.60%  0.87%  2.65%  1.69%  2.65%  0.68%
##    50:   1.61%  0.87%  2.79%  1.60%  2.65%  0.68%
##    51:   1.66%  0.96%  2.74%  1.64%  2.71%  0.72%
##    52:   1.63%  0.90%  2.88%  1.41%  2.65%  0.81%
##    53:   1.65%  0.96%  2.69%  1.55%  2.71%  0.81%
##    54:   1.60%  0.96%  2.69%  1.50%  2.55%  0.72%
##    55:   1.60%  0.87%  2.74%  1.55%  2.60%  0.77%
##    56:   1.63%  0.93%  2.60%  1.64%  2.65%  0.81%
##    57:   1.60%  0.93%  2.51%  1.55%  2.71%  0.77%
##    58:   1.59%  0.96%  2.74%  1.55%  2.45%  0.68%
##    59:   1.61%  0.90%  2.69%  1.64%  2.50%  0.81%
##    60:   1.66%  0.96%  2.74%  1.60%  2.60%  0.86%
##    61:   1.60%  0.84%  2.79%  1.69%  2.39%  0.77%
##    62:   1.59%  0.87%  2.69%  1.64%  2.39%  0.81%
##    63:   1.60%  0.87%  2.60%  1.64%  2.39%  0.95%
##    64:   1.59%  0.87%  2.69%  1.64%  2.39%  0.81%
##    65:   1.60%  0.87%  2.74%  1.64%  2.39%  0.81%
##    66:   1.58%  0.87%  2.65%  1.60%  2.39%  0.86%
##    67:   1.57%  0.93%  2.60%  1.60%  2.34%  0.81%
##    68:   1.53%  0.78%  2.60%  1.64%  2.39%  0.72%
##    69:   1.56%  0.72%  2.79%  1.64%  2.39%  0.81%
##    70:   1.52%  0.81%  2.47%  1.60%  2.34%  0.86%
##    71:   1.56%  0.81%  2.65%  1.69%  2.45%  0.72%
##    72:   1.58%  0.81%  2.56%  1.69%  2.50%  0.86%
##    73:   1.54%  0.78%  2.56%  1.69%  2.45%  0.72%
##    74:   1.52%  0.78%  2.60%  1.60%  2.45%  0.68%
##    75:   1.49%  0.75%  2.60%  1.46%  2.45%  0.72%
##    76:   1.49%  0.75%  2.42%  1.50%  2.55%  0.72%
##    77:   1.53%  0.66%  2.65%  1.64%  2.55%  0.72%
##    78:   1.57%  0.81%  2.60%  1.69%  2.55%  0.72%
##    79:   1.55%  0.75%  2.51%  1.69%  2.60%  0.77%
##    80:   1.46%  0.72%  2.19%  1.64%  2.55%  0.72%
##    81:   1.49%  0.72%  2.47%  1.60%  2.55%  0.68%
##    82:   1.46%  0.75%  2.42%  1.50%  2.50%  0.63%
##    83:   1.49%  0.75%  2.47%  1.50%  2.55%  0.68%
##    84:   1.49%  0.75%  2.51%  1.55%  2.45%  0.68%
##    85:   1.51%  0.78%  2.51%  1.60%  2.50%  0.68%
##    86:   1.51%  0.72%  2.47%  1.60%  2.60%  0.72%
##    87:   1.55%  0.78%  2.65%  1.55%  2.55%  0.72%
##    88:   1.52%  0.72%  2.65%  1.60%  2.55%  0.63%
##    89:   1.50%  0.69%  2.51%  1.55%  2.60%  0.72%
##    90:   1.50%  0.75%  2.51%  1.55%  2.50%  0.72%
##    91:   1.49%  0.69%  2.51%  1.50%  2.55%  0.72%
##    92:   1.51%  0.72%  2.56%  1.55%  2.60%  0.68%
##    93:   1.50%  0.72%  2.56%  1.60%  2.50%  0.68%
##    94:   1.54%  0.78%  2.56%  1.64%  2.60%  0.63%
##    95:   1.54%  0.72%  2.65%  1.64%  2.55%  0.68%
##    96:   1.51%  0.78%  2.47%  1.60%  2.55%  0.68%
##    97:   1.49%  0.66%  2.65%  1.50%  2.50%  0.68%
##    98:   1.45%  0.63%  2.51%  1.50%  2.45%  0.72%
##    99:   1.47%  0.63%  2.65%  1.46%  2.50%  0.68%
##   100:   1.49%  0.66%  2.60%  1.50%  2.60%  0.68%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   9.25%  5.87% 15.33% 11.18%  7.91%  7.12%
##     2:   8.90%  5.78% 15.68% 10.76%  7.50%  5.99%
##     3:   8.56%  5.59% 14.16%  9.58%  8.76%  6.21%
##     4:   8.13%  6.08% 12.68%  8.85%  8.08%  5.99%
##     5:   7.56%  5.57% 11.82%  8.08%  8.21%  5.27%
##     6:   7.07%  5.04% 11.19%  7.43%  7.46%  5.36%
##     7:   6.59%  4.84% 10.18%  7.25%  6.65%  4.97%
##     8:   5.87%  3.88%  9.30%  6.77%  6.63%  3.96%
##     9:   5.52%  3.62%  8.37%  6.49%  5.66%  4.52%
##    10:   5.09%  3.35%  7.89%  6.26%  5.46%  3.50%
##    11:   4.63%  2.97%  7.32%  6.10%  4.51%  3.13%
##    12:   4.31%  3.00%  6.73%  5.10%  4.81%  2.72%
##    13:   4.10%  2.60%  6.22%  5.14%  4.75%  2.71%
##    14:   3.72%  2.45%  5.49%  4.66%  4.44%  2.35%
##    15:   3.50%  2.23%  5.26%  4.28%  4.43%  2.12%
##    16:   3.54%  2.17%  5.67%  4.42%  4.33%  1.98%
##    17:   3.22%  1.99%  4.98%  3.99%  3.90%  1.98%
##    18:   3.16%  1.90%  4.89%  3.99%  3.85%  1.94%
##    19:   3.13%  1.87%  5.03%  3.95%  3.69%  1.85%
##    20:   2.99%  1.75%  4.84%  3.67%  3.59%  1.85%
##    21:   2.99%  1.93%  4.80%  3.81%  3.43%  1.62%
##    22:   2.79%  1.63%  4.52%  3.62%  3.28%  1.58%
##    23:   2.78%  1.51%  4.47%  3.52%  3.49%  1.67%
##    24:   2.62%  1.60%  4.06%  3.33%  3.23%  1.53%
##    25:   2.58%  1.60%  4.16%  3.24%  3.38%  1.17%
##    26:   2.62%  1.42%  4.02%  3.62%  3.59%  1.22%
##    27:   2.51%  1.48%  3.70%  3.43%  3.28%  1.35%
##    28:   2.60%  1.66%  3.93%  3.33%  3.33%  1.35%
##    29:   2.45%  1.57%  3.42%  3.24%  3.49%  1.17%
##    30:   2.44%  1.51%  3.61%  3.29%  3.17%  1.22%
##    31:   2.35%  1.42%  3.33%  3.33%  3.07%  1.22%
##    32:   2.24%  1.57%  3.01%  2.82%  3.07%  1.22%
##    33:   2.26%  1.45%  3.20%  2.91%  3.07%  1.22%
##    34:   2.20%  1.42%  3.11%  2.87%  3.12%  1.04%
##    35:   2.21%  1.45%  3.15%  2.91%  3.02%  1.04%
##    36:   2.13%  1.36%  3.15%  2.72%  3.02%  0.95%
##    37:   2.11%  1.36%  3.06%  2.72%  2.97%  0.95%
##    38:   2.17%  1.30%  3.38%  2.82%  3.07%  0.90%
##    39:   2.18%  1.42%  3.20%  2.68%  3.23%  0.95%
##    40:   2.14%  1.39%  3.15%  2.72%  3.07%  0.90%
##    41:   2.09%  1.27%  2.97%  2.77%  2.97%  1.04%
##    42:   2.00%  1.24%  3.01%  2.49%  2.86%  0.95%
##    43:   1.98%  1.21%  2.79%  2.72%  2.76%  0.95%
##    44:   2.00%  1.30%  2.83%  2.82%  2.71%  0.86%
##    45:   1.98%  1.18%  3.11%  2.68%  2.60%  0.86%
##    46:   2.03%  1.33%  3.01%  2.72%  2.71%  0.86%
##    47:   2.01%  1.24%  2.97%  2.68%  2.71%  0.99%
##    48:   2.03%  1.36%  2.92%  2.77%  2.71%  0.86%
##    49:   2.00%  1.24%  2.97%  2.77%  2.60%  0.95%
##    50:   2.02%  1.33%  2.83%  2.68%  2.76%  0.99%
##    51:   2.01%  1.30%  2.92%  2.68%  2.76%  0.90%
##    52:   2.09%  1.27%  3.15%  2.68%  2.86%  1.04%
##    53:   2.09%  1.33%  3.15%  2.63%  2.86%  0.99%
##    54:   2.00%  1.21%  2.92%  2.54%  2.97%  0.95%
##    55:   2.07%  1.36%  3.06%  2.54%  2.91%  0.99%
##    56:   2.03%  1.21%  3.01%  2.40%  3.07%  1.04%
##    57:   2.00%  1.39%  3.01%  2.30%  2.81%  0.90%
##    58:   1.99%  1.30%  3.06%  2.40%  2.71%  0.95%
##    59:   1.99%  1.27%  2.88%  2.40%  2.97%  0.95%
##    60:   1.99%  1.33%  2.88%  2.35%  2.97%  0.90%
##    61:   1.94%  1.24%  2.83%  2.40%  2.81%  0.90%
##    62:   2.00%  1.24%  2.83%  2.44%  3.07%  0.95%
##    63:   1.95%  1.21%  2.69%  2.49%  2.97%  0.95%
##    64:   1.94%  1.21%  2.74%  2.44%  2.97%  0.86%
##    65:   1.96%  1.21%  2.79%  2.54%  2.97%  0.86%
##    66:   2.00%  1.24%  2.74%  2.72%  2.97%  0.86%
##    67:   1.93%  1.21%  2.60%  2.63%  2.86%  0.86%
##    68:   1.92%  1.18%  2.60%  2.63%  2.86%  0.86%
##    69:   1.94%  1.24%  2.56%  2.63%  2.86%  0.90%
##    70:   1.90%  1.27%  2.37%  2.58%  2.86%  0.90%
##    71:   1.94%  1.24%  2.60%  2.63%  2.86%  0.90%
##    72:   1.94%  1.30%  2.65%  2.54%  2.86%  0.81%
##    73:   1.94%  1.21%  2.60%  2.63%  2.81%  0.95%
##    74:   1.92%  1.21%  2.47%  2.72%  2.76%  0.95%
##    75:   1.93%  1.21%  2.60%  2.63%  2.86%  0.86%
##    76:   1.94%  1.18%  2.65%  2.63%  2.81%  0.99%
##    77:   1.89%  1.18%  2.42%  2.49%  2.97%  0.90%
##    78:   1.91%  1.12%  2.56%  2.54%  2.97%  0.95%
##    79:   1.86%  1.27%  2.37%  2.40%  2.81%  0.90%
##    80:   1.91%  1.30%  2.47%  2.54%  2.81%  0.90%
##    81:   1.93%  1.33%  2.56%  2.58%  2.71%  0.90%
##    82:   1.86%  1.24%  2.47%  2.54%  2.65%  0.86%
##    83:   1.94%  1.18%  2.65%  2.58%  2.91%  0.90%
##    84:   1.94%  1.33%  2.56%  2.54%  2.71%  0.99%
##    85:   1.89%  1.18%  2.56%  2.54%  2.71%  0.95%
##    86:   1.89%  1.18%  2.51%  2.49%  2.71%  1.04%
##    87:   1.89%  1.21%  2.51%  2.49%  2.76%  0.95%
##    88:   1.87%  1.21%  2.56%  2.40%  2.76%  0.90%
##    89:   1.89%  1.21%  2.51%  2.44%  2.76%  0.99%
##    90:   1.83%  1.12%  2.51%  2.40%  2.71%  0.90%
##    91:   1.86%  1.15%  2.56%  2.40%  2.81%  0.90%
##    92:   1.86%  1.12%  2.60%  2.49%  2.76%  0.86%
##    93:   1.85%  1.15%  2.60%  2.44%  2.60%  0.95%
##    94:   1.86%  1.12%  2.56%  2.44%  2.76%  0.95%
##    95:   1.85%  1.12%  2.47%  2.49%  2.71%  0.99%
##    96:   1.78%  1.03%  2.51%  2.35%  2.60%  0.95%
##    97:   1.77%  1.03%  2.47%  2.40%  2.55%  0.95%
##    98:   1.78%  1.03%  2.47%  2.44%  2.60%  0.90%
##    99:   1.77%  1.03%  2.42%  2.40%  2.60%  0.90%
##   100:   1.75%  1.09%  2.37%  2.40%  2.45%  0.90%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   8.19%  5.51% 12.07%  9.01%  9.41%  6.53%
##     2:   8.54%  5.18% 11.91% 10.58%  9.41%  7.44%
##     3:   7.72%  4.76% 10.56% 10.72%  8.48%  5.87%
##     4:   7.68%  4.50% 10.78% 11.30%  8.03%  5.64%
##     5:   6.87%  4.45%  9.70%  9.95%  6.99%  4.66%
##     6:   6.45%  4.32%  9.25%  8.94%  6.92%  4.08%
##     7:   5.97%  4.12%  8.98%  8.28%  5.92%  3.61%
##     8:   5.69%  3.77%  8.41%  7.18%  6.62%  3.65%
##     9:   5.03%  3.46%  7.50%  6.59%  5.40%  3.16%
##    10:   4.92%  3.48%  7.22%  6.54%  5.25%  2.96%
##    11:   4.51%  3.13%  6.91%  5.87%  5.03%  2.45%
##    12:   4.16%  2.82%  6.25%  5.89%  4.29%  2.35%
##    13:   3.94%  2.84%  5.55%  5.41%  4.39%  2.21%
##    14:   3.66%  2.51%  5.26%  4.65%  4.54%  2.08%
##    15:   3.44%  2.45%  5.44%  4.13%  3.91%  1.90%
##    16:   3.23%  2.32%  4.89%  3.99%  3.80%  1.71%
##    17:   3.13%  2.26%  4.89%  3.99%  3.28%  1.71%
##    18:   3.08%  2.20%  4.79%  3.95%  3.43%  1.58%
##    19:   2.94%  2.26%  4.57%  3.48%  3.23%  1.58%
##    20:   2.93%  2.35%  4.34%  3.48%  3.17%  1.67%
##    21:   2.83%  2.02%  4.16%  3.52%  3.43%  1.53%
##    22:   2.76%  2.08%  4.29%  3.38%  3.02%  1.44%
##    23:   2.74%  2.14%  4.20%  3.10%  3.12%  1.53%
##    24:   2.65%  1.99%  4.02%  3.24%  3.07%  1.35%
##    25:   2.66%  1.90%  4.16%  3.10%  3.23%  1.40%
##    26:   2.51%  1.69%  3.84%  3.10%  2.91%  1.53%
##    27:   2.55%  1.81%  3.97%  3.15%  2.97%  1.31%
##    28:   2.40%  1.66%  3.97%  2.63%  3.02%  1.22%
##    29:   2.51%  1.54%  4.11%  2.96%  3.17%  1.35%
##    30:   2.45%  1.69%  3.79%  2.72%  3.23%  1.35%
##    31:   2.44%  1.66%  3.70%  2.91%  3.23%  1.22%
##    32:   2.45%  1.72%  3.70%  2.77%  3.43%  1.17%
##    33:   2.40%  1.66%  3.84%  2.82%  3.12%  1.08%
##    34:   2.44%  1.66%  3.65%  3.10%  3.12%  1.17%
##    35:   2.44%  1.48%  3.84%  2.91%  3.28%  1.31%
##    36:   2.33%  1.51%  3.65%  2.72%  3.07%  1.22%
##    37:   2.30%  1.54%  3.52%  2.68%  3.07%  1.22%
##    38:   2.30%  1.42%  3.70%  2.77%  3.07%  1.13%
##    39:   2.28%  1.48%  3.52%  2.68%  3.07%  1.22%
##    40:   2.15%  1.30%  3.33%  2.63%  3.12%  0.95%
##    41:   2.22%  1.48%  3.56%  2.77%  2.86%  0.90%
##    42:   2.23%  1.54%  3.42%  2.82%  2.91%  0.95%
##    43:   2.22%  1.51%  3.33%  2.68%  2.97%  1.08%
##    44:   2.18%  1.42%  3.33%  2.72%  2.91%  1.04%
##    45:   2.16%  1.51%  3.24%  2.58%  2.91%  0.99%
##    46:   2.17%  1.48%  3.20%  2.58%  3.02%  1.04%
##    47:   2.10%  1.45%  3.11%  2.63%  2.81%  0.95%
##    48:   2.06%  1.45%  3.20%  2.44%  2.76%  0.90%
##    49:   1.99%  1.45%  2.92%  2.40%  2.65%  0.90%
##    50:   2.01%  1.36%  2.97%  2.58%  2.60%  0.99%
##    51:   1.96%  1.27%  2.92%  2.49%  2.55%  1.04%
##    52:   2.01%  1.21%  2.92%  2.63%  2.81%  1.04%
##    53:   2.00%  1.24%  2.92%  2.49%  2.86%  0.99%
##    54:   2.06%  1.30%  3.11%  2.49%  2.86%  1.08%
##    55:   1.97%  1.33%  2.60%  2.49%  2.71%  1.17%
##    56:   1.98%  1.24%  2.92%  2.40%  2.76%  1.08%
##    57:   2.01%  1.24%  2.92%  2.49%  2.81%  1.13%
##    58:   2.01%  1.27%  2.88%  2.44%  2.76%  1.22%
##    59:   1.97%  1.30%  2.79%  2.35%  2.86%  1.04%
##    60:   1.98%  1.24%  2.88%  2.49%  2.86%  0.95%
##    61:   2.00%  1.27%  2.83%  2.40%  2.76%  1.22%
##    62:   1.95%  1.18%  2.97%  2.30%  2.86%  0.99%
##    63:   1.99%  1.21%  2.92%  2.40%  2.86%  1.08%
##    64:   1.90%  1.12%  2.65%  2.35%  2.97%  0.99%
##    65:   1.92%  1.24%  2.65%  2.40%  2.97%  0.86%
##    66:   1.83%  1.06%  2.79%  2.40%  2.60%  0.86%
##    67:   1.89%  1.12%  2.65%  2.49%  2.81%  0.95%
##    68:   1.86%  1.21%  2.65%  2.40%  2.76%  0.77%
##    69:   1.84%  1.24%  2.51%  2.35%  2.76%  0.81%
##    70:   1.94%  1.33%  2.92%  2.21%  2.81%  0.90%
##    71:   1.96%  1.21%  2.88%  2.21%  2.97%  1.08%
##    72:   1.94%  1.24%  2.69%  2.30%  2.91%  1.08%
##    73:   1.95%  1.18%  2.88%  2.40%  2.91%  0.95%
##    74:   1.89%  1.12%  2.65%  2.25%  2.91%  1.04%
##    75:   1.96%  1.18%  2.79%  2.30%  3.02%  1.08%
##    76:   1.95%  1.24%  2.88%  2.21%  2.97%  0.99%
##    77:   1.91%  1.18%  2.60%  2.30%  3.02%  0.99%
##    78:   1.87%  1.09%  2.47%  2.35%  3.02%  0.99%
##    79:   1.94%  1.12%  2.79%  2.35%  3.07%  0.95%
##    80:   1.94%  1.09%  2.88%  2.35%  3.02%  0.95%
##    81:   1.92%  1.03%  2.83%  2.35%  3.07%  0.95%
##    82:   1.88%  0.99%  2.83%  2.35%  2.91%  0.90%
##    83:   1.94%  1.03%  2.83%  2.40%  3.12%  0.99%
##    84:   1.89%  0.96%  2.88%  2.35%  2.91%  0.99%
##    85:   1.94%  1.06%  2.88%  2.49%  2.91%  0.95%
##    86:   1.85%  0.96%  2.69%  2.30%  3.02%  0.90%
##    87:   1.91%  1.06%  2.74%  2.44%  2.97%  0.95%
##    88:   1.86%  1.06%  2.60%  2.40%  2.97%  0.86%
##    89:   1.89%  1.06%  2.69%  2.44%  2.97%  0.90%
##    90:   1.83%  1.03%  2.56%  2.35%  2.91%  0.90%
##    91:   1.84%  0.99%  2.56%  2.40%  2.91%  0.95%
##    92:   1.81%  0.96%  2.60%  2.35%  2.86%  0.86%
##    93:   1.89%  0.93%  2.69%  2.44%  3.02%  1.04%
##    94:   1.88%  1.06%  2.56%  2.49%  2.97%  0.90%
##    95:   1.91%  0.99%  2.74%  2.40%  2.97%  1.08%
##    96:   1.88%  1.03%  2.65%  2.44%  2.91%  0.95%
##    97:   1.95%  1.06%  2.88%  2.40%  2.97%  1.08%
##    98:   1.90%  0.99%  2.83%  2.40%  2.91%  0.99%
##    99:   1.88%  0.96%  2.74%  2.44%  2.86%  0.99%
##   100:   1.85%  0.99%  2.69%  2.40%  2.81%  0.95%
## ntree      OOB      1      2      3      4      5
##     1:   9.73%  7.06% 10.62% 10.39% 12.32%  9.99%
##     2:  10.00%  6.23% 12.70% 10.79% 11.88% 10.61%
##     3:   9.86%  5.54% 13.45% 11.03% 11.43% 10.27%
##     4:   9.12%  5.39% 13.13% 10.95%  9.86%  8.32%
##     5:   8.77%  5.26% 12.68% 10.24%  9.96%  7.65%
##     6:   8.26%  5.05% 11.81%  9.38%  9.85%  7.06%
##     7:   7.70%  4.69% 12.06%  8.39%  8.81%  6.17%
##     8:   7.03%  4.55% 10.34%  8.00%  7.50%  6.08%
##     9:   6.07%  3.95%  9.56%  6.29%  6.77%  4.86%
##    10:   5.64%  3.87%  8.24%  6.21%  5.82%  4.96%
##    11:   5.17%  3.46%  7.37%  6.47%  5.41%  4.07%
##    12:   4.63%  3.16%  7.09%  5.32%  4.82%  3.50%
##    13:   4.13%  2.80%  6.54%  4.71%  4.50%  2.81%
##    14:   3.89%  2.38%  6.27%  4.41%  4.44%  2.76%
##    15:   3.55%  2.55%  5.57%  4.21%  3.92%  2.02%
##    16:   3.30%  1.90%  5.39%  4.06%  3.81%  2.11%
##    17:   3.16%  2.11%  4.99%  3.67%  3.50%  2.11%
##    18:   3.08%  1.93%  4.86%  3.71%  3.60%  1.93%
##    19:   2.91%  1.90%  4.42%  3.71%  3.23%  1.88%
##    20:   2.98%  1.87%  4.51%  3.81%  3.39%  1.97%
##    21:   2.73%  1.75%  3.94%  3.17%  3.76%  1.65%
##    22:   2.61%  1.66%  4.02%  3.02%  3.23%  1.65%
##    23:   2.62%  1.63%  3.94%  3.17%  3.34%  1.65%
##    24:   2.61%  1.75%  3.98%  3.17%  3.08%  1.56%
##    25:   2.38%  1.63%  3.59%  2.82%  2.92%  1.38%
##    26:   2.34%  1.45%  3.54%  2.77%  2.92%  1.51%
##    27:   2.32%  1.42%  3.59%  2.62%  2.92%  1.56%
##    28:   2.26%  1.45%  3.46%  2.48%  2.92%  1.47%
##    29:   2.21%  1.30%  3.19%  2.72%  3.13%  1.28%
##    30:   2.22%  1.45%  3.24%  2.52%  3.08%  1.28%
##    31:   2.15%  1.28%  3.32%  2.28%  2.87%  1.51%
##    32:   2.00%  1.25%  2.93%  2.03%  2.71%  1.56%
##    33:   2.02%  1.16%  3.06%  2.23%  2.61%  1.56%
##    34:   1.94%  1.16%  3.02%  2.13%  2.45%  1.42%
##    35:   1.96%  1.22%  3.06%  2.03%  2.56%  1.38%
##    36:   2.00%  1.33%  2.93%  2.13%  2.66%  1.38%
##    37:   1.92%  1.22%  3.11%  1.88%  2.56%  1.24%
##    38:   1.94%  1.28%  2.97%  1.93%  2.61%  1.33%
##    39:   1.90%  1.28%  2.84%  1.98%  2.50%  1.28%
##    40:   1.89%  1.19%  2.93%  2.08%  2.56%  1.10%
##    41:   1.81%  1.19%  2.76%  1.93%  2.40%  1.15%
##    42:   1.86%  1.28%  2.71%  2.08%  2.40%  1.19%
##    43:   1.81%  1.19%  2.80%  1.93%  2.35%  1.15%
##    44:   1.83%  1.28%  2.93%  1.83%  2.30%  1.10%
##    45:   1.80%  1.25%  2.76%  1.98%  2.24%  1.10%
##    46:   1.78%  1.19%  2.67%  2.08%  2.30%  1.05%
##    47:   1.77%  1.22%  2.54%  1.83%  2.35%  1.24%
##    48:   1.69%  1.22%  2.36%  1.73%  2.24%  1.19%
##    49:   1.69%  1.22%  2.36%  1.73%  2.35%  1.10%
##    50:   1.71%  1.19%  2.54%  1.63%  2.35%  1.15%
##    51:   1.69%  1.10%  2.32%  1.78%  2.40%  1.24%
##    52:   1.66%  1.07%  2.41%  1.73%  2.30%  1.15%
##    53:   1.60%  1.13%  2.23%  1.68%  2.19%  1.10%
##    54:   1.65%  1.13%  2.27%  1.78%  2.30%  1.10%
##    55:   1.67%  1.16%  2.36%  1.73%  2.24%  1.19%
##    56:   1.73%  1.16%  2.27%  2.03%  2.35%  1.24%
##    57:   1.68%  1.22%  2.27%  1.88%  2.35%  1.01%
##    58:   1.70%  1.16%  2.27%  1.83%  2.35%  1.24%
##    59:   1.65%  1.13%  2.23%  1.93%  2.19%  1.10%
##    60:   1.67%  1.13%  2.19%  1.93%  2.45%  1.05%
##    61:   1.71%  1.13%  2.41%  1.88%  2.35%  1.15%
##    62:   1.67%  1.16%  2.36%  1.93%  2.24%  1.01%
##    63:   1.69%  1.19%  2.32%  1.93%  2.30%  1.05%
##    64:   1.64%  1.13%  2.23%  1.78%  2.19%  1.19%
##    65:   1.62%  1.10%  2.32%  1.73%  2.30%  1.01%
##    66:   1.63%  1.13%  2.23%  1.93%  2.19%  1.01%
##    67:   1.53%  1.07%  2.10%  1.63%  2.14%  1.01%
##    68:   1.55%  1.13%  2.10%  1.73%  2.19%  0.92%
##    69:   1.52%  1.10%  2.06%  1.68%  2.14%  0.92%
##    70:   1.56%  1.16%  2.32%  1.63%  2.14%  0.83%
##    71:   1.52%  1.07%  2.23%  1.68%  2.09%  0.83%
##    72:   1.49%  1.07%  2.10%  1.58%  2.19%  0.83%
##    73:   1.50%  1.04%  2.14%  1.58%  2.19%  0.87%
##    74:   1.43%  1.07%  1.88%  1.58%  2.03%  0.83%
##    75:   1.48%  1.07%  2.01%  1.58%  2.14%  0.87%
##    76:   1.44%  1.04%  2.01%  1.53%  2.09%  0.78%
##    77:   1.49%  1.10%  2.06%  1.53%  2.24%  0.78%
##    78:   1.45%  1.04%  2.06%  1.53%  2.19%  0.73%
##    79:   1.45%  0.92%  2.14%  1.58%  2.24%  0.73%
##    80:   1.40%  0.92%  1.97%  1.58%  2.19%  0.69%
##    81:   1.44%  0.83%  2.10%  1.63%  2.24%  0.78%
##    82:   1.44%  0.95%  2.10%  1.63%  2.19%  0.69%
##    83:   1.41%  0.86%  2.10%  1.53%  2.24%  0.69%
##    84:   1.38%  0.80%  1.97%  1.53%  2.24%  0.73%
##    85:   1.44%  0.92%  2.06%  1.68%  2.24%  0.69%
##    86:   1.40%  0.86%  2.01%  1.58%  2.24%  0.69%
##    87:   1.39%  1.01%  1.97%  1.49%  2.19%  0.60%
##    88:   1.40%  0.92%  2.06%  1.49%  2.19%  0.69%
##    89:   1.39%  0.95%  2.01%  1.44%  2.19%  0.69%
##    90:   1.40%  0.89%  2.10%  1.44%  2.19%  0.73%
##    91:   1.39%  0.86%  2.01%  1.44%  2.19%  0.83%
##    92:   1.41%  0.92%  1.92%  1.53%  2.24%  0.78%
##    93:   1.40%  0.86%  2.06%  1.49%  2.19%  0.78%
##    94:   1.38%  0.86%  1.92%  1.53%  2.19%  0.73%
##    95:   1.38%  0.80%  2.01%  1.53%  2.24%  0.69%
##    96:   1.39%  0.80%  1.97%  1.63%  2.19%  0.78%
##    97:   1.41%  0.83%  2.01%  1.63%  2.19%  0.78%
##    98:   1.38%  0.83%  1.97%  1.53%  2.14%  0.78%
##    99:   1.38%  0.80%  2.06%  1.53%  2.09%  0.78%
##   100:   1.39%  0.83%  2.06%  1.53%  2.09%  0.83%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.53%  5.25%  9.26%  9.84%  7.37%  7.06%
##     2:   7.56%  5.45%  9.24%  9.72%  7.49%  7.08%
##     3:   7.30%  5.79%  9.75%  8.55%  7.02%  6.17%
##     4:   7.09%  5.15%  9.28%  8.17%  7.64%  6.31%
##     5:   6.30%  5.05%  8.19%  7.21%  6.71%  5.06%
##     6:   5.81%  4.74%  7.69%  6.22%  6.27%  4.72%
##     7:   5.56%  4.38%  7.03%  6.80%  6.08%  4.21%
##     8:   5.33%  4.06%  6.66%  6.11%  5.79%  4.76%
##     9:   5.00%  3.67%  6.53%  5.92%  5.80%  3.91%
##    10:   4.74%  3.17%  5.97%  5.95%  5.97%  3.70%
##    11:   4.36%  2.95%  5.50%  5.38%  5.25%  3.64%
##    12:   4.01%  2.62%  5.27%  4.47%  4.72%  3.77%
##    13:   3.84%  2.68%  5.04%  4.31%  4.81%  3.08%
##    14:   3.68%  2.44%  4.73%  4.26%  4.86%  2.94%
##    15:   3.74%  2.32%  4.77%  4.56%  4.75%  3.21%
##    16:   3.42%  2.29%  4.24%  4.01%  4.75%  2.57%
##    17:   3.22%  2.37%  3.54%  4.26%  3.96%  2.57%
##    18:   3.10%  1.99%  3.54%  4.31%  4.02%  2.43%
##    19:   3.05%  1.99%  3.63%  3.86%  4.33%  2.20%
##    20:   2.84%  2.08%  3.24%  3.56%  4.07%  1.88%
##    21:   2.70%  1.93%  2.67%  3.51%  4.17%  1.88%
##    22:   2.72%  1.75%  2.97%  3.47%  4.02%  2.11%
##    23:   2.59%  1.69%  2.93%  3.32%  3.81%  1.88%
##    24:   2.50%  1.72%  2.62%  3.42%  3.50%  1.83%
##    25:   2.37%  1.60%  2.62%  3.22%  3.34%  1.65%
##    26:   2.39%  1.60%  2.45%  3.37%  3.55%  1.60%
##    27:   2.45%  1.60%  2.67%  3.12%  3.70%  1.79%
##    28:   2.39%  1.66%  2.54%  3.27%  3.50%  1.56%
##    29:   2.39%  1.75%  2.71%  2.92%  3.50%  1.56%
##    30:   2.36%  1.57%  2.67%  2.97%  3.65%  1.56%
##    31:   2.25%  1.54%  2.54%  2.87%  3.39%  1.47%
##    32:   2.24%  1.54%  2.54%  2.77%  3.34%  1.56%
##    33:   2.23%  1.60%  2.67%  2.67%  3.29%  1.42%
##    34:   2.27%  1.51%  2.49%  2.97%  3.44%  1.51%
##    35:   2.22%  1.36%  2.45%  2.92%  3.39%  1.60%
##    36:   2.26%  1.51%  2.54%  2.92%  3.29%  1.60%
##    37:   2.23%  1.33%  2.62%  2.82%  3.29%  1.74%
##    38:   2.17%  1.39%  2.49%  2.67%  3.29%  1.60%
##    39:   2.17%  1.25%  2.54%  2.87%  3.34%  1.56%
##    40:   2.06%  1.22%  2.41%  2.82%  3.18%  1.33%
##    41:   2.05%  1.22%  2.32%  2.92%  3.03%  1.38%
##    42:   2.10%  1.16%  2.54%  3.02%  2.92%  1.51%
##    43:   2.04%  1.19%  2.36%  2.82%  3.08%  1.38%
##    44:   1.96%  1.22%  2.14%  2.67%  3.08%  1.28%
##    45:   2.00%  1.16%  2.36%  2.82%  2.97%  1.28%
##    46:   2.02%  1.10%  2.36%  3.02%  2.97%  1.33%
##    47:   2.00%  1.19%  2.32%  2.87%  2.97%  1.28%
##    48:   2.07%  1.10%  2.62%  3.02%  3.08%  1.24%
##    49:   1.97%  1.16%  2.19%  2.92%  2.97%  1.24%
##    50:   1.98%  1.13%  2.14%  2.97%  3.03%  1.28%
##    51:   1.97%  1.13%  2.27%  2.92%  2.92%  1.24%
##    52:   1.94%  1.16%  2.19%  2.77%  2.97%  1.24%
##    53:   1.89%  1.19%  2.01%  2.67%  2.92%  1.24%
##    54:   1.95%  1.16%  2.23%  2.72%  3.03%  1.24%
##    55:   1.91%  1.19%  2.14%  2.67%  2.97%  1.15%
##    56:   1.91%  1.16%  2.01%  2.62%  3.08%  1.28%
##    57:   1.94%  1.16%  2.10%  2.67%  3.18%  1.24%
##    58:   1.94%  1.07%  2.14%  2.87%  3.08%  1.19%
##    59:   1.94%  1.10%  2.14%  2.62%  3.23%  1.24%
##    60:   1.91%  1.10%  2.14%  2.67%  3.08%  1.19%
##    61:   1.89%  1.10%  2.10%  2.52%  3.08%  1.24%
##    62:   1.88%  1.07%  2.10%  2.52%  3.03%  1.28%
##    63:   1.87%  1.10%  2.10%  2.43%  2.97%  1.33%
##    64:   1.89%  1.10%  2.06%  2.52%  3.08%  1.28%
##    65:   1.86%  1.10%  2.14%  2.33%  3.08%  1.24%
##    66:   1.85%  1.04%  2.06%  2.43%  3.18%  1.19%
##    67:   1.86%  1.10%  2.06%  2.33%  3.13%  1.28%
##    68:   1.88%  1.10%  2.06%  2.38%  3.18%  1.28%
##    69:   1.87%  1.13%  1.97%  2.52%  2.97%  1.33%
##    70:   1.89%  1.22%  1.92%  2.48%  3.13%  1.28%
##    71:   1.86%  1.19%  1.88%  2.48%  3.03%  1.28%
##    72:   1.89%  1.19%  2.01%  2.52%  2.97%  1.33%
##    73:   1.86%  1.22%  1.92%  2.43%  2.92%  1.33%
##    74:   1.87%  1.13%  1.97%  2.43%  3.08%  1.33%
##    75:   1.88%  1.19%  1.92%  2.52%  2.97%  1.33%
##    76:   1.89%  1.16%  1.97%  2.43%  3.08%  1.38%
##    77:   1.86%  1.16%  1.88%  2.38%  3.13%  1.33%
##    78:   1.89%  1.16%  1.97%  2.33%  3.18%  1.38%
##    79:   1.87%  1.16%  1.92%  2.38%  3.13%  1.33%
##    80:   1.92%  1.19%  1.97%  2.43%  3.23%  1.38%
##    81:   1.91%  1.16%  2.06%  2.43%  3.13%  1.38%
##    82:   1.94%  1.16%  2.19%  2.33%  3.23%  1.38%
##    83:   1.94%  1.16%  2.06%  2.48%  3.18%  1.42%
##    84:   1.94%  1.19%  2.01%  2.48%  3.34%  1.33%
##    85:   1.91%  1.10%  2.10%  2.48%  3.23%  1.28%
##    86:   1.91%  1.13%  2.14%  2.43%  3.18%  1.28%
##    87:   1.94%  1.16%  2.14%  2.48%  3.13%  1.38%
##    88:   1.87%  1.13%  2.06%  2.43%  3.03%  1.28%
##    89:   1.89%  1.13%  2.14%  2.38%  3.08%  1.33%
##    90:   1.88%  1.07%  2.14%  2.43%  3.03%  1.33%
##    91:   1.87%  1.10%  2.06%  2.43%  3.03%  1.33%
##    92:   1.88%  1.10%  2.06%  2.48%  3.03%  1.33%
##    93:   1.85%  1.07%  2.10%  2.43%  2.92%  1.33%
##    94:   1.88%  1.10%  2.10%  2.43%  3.03%  1.33%
##    95:   1.87%  1.16%  2.01%  2.43%  2.97%  1.33%
##    96:   1.85%  1.13%  2.06%  2.38%  2.92%  1.33%
##    97:   1.85%  1.07%  2.10%  2.48%  2.92%  1.28%
##    98:   1.85%  1.07%  2.06%  2.43%  3.03%  1.28%
##    99:   1.83%  1.10%  2.06%  2.38%  2.87%  1.28%
##   100:   1.81%  0.98%  2.01%  2.38%  2.97%  1.33%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.58%  4.94% 11.39%  8.73%  6.73%  7.53%
##     2:   7.79%  5.88% 11.25%  8.28%  7.80%  6.71%
##     3:   7.22%  5.47%  9.49%  8.24%  7.77%  6.16%
##     4:   6.76%  5.18%  8.42%  8.53%  7.50%  5.19%
##     5:   6.70%  4.72%  8.83%  7.99%  7.74%  5.42%
##     6:   6.07%  3.65%  8.76%  7.15%  7.26%  4.95%
##     7:   5.50%  3.59%  7.92%  6.76%  6.12%  4.18%
##     8:   5.26%  3.08%  7.22%  7.03%  6.20%  4.11%
##     9:   4.84%  3.23%  6.07%  6.63%  5.45%  3.84%
##    10:   4.54%  2.88%  5.70%  6.51%  5.15%  3.53%
##    11:   4.25%  2.72%  4.84%  5.99%  5.29%  3.47%
##    12:   3.96%  2.62%  4.65%  5.52%  4.55%  3.37%
##    13:   3.79%  2.71%  4.60%  5.41%  4.18%  2.76%
##    14:   3.53%  2.41%  4.07%  5.06%  3.97%  2.90%
##    15:   3.32%  2.40%  3.77%  4.52%  4.28%  2.34%
##    16:   3.30%  2.11%  4.12%  4.37%  3.97%  2.71%
##    17:   3.27%  2.20%  4.25%  4.36%  3.81%  2.43%
##    18:   2.91%  1.90%  3.81%  3.87%  3.44%  2.16%
##    19:   3.02%  2.11%  3.89%  3.97%  3.55%  2.20%
##    20:   2.88%  1.93%  3.63%  3.76%  3.70%  2.02%
##    21:   2.84%  1.72%  3.63%  3.86%  3.55%  2.15%
##    22:   2.83%  1.90%  3.41%  3.51%  3.96%  2.02%
##    23:   2.66%  1.87%  3.02%  3.51%  3.50%  1.97%
##    24:   2.69%  1.90%  3.28%  3.27%  3.60%  1.97%
##    25:   2.68%  1.81%  3.11%  3.66%  3.55%  1.93%
##    26:   2.62%  1.66%  3.15%  3.27%  3.34%  2.34%
##    27:   2.50%  1.63%  3.02%  3.07%  3.34%  2.02%
##    28:   2.55%  1.63%  3.19%  3.02%  3.65%  1.88%
##    29:   2.53%  1.54%  2.97%  3.27%  3.65%  1.93%
##    30:   2.39%  1.51%  2.93%  3.17%  3.23%  1.74%
##    31:   2.33%  1.48%  2.80%  2.92%  3.23%  1.79%
##    32:   2.30%  1.39%  2.84%  2.87%  3.23%  1.79%
##    33:   2.26%  1.36%  2.76%  2.87%  3.18%  1.74%
##    34:   2.17%  1.19%  2.67%  2.77%  3.29%  1.65%
##    35:   2.19%  1.22%  2.54%  2.92%  3.23%  1.74%
##    36:   2.23%  1.36%  2.54%  2.92%  3.18%  1.79%
##    37:   2.13%  1.36%  2.45%  2.62%  3.08%  1.70%
##    38:   2.11%  1.22%  2.36%  2.67%  3.29%  1.70%
##    39:   2.14%  1.22%  2.36%  2.82%  3.29%  1.70%
##    40:   2.10%  1.19%  2.32%  2.57%  3.29%  1.79%
##    41:   2.11%  1.28%  2.23%  2.62%  3.23%  1.83%
##    42:   2.11%  1.25%  2.19%  2.52%  3.34%  1.88%
##    43:   2.07%  1.30%  2.41%  2.48%  3.13%  1.60%
##    44:   2.11%  1.19%  2.49%  2.67%  3.08%  1.79%
##    45:   2.03%  1.19%  2.27%  2.62%  2.87%  1.79%
##    46:   2.01%  1.16%  2.45%  2.33%  2.92%  1.79%
##    47:   1.99%  1.07%  2.49%  2.28%  3.03%  1.70%
##    48:   1.99%  1.04%  2.45%  2.43%  2.87%  1.79%
##    49:   2.00%  1.13%  2.45%  2.38%  2.87%  1.74%
##    50:   2.00%  1.04%  2.58%  2.38%  2.76%  1.83%
##    51:   1.99%  1.07%  2.54%  2.57%  2.66%  1.70%
##    52:   1.99%  1.01%  2.45%  2.62%  2.71%  1.79%
##    53:   2.00%  1.04%  2.49%  2.57%  2.87%  1.70%
##    54:   1.97%  1.07%  2.45%  2.28%  2.92%  1.74%
##    55:   1.97%  1.16%  2.45%  2.48%  2.61%  1.70%
##    56:   1.97%  1.10%  2.45%  2.52%  2.66%  1.70%
##    57:   1.98%  1.13%  2.36%  2.43%  2.76%  1.79%
##    58:   1.95%  1.07%  2.32%  2.43%  2.87%  1.70%
##    59:   1.95%  1.07%  2.41%  2.38%  2.92%  1.60%
##    60:   1.84%  1.01%  2.27%  2.28%  2.82%  1.42%
##    61:   1.94%  1.10%  2.27%  2.38%  2.92%  1.60%
##    62:   1.95%  1.13%  2.32%  2.43%  2.87%  1.60%
##    63:   1.99%  1.16%  2.58%  2.38%  2.92%  1.47%
##    64:   1.95%  1.10%  2.41%  2.33%  2.87%  1.65%
##    65:   1.92%  1.13%  2.23%  2.33%  2.87%  1.60%
##    66:   1.89%  1.01%  2.23%  2.38%  2.92%  1.51%
##    67:   1.87%  1.04%  2.10%  2.38%  2.82%  1.60%
##    68:   1.87%  1.01%  2.27%  2.28%  2.82%  1.56%
##    69:   1.86%  0.98%  2.19%  2.28%  2.92%  1.56%
##    70:   1.85%  0.95%  2.06%  2.28%  3.03%  1.60%
##    71:   1.83%  0.98%  2.01%  2.38%  2.87%  1.51%
##    72:   1.85%  0.98%  2.10%  2.38%  2.87%  1.56%
##    73:   1.86%  1.04%  2.14%  2.33%  2.87%  1.51%
##    74:   1.85%  1.01%  2.19%  2.28%  2.87%  1.51%
##    75:   1.89%  1.01%  2.23%  2.38%  2.97%  1.51%
##    76:   1.91%  1.01%  2.32%  2.43%  2.92%  1.51%
##    77:   1.89%  1.04%  2.32%  2.28%  2.82%  1.56%
##    78:   1.87%  1.04%  2.14%  2.38%  2.87%  1.51%
##    79:   1.87%  1.01%  2.23%  2.28%  3.03%  1.42%
##    80:   1.90%  1.04%  2.19%  2.38%  3.03%  1.51%
##    81:   1.88%  1.01%  2.23%  2.23%  2.97%  1.56%
##    82:   1.88%  0.98%  2.14%  2.28%  3.08%  1.56%
##    83:   1.86%  1.01%  2.14%  2.28%  3.08%  1.42%
##    84:   1.88%  1.01%  2.23%  2.28%  3.08%  1.42%
##    85:   1.83%  1.01%  2.14%  2.23%  2.97%  1.42%
##    86:   1.81%  0.98%  2.06%  2.23%  2.97%  1.42%
##    87:   1.83%  1.04%  2.14%  2.23%  2.87%  1.42%
##    88:   1.84%  1.01%  2.19%  2.33%  2.92%  1.38%
##    89:   1.86%  1.01%  2.19%  2.28%  3.03%  1.42%
##    90:   1.84%  1.04%  2.19%  2.23%  2.97%  1.38%
##    91:   1.84%  0.95%  2.27%  2.23%  2.97%  1.42%
##    92:   1.82%  0.95%  2.19%  2.13%  3.08%  1.38%
##    93:   1.85%  0.95%  2.27%  2.08%  3.18%  1.42%
##    94:   1.85%  1.04%  2.19%  2.08%  3.18%  1.38%
##    95:   1.80%  0.95%  2.14%  2.03%  3.08%  1.42%
##    96:   1.82%  1.01%  2.14%  2.13%  2.97%  1.42%
##    97:   1.80%  1.01%  2.06%  2.13%  2.97%  1.42%
##    98:   1.82%  1.04%  2.10%  2.13%  2.97%  1.42%
##    99:   1.79%  1.01%  2.01%  2.13%  2.97%  1.42%
##   100:   1.77%  1.01%  2.06%  2.13%  2.87%  1.38%
## ntree      OOB      1      2      3      4      5
##     1:  10.59%  5.97% 13.52% 13.93% 13.26%  8.97%
##     2:  11.43%  7.27% 15.69% 13.71% 12.95%  9.86%
##     3:  11.10%  6.99% 15.07% 13.44% 13.08%  9.28%
##     4:  10.05%  5.96% 13.21% 13.76% 10.94%  8.76%
##     5:   9.44%  5.91% 12.90% 11.17% 10.68%  8.52%
##     6:   8.38%  4.92% 11.73% 10.74%  9.10%  7.34%
##     7:   7.96%  5.11% 11.56%  9.89%  8.03%  6.69%
##     8:   7.11%  4.64%  9.83%  9.16%  7.88%  5.40%
##     9:   6.36%  4.23%  8.65%  8.22%  7.52%  4.39%
##    10:   5.68%  3.88%  7.76%  7.18%  6.16%  4.41%
##    11:   5.08%  3.24%  7.51%  6.26%  5.63%  3.73%
##    12:   4.77%  2.90%  6.66%  6.84%  5.46%  3.05%
##    13:   4.25%  2.81%  5.86%  5.48%  4.85%  3.05%
##    14:   3.80%  2.42%  5.47%  4.83%  4.35%  2.67%
##    15:   3.77%  2.45%  5.33%  5.23%  4.09%  2.48%
##    16:   3.56%  2.21%  4.76%  4.93%  3.94%  2.71%
##    17:   3.52%  2.30%  4.84%  4.77%  3.94%  2.42%
##    18:   3.09%  1.97%  4.45%  4.42%  3.54%  1.71%
##    19:   3.03%  1.97%  4.49%  4.32%  3.09%  1.85%
##    20:   2.88%  2.00%  4.32%  4.07%  2.99%  1.47%
##    21:   2.88%  2.03%  3.93%  4.32%  3.04%  1.57%
##    22:   2.60%  1.94%  3.71%  3.47%  2.64%  1.57%
##    23:   2.53%  1.64%  3.71%  3.33%  2.84%  1.61%
##    24:   2.42%  1.73%  3.49%  3.13%  2.64%  1.47%
##    25:   2.36%  1.67%  3.62%  3.08%  2.44%  1.33%
##    26:   2.33%  1.70%  3.45%  3.03%  2.49%  1.28%
##    27:   2.32%  1.73%  3.49%  2.58%  2.79%  1.28%
##    28:   2.21%  1.64%  3.40%  2.58%  2.59%  1.09%
##    29:   2.19%  1.49%  3.40%  2.58%  2.74%  1.09%
##    30:   2.10%  1.61%  3.14%  2.48%  2.44%  1.04%
##    31:   2.16%  1.70%  3.27%  2.38%  2.49%  1.14%
##    32:   1.97%  1.37%  2.88%  2.43%  2.49%  1.00%
##    33:   1.98%  1.40%  2.79%  2.68%  2.39%  0.95%
##    34:   1.92%  1.46%  2.70%  2.53%  2.29%  0.85%
##    35:   1.82%  1.37%  2.31%  2.38%  2.39%  0.90%
##    36:   1.83%  1.40%  2.49%  2.38%  2.19%  0.95%
##    37:   1.88%  1.43%  2.62%  2.38%  2.39%  0.81%
##    38:   1.78%  1.40%  2.44%  2.18%  2.29%  0.81%
##    39:   1.76%  1.25%  2.44%  2.18%  2.29%  0.90%
##    40:   1.74%  1.25%  2.40%  2.08%  2.34%  0.90%
##    41:   1.70%  1.25%  2.27%  2.08%  2.14%  1.00%
##    42:   1.64%  1.25%  2.14%  2.08%  1.99%  0.95%
##    43:   1.69%  1.22%  2.27%  2.13%  2.24%  0.85%
##    44:   1.70%  1.25%  2.14%  2.38%  2.04%  0.95%
##    45:   1.62%  1.10%  2.27%  2.18%  1.89%  0.95%
##    46:   1.61%  1.25%  2.18%  2.18%  1.74%  0.90%
##    47:   1.64%  1.22%  2.14%  2.23%  1.99%  0.85%
##    48:   1.58%  1.19%  2.14%  2.08%  1.84%  0.85%
##    49:   1.66%  1.25%  2.18%  2.18%  1.94%  0.95%
##    50:   1.59%  1.19%  2.09%  2.03%  1.99%  0.85%
##    51:   1.60%  1.19%  2.05%  2.08%  1.94%  0.95%
##    52:   1.61%  1.16%  1.96%  2.13%  2.14%  0.95%
##    53:   1.60%  1.13%  2.01%  2.03%  2.09%  1.00%
##    54:   1.57%  1.10%  1.92%  2.08%  2.14%  0.90%
##    55:   1.52%  1.07%  2.01%  1.99%  1.84%  0.95%
##    56:   1.56%  1.13%  2.01%  1.94%  2.04%  0.95%
##    57:   1.49%  1.10%  1.79%  1.99%  1.99%  0.85%
##    58:   1.46%  1.04%  1.83%  1.89%  1.89%  0.90%
##    59:   1.44%  0.98%  1.92%  1.89%  1.84%  0.85%
##    60:   1.44%  0.92%  1.79%  2.03%  1.89%  0.85%
##    61:   1.41%  0.98%  1.79%  1.84%  1.79%  0.90%
##    62:   1.45%  1.01%  1.96%  1.79%  1.84%  0.90%
##    63:   1.38%  0.95%  1.70%  1.69%  1.89%  0.90%
##    64:   1.42%  1.04%  1.88%  1.79%  1.74%  0.85%
##    65:   1.44%  1.04%  1.88%  1.74%  1.89%  0.90%
##    66:   1.38%  0.95%  1.83%  1.74%  1.74%  0.85%
##    67:   1.42%  0.98%  1.88%  1.79%  1.89%  0.81%
##    68:   1.43%  0.98%  1.92%  1.79%  1.84%  0.85%
##    69:   1.43%  0.98%  2.01%  1.79%  1.74%  0.85%
##    70:   1.34%  0.92%  1.92%  1.59%  1.69%  0.81%
##    71:   1.38%  0.95%  1.92%  1.69%  1.74%  0.85%
##    72:   1.37%  0.92%  1.96%  1.64%  1.74%  0.81%
##    73:   1.38%  0.98%  2.09%  1.59%  1.69%  0.76%
##    74:   1.39%  0.92%  2.09%  1.64%  1.79%  0.76%
##    75:   1.40%  0.98%  2.09%  1.54%  1.89%  0.71%
##    76:   1.40%  0.95%  2.14%  1.54%  1.84%  0.76%
##    77:   1.42%  0.89%  2.14%  1.74%  1.79%  0.81%
##    78:   1.39%  1.01%  2.05%  1.54%  1.84%  0.71%
##    79:   1.39%  0.95%  2.05%  1.59%  1.84%  0.76%
##    80:   1.38%  0.92%  2.05%  1.59%  1.79%  0.76%
##    81:   1.37%  0.95%  1.96%  1.59%  1.79%  0.76%
##    82:   1.38%  0.95%  2.09%  1.54%  1.79%  0.76%
##    83:   1.34%  0.89%  2.01%  1.49%  1.79%  0.76%
##    84:   1.35%  0.89%  2.09%  1.44%  1.79%  0.76%
##    85:   1.37%  0.92%  2.05%  1.54%  1.74%  0.81%
##    86:   1.33%  0.89%  1.96%  1.44%  1.79%  0.81%
##    87:   1.35%  0.92%  2.05%  1.44%  1.79%  0.76%
##    88:   1.38%  0.95%  2.18%  1.44%  1.74%  0.76%
##    89:   1.37%  0.98%  2.22%  1.39%  1.69%  0.71%
##    90:   1.34%  0.89%  2.09%  1.44%  1.74%  0.76%
##    91:   1.35%  0.92%  2.09%  1.44%  1.74%  0.76%
##    92:   1.34%  0.86%  2.14%  1.44%  1.74%  0.76%
##    93:   1.34%  0.92%  2.05%  1.49%  1.69%  0.76%
##    94:   1.35%  0.92%  2.05%  1.54%  1.69%  0.76%
##    95:   1.30%  0.95%  1.96%  1.39%  1.65%  0.71%
##    96:   1.30%  0.89%  1.96%  1.44%  1.69%  0.71%
##    97:   1.32%  0.86%  2.09%  1.44%  1.74%  0.71%
##    98:   1.31%  0.89%  1.96%  1.44%  1.74%  0.71%
##    99:   1.30%  0.86%  1.92%  1.44%  1.74%  0.76%
##   100:   1.28%  0.83%  1.88%  1.49%  1.69%  0.76%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.71%  5.07%  9.14% 11.31%  9.64%  5.06%
##     2:   7.64%  5.03%  9.73% 10.86%  8.19%  5.88%
##     3:   7.45%  4.92% 10.61% 10.27%  7.66%  5.09%
##     4:   7.33%  4.98%  9.74% 10.53%  7.55%  5.21%
##     5:   6.75%  4.50%  9.23%  9.02%  7.03%  5.24%
##     6:   6.42%  4.22%  9.35%  8.49%  6.64%  4.58%
##     7:   5.81%  4.38%  8.16%  7.44%  5.62%  4.19%
##     8:   5.32%  4.05%  7.41%  6.71%  5.45%  3.60%
##     9:   5.02%  3.68%  6.64%  6.51%  5.51%  3.50%
##    10:   4.56%  3.37%  6.27%  5.57%  4.97%  3.25%
##    11:   4.29%  3.17%  5.79%  5.44%  4.67%  2.96%
##    12:   4.15%  3.02%  5.16%  5.28%  5.01%  2.95%
##    13:   3.70%  2.63%  4.85%  5.02%  4.25%  2.38%
##    14:   3.57%  2.54%  4.94%  4.37%  4.45%  2.14%
##    15:   3.43%  2.24%  4.63%  4.37%  4.44%  2.19%
##    16:   3.25%  2.30%  3.93%  4.32%  4.24%  2.04%
##    17:   3.02%  2.21%  3.97%  4.02%  3.59%  1.76%
##    18:   2.97%  2.03%  3.71%  3.97%  3.89%  1.85%
##    19:   2.94%  2.18%  3.71%  3.77%  3.74%  1.76%
##    20:   2.92%  1.97%  3.71%  3.97%  3.69%  1.85%
##    21:   2.73%  1.85%  3.45%  3.92%  3.44%  1.57%
##    22:   2.68%  1.79%  3.75%  3.37%  3.39%  1.57%
##    23:   2.57%  1.70%  3.36%  3.33%  3.44%  1.57%
##    24:   2.56%  1.70%  3.62%  3.37%  3.24%  1.38%
##    25:   2.48%  1.76%  3.18%  3.33%  3.29%  1.28%
##    26:   2.42%  1.76%  3.14%  3.23%  3.34%  1.04%
##    27:   2.34%  1.67%  3.01%  3.08%  3.29%  1.09%
##    28:   2.35%  1.70%  2.92%  3.18%  3.19%  1.19%
##    29:   2.36%  1.58%  3.05%  3.23%  3.19%  1.23%
##    30:   2.34%  1.61%  3.18%  3.13%  2.99%  1.23%
##    31:   2.25%  1.49%  3.01%  2.93%  3.04%  1.23%
##    32:   2.24%  1.52%  2.97%  3.13%  2.84%  1.19%
##    33:   2.17%  1.52%  3.01%  2.93%  2.84%  0.90%
##    34:   2.34%  1.64%  3.18%  3.08%  3.09%  1.09%
##    35:   2.30%  1.61%  3.10%  3.03%  3.04%  1.14%
##    36:   2.17%  1.52%  2.83%  2.88%  3.04%  1.00%
##    37:   2.15%  1.55%  2.88%  2.83%  2.89%  0.95%
##    38:   2.15%  1.55%  2.75%  2.78%  3.09%  0.95%
##    39:   2.19%  1.49%  2.92%  2.78%  3.09%  1.09%
##    40:   2.13%  1.43%  2.88%  2.88%  2.89%  1.00%
##    41:   2.09%  1.43%  2.70%  2.68%  2.74%  1.28%
##    42:   2.11%  1.34%  2.88%  2.68%  2.89%  1.23%
##    43:   2.13%  1.43%  2.83%  2.68%  2.89%  1.23%
##    44:   2.11%  1.34%  2.97%  2.88%  2.64%  1.19%
##    45:   1.98%  1.25%  2.66%  2.73%  2.59%  1.09%
##    46:   2.07%  1.31%  2.97%  2.73%  2.69%  1.09%
##    47:   2.07%  1.40%  2.83%  2.63%  2.74%  1.14%
##    48:   2.08%  1.37%  2.83%  2.58%  2.84%  1.19%
##    49:   2.08%  1.28%  2.88%  2.68%  2.69%  1.33%
##    50:   2.03%  1.25%  2.83%  2.78%  2.59%  1.14%
##    51:   2.04%  1.31%  2.70%  2.73%  2.64%  1.23%
##    52:   2.00%  1.31%  2.83%  2.63%  2.54%  1.09%
##    53:   2.01%  1.31%  2.79%  2.63%  2.64%  1.09%
##    54:   1.97%  1.34%  2.66%  2.43%  2.64%  1.14%
##    55:   1.99%  1.37%  2.66%  2.58%  2.54%  1.14%
##    56:   1.97%  1.28%  2.62%  2.48%  2.59%  1.28%
##    57:   1.94%  1.28%  2.53%  2.48%  2.49%  1.28%
##    58:   1.95%  1.25%  2.49%  2.48%  2.64%  1.33%
##    59:   1.96%  1.28%  2.49%  2.43%  2.69%  1.33%
##    60:   1.99%  1.28%  2.57%  2.53%  2.69%  1.28%
##    61:   2.02%  1.28%  2.70%  2.63%  2.64%  1.28%
##    62:   1.98%  1.31%  2.62%  2.58%  2.64%  1.14%
##    63:   1.94%  1.31%  2.44%  2.48%  2.59%  1.28%
##    64:   1.86%  1.25%  2.40%  2.38%  2.49%  1.14%
##    65:   1.94%  1.19%  2.62%  2.63%  2.44%  1.23%
##    66:   1.92%  1.31%  2.49%  2.43%  2.54%  1.19%
##    67:   1.88%  1.22%  2.44%  2.43%  2.49%  1.19%
##    68:   1.87%  1.22%  2.35%  2.43%  2.49%  1.23%
##    69:   1.92%  1.22%  2.49%  2.48%  2.59%  1.23%
##    70:   1.91%  1.19%  2.44%  2.48%  2.64%  1.23%
##    71:   1.89%  1.13%  2.57%  2.38%  2.59%  1.23%
##    72:   1.95%  1.28%  2.66%  2.38%  2.59%  1.23%
##    73:   1.90%  1.16%  2.57%  2.43%  2.54%  1.23%
##    74:   1.92%  1.19%  2.53%  2.43%  2.64%  1.23%
##    75:   1.89%  1.16%  2.53%  2.43%  2.59%  1.19%
##    76:   1.90%  1.19%  2.57%  2.38%  2.59%  1.19%
##    77:   1.93%  1.16%  2.62%  2.43%  2.69%  1.19%
##    78:   1.89%  1.22%  2.49%  2.43%  2.54%  1.19%
##    79:   1.91%  1.10%  2.53%  2.53%  2.69%  1.19%
##    80:   1.87%  1.10%  2.53%  2.48%  2.49%  1.19%
##    81:   1.87%  1.10%  2.49%  2.43%  2.64%  1.14%
##    82:   1.93%  1.13%  2.57%  2.48%  2.74%  1.19%
##    83:   1.85%  1.07%  2.49%  2.43%  2.54%  1.19%
##    84:   1.83%  1.04%  2.49%  2.38%  2.59%  1.14%
##    85:   1.80%  1.13%  2.35%  2.28%  2.49%  1.14%
##    86:   1.83%  1.10%  2.40%  2.38%  2.59%  1.14%
##    87:   1.78%  0.98%  2.40%  2.23%  2.64%  1.14%
##    88:   1.79%  1.07%  2.35%  2.18%  2.64%  1.14%
##    89:   1.80%  1.10%  2.40%  2.23%  2.59%  1.09%
##    90:   1.79%  1.04%  2.44%  2.18%  2.64%  1.09%
##    91:   1.84%  1.10%  2.53%  2.28%  2.64%  1.09%
##    92:   1.80%  1.04%  2.53%  2.23%  2.54%  1.09%
##    93:   1.82%  1.01%  2.49%  2.33%  2.54%  1.19%
##    94:   1.82%  1.10%  2.40%  2.23%  2.64%  1.14%
##    95:   1.78%  1.01%  2.40%  2.28%  2.64%  1.04%
##    96:   1.83%  1.04%  2.40%  2.43%  2.64%  1.09%
##    97:   1.80%  1.07%  2.35%  2.28%  2.64%  1.09%
##    98:   1.83%  1.10%  2.35%  2.33%  2.74%  1.04%
##    99:   1.78%  1.01%  2.40%  2.28%  2.59%  1.09%
##   100:   1.78%  1.04%  2.44%  2.23%  2.59%  1.04%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   8.41%  6.45% 11.79% 10.62%  7.63%  6.55%
##     2:   7.78%  6.00% 10.01%  9.37%  8.29%  6.20%
##     3:   7.43%  5.49%  9.13%  9.88%  8.23%  5.51%
##     4:   7.26%  5.01%  9.05% 10.50%  8.18%  4.88%
##     5:   6.56%  4.51%  8.69%  9.27%  7.12%  4.41%
##     6:   6.03%  4.16%  8.24%  8.26%  6.77%  3.75%
##     7:   5.80%  4.31%  7.76%  8.26%  6.24%  3.31%
##     8:   5.11%  3.82%  7.01%  6.90%  5.60%  2.95%
##     9:   4.57%  3.56%  6.19%  6.17%  4.80%  2.70%
##    10:   4.41%  3.54%  6.08%  5.43%  4.88%  2.54%
##    11:   4.05%  3.20%  5.22%  5.71%  4.32%  2.29%
##    12:   3.72%  2.84%  4.91%  5.08%  4.25%  2.00%
##    13:   3.62%  2.63%  4.63%  4.98%  4.34%  2.09%
##    14:   3.47%  2.51%  4.59%  4.47%  4.34%  2.00%
##    15:   3.14%  2.12%  4.67%  4.17%  3.49%  1.76%
##    16:   2.98%  2.26%  4.10%  3.72%  3.59%  1.62%
##    17:   2.93%  1.97%  4.23%  3.57%  3.79%  1.62%
##    18:   2.91%  1.97%  4.10%  3.62%  3.69%  1.71%
##    19:   2.67%  2.15%  3.66%  3.18%  3.29%  1.33%
##    20:   2.47%  1.88%  3.18%  3.18%  3.09%  1.38%
##    21:   2.36%  1.76%  3.10%  2.83%  3.09%  1.38%
##    22:   2.40%  1.64%  3.14%  3.13%  3.29%  1.28%
##    23:   2.34%  1.70%  3.14%  2.98%  2.99%  1.28%
##    24:   2.31%  1.79%  2.75%  2.88%  3.19%  1.28%
##    25:   2.21%  1.76%  2.79%  2.38%  3.29%  1.09%
##    26:   2.14%  1.49%  2.75%  2.43%  3.09%  1.33%
##    27:   2.12%  1.58%  2.88%  2.43%  2.99%  1.04%
##    28:   2.17%  1.67%  3.10%  2.28%  3.04%  1.04%
##    29:   2.16%  1.82%  2.79%  2.38%  2.99%  1.00%
##    30:   2.19%  1.73%  3.05%  2.33%  3.04%  1.04%
##    31:   2.12%  1.52%  2.97%  2.23%  3.19%  1.04%
##    32:   2.07%  1.52%  2.92%  1.94%  3.24%  1.04%
##    33:   2.01%  1.55%  2.83%  1.99%  3.04%  0.90%
##    34:   1.95%  1.49%  2.57%  2.03%  2.99%  0.95%
##    35:   1.96%  1.49%  2.75%  1.94%  2.94%  0.95%
##    36:   1.95%  1.49%  2.62%  2.03%  2.94%  0.95%
##    37:   1.99%  1.52%  2.66%  2.08%  2.99%  0.95%
##    38:   1.93%  1.46%  2.83%  1.89%  2.79%  0.90%
##    39:   1.90%  1.37%  2.75%  1.99%  2.84%  0.85%
##    40:   1.88%  1.40%  2.75%  1.89%  2.69%  0.90%
##    41:   1.89%  1.43%  2.66%  2.03%  2.69%  0.85%
##    42:   1.93%  1.43%  2.75%  1.99%  2.84%  0.90%
##    43:   1.88%  1.37%  2.57%  2.03%  2.89%  0.81%
##    44:   1.83%  1.40%  2.49%  1.84%  2.84%  0.81%
##    45:   1.84%  1.46%  2.40%  1.94%  2.79%  0.85%
##    46:   1.79%  1.43%  2.35%  1.69%  2.84%  0.85%
##    47:   1.80%  1.37%  2.40%  1.74%  2.84%  0.90%
##    48:   1.77%  1.37%  2.22%  1.69%  2.89%  0.90%
##    49:   1.80%  1.34%  2.35%  1.79%  2.94%  0.85%
##    50:   1.76%  1.28%  2.40%  1.69%  2.84%  0.85%
##    51:   1.74%  1.31%  2.44%  1.54%  2.74%  0.90%
##    52:   1.77%  1.25%  2.53%  1.69%  2.74%  0.90%
##    53:   1.77%  1.37%  2.31%  1.69%  2.79%  0.90%
##    54:   1.75%  1.34%  2.35%  1.64%  2.79%  0.85%
##    55:   1.77%  1.37%  2.44%  1.69%  2.74%  0.85%
##    56:   1.77%  1.22%  2.57%  1.59%  2.84%  0.90%
##    57:   1.78%  1.31%  2.44%  1.69%  2.84%  0.90%
##    58:   1.75%  1.19%  2.53%  1.59%  2.79%  0.95%
##    59:   1.72%  1.16%  2.44%  1.64%  2.74%  0.95%
##    60:   1.72%  1.16%  2.40%  1.69%  2.79%  0.90%
##    61:   1.72%  1.13%  2.40%  1.74%  2.74%  0.90%
##    62:   1.66%  1.10%  2.31%  1.59%  2.74%  0.90%
##    63:   1.70%  1.10%  2.27%  1.74%  2.84%  0.90%
##    64:   1.69%  1.07%  2.31%  1.74%  2.79%  0.90%
##    65:   1.69%  1.07%  2.27%  1.74%  2.84%  0.90%
##    66:   1.68%  1.19%  2.22%  1.54%  2.84%  0.90%
##    67:   1.64%  1.07%  2.22%  1.64%  2.69%  0.90%
##    68:   1.68%  1.13%  2.14%  1.69%  2.89%  0.90%
##    69:   1.66%  1.10%  2.09%  1.64%  2.89%  0.90%
##    70:   1.69%  1.13%  2.22%  1.64%  2.89%  0.90%
##    71:   1.72%  1.10%  2.35%  1.69%  2.94%  0.85%
##    72:   1.66%  1.10%  2.22%  1.59%  2.89%  0.85%
##    73:   1.66%  1.10%  2.18%  1.64%  2.89%  0.85%
##    74:   1.68%  1.07%  2.18%  1.64%  3.04%  0.85%
##    75:   1.66%  1.04%  2.22%  1.74%  2.84%  0.85%
##    76:   1.69%  1.10%  2.22%  1.69%  2.94%  0.85%
##    77:   1.68%  1.16%  2.18%  1.69%  2.84%  0.85%
##    78:   1.65%  1.07%  2.18%  1.59%  2.89%  0.85%
##    79:   1.68%  1.07%  2.31%  1.69%  2.84%  0.85%
##    80:   1.70%  1.10%  2.27%  1.79%  2.84%  0.85%
##    81:   1.71%  1.07%  2.31%  1.74%  2.94%  0.85%
##    82:   1.65%  1.01%  2.22%  1.79%  2.74%  0.85%
##    83:   1.70%  1.07%  2.27%  1.84%  2.84%  0.85%
##    84:   1.67%  1.07%  2.27%  1.84%  2.69%  0.85%
##    85:   1.66%  1.07%  2.27%  1.84%  2.64%  0.85%
##    86:   1.66%  1.04%  2.27%  1.79%  2.69%  0.85%
##    87:   1.70%  1.07%  2.31%  1.79%  2.84%  0.85%
##    88:   1.68%  1.13%  2.27%  1.69%  2.79%  0.85%
##    89:   1.68%  1.10%  2.31%  1.64%  2.84%  0.85%
##    90:   1.69%  1.07%  2.27%  1.84%  2.84%  0.81%
##    91:   1.71%  1.07%  2.31%  1.74%  2.94%  0.85%
##    92:   1.69%  1.04%  2.18%  1.89%  2.89%  0.85%
##    93:   1.66%  1.07%  2.18%  1.79%  2.84%  0.81%
##    94:   1.70%  1.07%  2.31%  1.74%  2.94%  0.81%
##    95:   1.65%  1.04%  2.22%  1.69%  2.84%  0.81%
##    96:   1.67%  1.07%  2.22%  1.69%  2.94%  0.81%
##    97:   1.66%  1.07%  2.22%  1.69%  2.84%  0.81%
##    98:   1.68%  1.07%  2.31%  1.64%  2.94%  0.81%
##    99:   1.68%  1.10%  2.22%  1.74%  2.89%  0.81%
##   100:   1.62%  0.98%  2.22%  1.64%  2.84%  0.81%
## ntree      OOB      1      2      3      4      5
##     1:  10.49%  5.34% 14.06% 13.07% 13.55%  9.40%
##     2:  10.83%  5.39% 14.20% 13.93% 13.04% 10.79%
##     3:  10.81%  6.03% 12.87% 14.10% 12.65% 11.42%
##     4:  10.20%  6.24% 12.67% 13.05% 11.78%  9.64%
##     5:   9.31%  5.53% 11.68% 11.97% 11.19%  8.48%
##     6:   8.38%  4.79% 11.47% 10.14% 10.15%  7.52%
##     7:   7.89%  4.82% 10.97%  9.09%  9.73%  6.66%
##     8:   6.68%  4.11%  9.48%  7.48%  8.54%  5.35%
##     9:   6.39%  3.77%  9.67%  7.70%  7.37%  4.92%
##    10:   5.68%  3.89%  8.30%  6.54%  7.02%  3.77%
##    11:   5.39%  3.75%  8.26%  5.97%  6.28%  3.61%
##    12:   4.65%  3.18%  6.94%  5.52%  5.60%  2.92%
##    13:   4.26%  2.87%  6.58%  4.88%  5.08%  2.73%
##    14:   4.09%  2.75%  6.16%  4.92%  4.97%  2.46%
##    15:   3.68%  2.39%  5.80%  4.09%  4.76%  2.14%
##    16:   3.33%  1.91%  5.31%  3.99%  4.29%  2.00%
##    17:   3.19%  1.94%  4.73%  3.95%  4.39%  1.77%
##    18:   3.12%  1.79%  5.00%  3.65%  4.34%  1.68%
##    19:   2.97%  1.61%  4.51%  3.65%  4.24%  1.73%
##    20:   2.90%  1.55%  4.64%  3.56%  3.93%  1.68%
##    21:   2.87%  1.49%  4.82%  3.26%  4.19%  1.46%
##    22:   2.62%  1.46%  4.19%  3.36%  3.67%  1.18%
##    23:   2.53%  1.55%  3.93%  3.07%  3.46%  1.27%
##    24:   2.45%  1.37%  3.88%  3.07%  3.41%  1.18%
##    25:   2.39%  1.49%  3.79%  2.87%  3.15%  1.23%
##    26:   2.34%  1.37%  3.53%  3.26%  3.26%  0.96%
##    27:   2.23%  1.40%  3.26%  2.78%  3.51%  0.82%
##    28:   2.15%  1.52%  3.30%  2.58%  2.89%  0.86%
##    29:   2.18%  1.58%  3.21%  2.78%  2.89%  0.86%
##    30:   2.04%  1.52%  2.95%  2.24%  2.95%  0.91%
##    31:   1.99%  1.46%  2.95%  2.34%  2.84%  0.73%
##    32:   1.95%  1.25%  2.81%  2.44%  2.84%  0.91%
##    33:   1.98%  1.28%  2.86%  2.58%  2.84%  0.82%
##    34:   1.89%  1.22%  2.63%  2.44%  2.79%  0.82%
##    35:   1.88%  1.25%  2.72%  2.53%  2.48%  0.82%
##    36:   1.81%  1.22%  2.72%  2.48%  2.38%  0.64%
##    37:   1.87%  1.16%  2.90%  2.34%  2.79%  0.64%
##    38:   1.83%  1.16%  2.86%  2.34%  2.58%  0.64%
##    39:   1.76%  1.07%  2.68%  2.24%  2.58%  0.68%
##    40:   1.72%  1.07%  2.68%  2.14%  2.64%  0.55%
##    41:   1.71%  1.07%  2.45%  2.39%  2.58%  0.50%
##    42:   1.66%  0.96%  2.63%  2.19%  2.58%  0.41%
##    43:   1.59%  0.93%  2.32%  2.19%  2.48%  0.50%
##    44:   1.64%  1.02%  2.59%  2.19%  2.38%  0.45%
##    45:   1.69%  1.02%  2.45%  2.34%  2.48%  0.64%
##    46:   1.67%  1.02%  2.50%  2.44%  2.33%  0.55%
##    47:   1.60%  0.87%  2.28%  2.44%  2.38%  0.55%
##    48:   1.55%  0.87%  2.19%  2.24%  2.27%  0.68%
##    49:   1.55%  0.81%  2.28%  2.39%  2.27%  0.55%
##    50:   1.49%  0.93%  2.14%  2.05%  2.07%  0.64%
##    51:   1.57%  0.93%  2.45%  2.14%  2.12%  0.64%
##    52:   1.55%  0.87%  2.37%  2.09%  2.22%  0.68%
##    53:   1.54%  0.93%  2.28%  1.95%  2.27%  0.68%
##    54:   1.56%  0.90%  2.41%  2.14%  2.17%  0.64%
##    55:   1.53%  0.93%  2.32%  1.95%  2.27%  0.59%
##    56:   1.55%  0.96%  2.45%  2.05%  2.07%  0.64%
##    57:   1.55%  0.96%  2.50%  1.95%  2.17%  0.59%
##    58:   1.58%  0.93%  2.50%  2.05%  2.22%  0.64%
##    59:   1.52%  0.81%  2.45%  2.09%  2.12%  0.59%
##    60:   1.51%  0.90%  2.32%  2.05%  2.12%  0.59%
##    61:   1.52%  0.90%  2.32%  2.09%  2.12%  0.59%
##    62:   1.50%  0.78%  2.41%  2.09%  2.17%  0.55%
##    63:   1.48%  0.84%  2.45%  1.90%  2.17%  0.45%
##    64:   1.49%  0.78%  2.37%  2.05%  2.22%  0.55%
##    65:   1.48%  0.81%  2.50%  1.95%  1.96%  0.59%
##    66:   1.43%  0.75%  2.41%  1.95%  2.02%  0.45%
##    67:   1.43%  0.78%  2.41%  1.75%  2.02%  0.59%
##    68:   1.44%  0.84%  2.45%  1.80%  2.02%  0.45%
##    69:   1.49%  0.93%  2.37%  1.90%  2.02%  0.59%
##    70:   1.46%  0.87%  2.37%  1.75%  2.02%  0.68%
##    71:   1.44%  0.93%  2.23%  1.80%  1.96%  0.64%
##    72:   1.43%  0.84%  2.23%  1.80%  2.02%  0.64%
##    73:   1.41%  0.87%  2.10%  1.80%  2.07%  0.59%
##    74:   1.43%  0.93%  2.14%  1.75%  2.07%  0.59%
##    75:   1.44%  0.87%  2.14%  1.95%  2.07%  0.55%
##    76:   1.38%  0.87%  2.01%  1.75%  2.02%  0.59%
##    77:   1.39%  0.90%  2.01%  1.75%  2.02%  0.64%
##    78:   1.32%  0.84%  1.96%  1.75%  1.86%  0.55%
##    79:   1.36%  0.81%  2.01%  1.75%  2.07%  0.55%
##    80:   1.38%  0.84%  2.01%  1.90%  1.96%  0.59%
##    81:   1.38%  0.87%  2.10%  1.80%  1.96%  0.50%
##    82:   1.37%  0.87%  2.10%  1.75%  1.91%  0.55%
##    83:   1.38%  0.87%  2.10%  1.75%  1.96%  0.55%
##    84:   1.38%  0.81%  2.14%  1.85%  1.86%  0.59%
##    85:   1.40%  0.87%  2.14%  1.85%  1.91%  0.59%
##    86:   1.37%  0.75%  2.14%  1.80%  1.96%  0.59%
##    87:   1.38%  0.84%  2.14%  1.70%  1.96%  0.64%
##    88:   1.38%  0.72%  2.14%  1.85%  1.96%  0.64%
##    89:   1.38%  0.72%  2.14%  1.95%  1.91%  0.64%
##    90:   1.37%  0.72%  2.10%  1.90%  1.96%  0.59%
##    91:   1.35%  0.75%  2.01%  1.90%  1.96%  0.55%
##    92:   1.34%  0.72%  2.05%  1.75%  2.02%  0.59%
##    93:   1.35%  0.81%  2.05%  1.75%  1.96%  0.55%
##    94:   1.38%  0.84%  2.05%  1.85%  1.96%  0.55%
##    95:   1.32%  0.81%  1.96%  1.70%  1.96%  0.55%
##    96:   1.33%  0.81%  1.96%  1.80%  1.96%  0.50%
##    97:   1.34%  0.81%  2.01%  1.80%  1.96%  0.50%
##    98:   1.33%  0.78%  2.01%  1.75%  2.02%  0.50%
##    99:   1.35%  0.81%  2.05%  1.75%  2.02%  0.50%
##   100:   1.34%  0.75%  2.05%  1.75%  2.07%  0.50%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   8.19%  5.25% 12.26% 10.04% 10.69%  4.49%
##     2:   7.40%  4.85% 10.79%  9.36%  8.74%  4.82%
##     3:   7.14%  4.52% 10.01%  9.35%  8.04%  5.37%
##     4:   6.66%  4.46%  8.91%  8.16%  7.96%  5.12%
##     5:   6.29%  4.12%  8.69%  7.49%  7.60%  4.82%
##     6:   5.97%  3.95%  8.40%  7.04%  7.41%  4.27%
##     7:   5.49%  3.53%  7.98%  5.95%  6.75%  4.36%
##     8:   4.82%  3.10%  6.56%  5.42%  6.17%  3.87%
##     9:   4.57%  2.64%  6.47%  5.75%  5.97%  3.23%
##    10:   4.05%  2.41%  5.85%  4.58%  5.57%  2.89%
##    11:   3.67%  2.50%  4.98%  4.56%  4.37%  2.65%
##    12:   3.43%  2.28%  4.79%  4.26%  4.26%  2.28%
##    13:   3.38%  2.13%  4.74%  3.86%  4.56%  2.42%
##    14:   3.18%  1.98%  4.20%  3.85%  4.19%  2.46%
##    15:   3.10%  2.09%  4.15%  3.80%  3.62%  2.46%
##    16:   2.97%  2.06%  3.79%  3.66%  3.62%  2.28%
##    17:   2.67%  1.79%  3.61%  3.31%  3.52%  1.68%
##    18:   2.52%  1.73%  3.12%  3.22%  3.26%  1.82%
##    19:   2.52%  1.64%  3.17%  3.31%  3.36%  1.73%
##    20:   2.46%  1.58%  3.30%  3.02%  3.36%  1.64%
##    21:   2.38%  1.52%  2.90%  3.12%  3.26%  1.68%
##    22:   2.30%  1.43%  2.81%  3.17%  3.20%  1.50%
##    23:   2.30%  1.46%  2.63%  3.22%  3.20%  1.59%
##    24:   2.34%  1.43%  2.99%  3.31%  3.05%  1.55%
##    25:   2.25%  1.43%  2.72%  3.21%  3.10%  1.36%
##    26:   2.22%  1.31%  2.77%  3.26%  2.95%  1.46%
##    27:   2.22%  1.34%  2.86%  3.07%  2.84%  1.59%
##    28:   2.14%  1.31%  2.54%  3.02%  2.89%  1.50%
##    29:   2.17%  1.40%  2.45%  2.97%  2.84%  1.68%
##    30:   2.07%  1.31%  2.45%  3.12%  2.48%  1.50%
##    31:   2.10%  1.28%  2.63%  2.83%  2.79%  1.50%
##    32:   2.02%  1.13%  2.63%  2.92%  2.74%  1.27%
##    33:   2.07%  1.25%  2.45%  2.97%  2.84%  1.41%
##    34:   2.06%  1.34%  2.37%  2.97%  2.84%  1.27%
##    35:   1.97%  1.28%  2.37%  2.83%  2.84%  1.05%
##    36:   1.91%  1.16%  2.45%  2.58%  2.84%  1.05%
##    37:   1.87%  1.10%  2.41%  2.63%  2.79%  0.96%
##    38:   1.86%  1.10%  2.37%  2.63%  2.58%  1.14%
##    39:   1.89%  1.10%  2.23%  2.92%  2.69%  1.09%
##    40:   1.86%  0.99%  2.23%  2.87%  2.84%  1.00%
##    41:   1.83%  0.99%  2.10%  2.78%  2.95%  1.00%
##    42:   1.88%  1.07%  2.28%  2.68%  2.84%  1.09%
##    43:   1.72%  0.93%  2.19%  2.58%  2.48%  1.00%
##    44:   1.83%  1.05%  2.19%  2.58%  2.79%  1.09%
##    45:   1.78%  0.99%  2.19%  2.68%  2.69%  0.96%
##    46:   1.81%  1.02%  2.14%  2.73%  2.69%  1.05%
##    47:   1.77%  0.96%  2.05%  2.78%  2.64%  1.05%
##    48:   1.68%  0.93%  2.01%  2.58%  2.53%  0.91%
##    49:   1.70%  0.90%  2.01%  2.68%  2.64%  0.86%
##    50:   1.64%  0.84%  1.92%  2.53%  2.53%  0.96%
##    51:   1.64%  0.87%  1.83%  2.68%  2.58%  0.82%
##    52:   1.62%  0.90%  1.74%  2.58%  2.58%  0.86%
##    53:   1.65%  0.90%  1.96%  2.63%  2.48%  0.82%
##    54:   1.58%  0.81%  1.92%  2.48%  2.43%  0.82%
##    55:   1.60%  0.90%  1.87%  2.53%  2.48%  0.77%
##    56:   1.65%  0.84%  2.01%  2.48%  2.69%  0.82%
##    57:   1.66%  0.84%  1.83%  2.48%  2.74%  1.00%
##    58:   1.65%  0.81%  1.96%  2.53%  2.64%  0.91%
##    59:   1.63%  0.81%  2.01%  2.34%  2.64%  0.96%
##    60:   1.65%  0.90%  1.96%  2.34%  2.69%  0.91%
##    61:   1.68%  0.90%  2.01%  2.44%  2.74%  0.91%
##    62:   1.61%  0.87%  1.83%  2.34%  2.64%  0.96%
##    63:   1.61%  0.81%  1.92%  2.24%  2.79%  0.91%
##    64:   1.69%  0.81%  2.01%  2.53%  2.84%  0.91%
##    65:   1.61%  0.90%  1.78%  2.34%  2.69%  0.91%
##    66:   1.56%  0.84%  1.87%  2.29%  2.53%  0.82%
##    67:   1.55%  0.81%  1.78%  2.29%  2.64%  0.82%
##    68:   1.58%  0.87%  1.78%  2.29%  2.64%  0.86%
##    69:   1.55%  0.90%  1.83%  2.29%  2.53%  0.68%
##    70:   1.55%  0.81%  1.83%  2.24%  2.74%  0.68%
##    71:   1.56%  0.78%  1.83%  2.34%  2.74%  0.73%
##    72:   1.51%  0.72%  1.74%  2.24%  2.74%  0.73%
##    73:   1.55%  0.87%  1.78%  2.24%  2.74%  0.68%
##    74:   1.60%  0.87%  1.83%  2.29%  2.74%  0.86%
##    75:   1.60%  0.87%  1.87%  2.34%  2.79%  0.68%
##    76:   1.61%  0.99%  1.87%  2.19%  2.69%  0.82%
##    77:   1.67%  0.93%  1.96%  2.29%  2.89%  0.86%
##    78:   1.61%  0.90%  1.78%  2.29%  2.79%  0.86%
##    79:   1.66%  0.87%  2.01%  2.29%  2.89%  0.82%
##    80:   1.64%  0.87%  2.10%  2.19%  2.89%  0.73%
##    81:   1.67%  0.90%  2.01%  2.24%  3.00%  0.82%
##    82:   1.67%  0.90%  1.96%  2.34%  3.00%  0.77%
##    83:   1.63%  0.90%  1.96%  2.19%  2.95%  0.73%
##    84:   1.59%  0.90%  1.83%  2.14%  2.84%  0.77%
##    85:   1.62%  0.90%  1.78%  2.19%  3.05%  0.77%
##    86:   1.60%  0.93%  1.87%  2.14%  2.84%  0.73%
##    87:   1.60%  0.96%  1.87%  2.05%  2.89%  0.77%
##    88:   1.60%  0.93%  1.92%  2.09%  2.79%  0.77%
##    89:   1.58%  0.93%  1.87%  2.00%  2.84%  0.77%
##    90:   1.57%  0.93%  1.87%  2.05%  2.74%  0.77%
##    91:   1.52%  0.84%  1.92%  1.85%  2.74%  0.77%
##    92:   1.55%  0.78%  1.96%  1.90%  2.84%  0.82%
##    93:   1.54%  0.78%  1.96%  1.85%  2.84%  0.82%
##    94:   1.55%  0.78%  2.01%  1.90%  2.84%  0.82%
##    95:   1.55%  0.78%  2.05%  2.00%  2.74%  0.77%
##    96:   1.55%  0.81%  2.01%  1.95%  2.74%  0.82%
##    97:   1.55%  0.78%  1.96%  2.00%  2.74%  0.82%
##    98:   1.55%  0.84%  2.01%  2.00%  2.69%  0.73%
##    99:   1.54%  0.81%  1.96%  2.05%  2.69%  0.73%
##   100:   1.55%  0.78%  1.96%  2.14%  2.69%  0.77%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.14%  4.91%  9.51%  9.63% 10.17%  3.29%
##     2:   7.59%  5.36%  9.18%  9.90%  9.59%  5.51%
##     3:   7.43%  4.93%  9.78%  9.59%  8.96%  5.53%
##     4:   7.00%  4.53%  9.26%  9.32%  8.41%  5.12%
##     5:   6.40%  4.29%  8.46%  7.96%  7.75%  4.89%
##     6:   5.79%  3.87%  7.80%  7.31%  7.21%  4.01%
##     7:   5.63%  3.72%  7.64%  6.67%  7.06%  4.30%
##     8:   5.15%  3.34%  7.09%  6.76%  6.19%  3.54%
##     9:   4.72%  3.28%  6.72%  5.94%  5.53%  3.05%
##    10:   4.35%  2.80%  6.09%  5.81%  4.96%  3.03%
##    11:   3.94%  2.37%  5.17%  5.14%  4.83%  3.16%
##    12:   3.67%  2.36%  5.02%  4.74%  4.77%  2.33%
##    13:   3.64%  2.24%  5.01%  5.22%  4.14%  2.46%
##    14:   3.21%  2.03%  3.84%  4.49%  4.09%  2.41%
##    15:   3.18%  2.09%  4.02%  4.19%  3.83%  2.46%
##    16:   3.04%  2.12%  3.75%  3.95%  3.72%  2.27%
##    17:   2.97%  2.18%  3.79%  3.95%  3.31%  2.14%
##    18:   2.86%  1.91%  3.84%  3.95%  3.31%  1.91%
##    19:   2.78%  1.85%  3.61%  3.85%  3.21%  1.96%
##    20:   2.62%  1.73%  3.70%  3.56%  2.90%  1.73%
##    21:   2.56%  1.61%  3.66%  3.51%  3.00%  1.64%
##    22:   2.59%  1.52%  3.75%  3.41%  3.26%  1.68%
##    23:   2.39%  1.49%  3.21%  3.56%  2.74%  1.50%
##    24:   2.37%  1.55%  2.90%  3.41%  3.05%  1.50%
##    25:   2.36%  1.49%  3.03%  3.51%  2.84%  1.50%
##    26:   2.28%  1.55%  3.03%  3.17%  2.84%  1.27%
##    27:   2.29%  1.58%  3.12%  3.07%  2.79%  1.36%
##    28:   2.17%  1.49%  2.86%  3.07%  2.53%  1.36%
##    29:   2.11%  1.31%  2.77%  3.07%  2.58%  1.36%
##    30:   2.01%  1.22%  2.63%  3.02%  2.53%  1.18%
##    31:   2.10%  1.37%  2.63%  3.07%  2.64%  1.27%
##    32:   2.08%  1.28%  2.81%  2.92%  2.69%  1.23%
##    33:   2.16%  1.37%  2.77%  2.92%  2.84%  1.41%
##    34:   2.10%  1.37%  2.72%  2.83%  2.84%  1.23%
##    35:   2.09%  1.31%  2.99%  2.68%  2.64%  1.32%
##    36:   2.04%  1.34%  2.95%  2.73%  2.64%  1.00%
##    37:   2.00%  1.31%  2.90%  2.58%  2.58%  1.09%
##    38:   2.01%  1.40%  2.81%  2.63%  2.48%  1.14%
##    39:   2.04%  1.37%  2.86%  2.83%  2.48%  1.09%
##    40:   1.95%  1.25%  2.86%  2.58%  2.38%  1.14%
##    41:   1.96%  1.19%  2.90%  2.63%  2.43%  1.14%
##    42:   1.90%  1.10%  2.95%  2.44%  2.48%  1.05%
##    43:   1.90%  1.16%  2.81%  2.58%  2.48%  0.96%
##    44:   1.90%  1.07%  2.81%  2.58%  2.48%  1.09%
##    45:   1.85%  1.10%  2.77%  2.53%  2.48%  0.86%
##    46:   1.83%  1.10%  2.81%  2.34%  2.43%  0.96%
##    47:   1.83%  1.07%  2.77%  2.44%  2.43%  0.96%
##    48:   1.80%  1.07%  2.68%  2.44%  2.43%  0.86%
##    49:   1.89%  1.13%  2.77%  2.53%  2.64%  0.86%
##    50:   1.85%  1.07%  2.77%  2.48%  2.64%  0.82%
##    51:   1.81%  1.02%  2.72%  2.44%  2.53%  0.86%
##    52:   1.84%  1.05%  2.59%  2.58%  2.64%  0.91%
##    53:   1.84%  0.99%  2.59%  2.48%  2.79%  0.96%
##    54:   1.82%  0.96%  2.59%  2.44%  2.74%  0.96%
##    55:   1.81%  1.05%  2.54%  2.24%  2.79%  0.96%
##    56:   1.80%  1.02%  2.50%  2.34%  2.74%  0.96%
##    57:   1.77%  0.96%  2.45%  2.29%  2.79%  0.91%
##    58:   1.77%  1.02%  2.54%  2.24%  2.64%  0.96%
##    59:   1.83%  1.02%  2.59%  2.44%  2.79%  0.86%
##    60:   1.78%  0.93%  2.59%  2.29%  2.79%  0.91%
##    61:   1.83%  0.99%  2.59%  2.44%  2.74%  1.00%
##    62:   1.77%  1.02%  2.50%  2.39%  2.58%  0.86%
##    63:   1.79%  0.99%  2.68%  2.39%  2.58%  0.86%
##    64:   1.73%  0.87%  2.50%  2.39%  2.69%  0.82%
##    65:   1.74%  0.93%  2.59%  2.34%  2.64%  0.77%
##    66:   1.75%  0.84%  2.63%  2.29%  2.74%  0.86%
##    67:   1.78%  0.90%  2.68%  2.39%  2.69%  0.86%
##    68:   1.77%  0.93%  2.59%  2.29%  2.69%  0.91%
##    69:   1.72%  0.84%  2.59%  2.29%  2.58%  0.91%
##    70:   1.72%  0.81%  2.68%  2.24%  2.58%  0.91%
##    71:   1.69%  0.78%  2.63%  2.19%  2.48%  0.96%
##    72:   1.72%  0.87%  2.72%  2.24%  2.48%  0.86%
##    73:   1.67%  0.87%  2.59%  2.05%  2.53%  0.86%
##    74:   1.70%  0.93%  2.63%  2.19%  2.43%  0.82%
##    75:   1.68%  0.90%  2.54%  2.19%  2.48%  0.82%
##    76:   1.71%  0.87%  2.54%  2.39%  2.48%  0.82%
##    77:   1.70%  0.87%  2.54%  2.29%  2.43%  0.91%
##    78:   1.67%  0.84%  2.59%  2.14%  2.48%  0.86%
##    79:   1.66%  0.90%  2.50%  2.19%  2.43%  0.82%
##    80:   1.66%  0.87%  2.54%  2.19%  2.38%  0.86%
##    81:   1.68%  0.81%  2.59%  2.29%  2.48%  0.82%
##    82:   1.65%  0.84%  2.63%  2.14%  2.43%  0.73%
##    83:   1.71%  0.90%  2.63%  2.29%  2.48%  0.77%
##    84:   1.65%  0.87%  2.54%  2.09%  2.48%  0.77%
##    85:   1.67%  0.90%  2.63%  2.19%  2.38%  0.77%
##    86:   1.64%  0.90%  2.50%  2.14%  2.38%  0.77%
##    87:   1.65%  0.87%  2.54%  2.09%  2.48%  0.77%
##    88:   1.67%  0.93%  2.54%  2.14%  2.48%  0.77%
##    89:   1.66%  0.81%  2.59%  2.19%  2.48%  0.77%
##    90:   1.68%  0.96%  2.59%  2.19%  2.38%  0.77%
##    91:   1.66%  0.93%  2.59%  2.09%  2.38%  0.77%
##    92:   1.66%  0.90%  2.63%  2.14%  2.33%  0.82%
##    93:   1.68%  0.93%  2.63%  2.14%  2.38%  0.82%
##    94:   1.62%  0.84%  2.54%  2.09%  2.33%  0.82%
##    95:   1.66%  0.84%  2.59%  2.19%  2.38%  0.82%
##    96:   1.68%  0.87%  2.59%  2.19%  2.48%  0.82%
##    97:   1.66%  0.87%  2.45%  2.19%  2.48%  0.82%
##    98:   1.67%  0.90%  2.50%  2.24%  2.43%  0.82%
##    99:   1.62%  0.84%  2.45%  2.19%  2.33%  0.82%
##   100:   1.64%  0.84%  2.54%  2.14%  2.38%  0.82%
## ntree      OOB      1      2      3      4      5
##     1:  10.39%  8.35% 14.90% 10.64%  9.72%  8.91%
##     2:  10.73%  6.40% 16.16% 12.64% 11.79%  8.79%
##     3:  10.27%  6.24% 15.81% 11.82% 11.41%  8.08%
##     4:   9.57%  6.27% 13.83% 11.28% 10.24%  7.90%
##     5:   8.89%  5.98% 13.40% 10.84%  9.21%  6.48%
##     6:   8.09%  5.38% 12.23% 10.25%  8.39%  5.58%
##     7:   6.96%  4.54% 10.58%  8.82%  7.30%  4.81%
##     8:   6.77%  4.32% 10.61%  8.23%  7.34%  4.60%
##     9:   5.99%  3.70%  9.30%  8.13%  6.18%  3.81%
##    10:   5.38%  3.37%  7.71%  7.58%  5.74%  3.64%
##    11:   4.84%  3.08%  7.68%  6.91%  4.97%  2.52%
##    12:   4.56%  2.90%  6.91%  6.29%  4.96%  2.65%
##    13:   4.26%  3.02%  6.07%  5.58%  5.24%  2.10%
##    14:   3.69%  2.47%  5.46%  4.96%  4.59%  1.69%
##    15:   3.52%  2.22%  5.33%  5.11%  4.29%  1.41%
##    16:   3.46%  2.22%  5.41%  5.01%  4.19%  1.19%
##    17:   3.36%  2.19%  5.40%  4.45%  4.08%  1.28%
##    18:   3.06%  1.83%  5.06%  4.10%  3.69%  1.28%
##    19:   2.93%  1.83%  4.80%  4.05%  3.74%  0.87%
##    20:   2.85%  1.86%  4.28%  4.45%  3.38%  0.91%
##    21:   2.74%  1.67%  4.24%  4.05%  3.38%  1.00%
##    22:   2.65%  1.58%  4.15%  4.05%  3.19%  0.91%
##    23:   2.54%  1.71%  3.76%  3.74%  3.14%  0.87%
##    24:   2.51%  1.49%  4.02%  3.85%  3.04%  0.78%
##    25:   2.34%  1.43%  3.59%  3.24%  3.14%  0.82%
##    26:   2.37%  1.49%  3.59%  3.29%  3.04%  0.96%
##    27:   2.25%  1.49%  3.50%  3.19%  2.94%  0.59%
##    28:   2.28%  1.58%  3.59%  3.14%  2.74%  0.73%
##    29:   2.22%  1.43%  3.33%  3.14%  2.84%  0.82%
##    30:   2.19%  1.34%  3.63%  3.24%  2.54%  0.68%
##    31:   2.19%  1.40%  3.37%  3.19%  2.59%  0.87%
##    32:   2.16%  1.22%  3.37%  3.29%  2.69%  0.77%
##    33:   2.16%  1.31%  3.29%  3.14%  2.89%  0.68%
##    34:   2.15%  1.40%  3.20%  3.34%  2.59%  0.68%
##    35:   2.09%  1.19%  3.24%  3.34%  2.44%  0.77%
##    36:   2.06%  1.28%  3.29%  2.99%  2.29%  0.87%
##    37:   2.09%  1.31%  3.16%  3.19%  2.39%  0.87%
##    38:   1.94%  1.28%  3.03%  2.88%  2.19%  0.68%
##    39:   1.89%  1.07%  2.98%  2.94%  2.29%  0.68%
##    40:   1.87%  1.00%  2.98%  2.83%  2.34%  0.68%
##    41:   1.84%  1.10%  2.77%  2.83%  2.24%  0.73%
##    42:   1.81%  1.07%  2.77%  2.78%  2.19%  0.68%
##    43:   1.78%  1.07%  2.81%  2.63%  2.19%  0.64%
##    44:   1.82%  1.16%  2.68%  2.73%  2.39%  0.55%
##    45:   1.76%  0.97%  2.68%  2.63%  2.39%  0.59%
##    46:   1.81%  1.07%  2.64%  2.78%  2.29%  0.73%
##    47:   1.77%  1.07%  2.72%  2.63%  2.09%  0.73%
##    48:   1.78%  1.04%  2.77%  2.58%  2.19%  0.77%
##    49:   1.77%  1.13%  2.59%  2.68%  2.14%  0.68%
##    50:   1.71%  1.04%  2.46%  2.68%  2.24%  0.55%
##    51:   1.77%  1.10%  2.51%  2.63%  2.29%  0.73%
##    52:   1.80%  1.16%  2.72%  2.58%  2.19%  0.73%
##    53:   1.74%  1.10%  2.42%  2.53%  2.24%  0.82%
##    54:   1.78%  1.07%  2.64%  2.63%  2.29%  0.73%
##    55:   1.77%  1.13%  2.68%  2.53%  2.19%  0.68%
##    56:   1.85%  1.10%  2.94%  2.68%  2.24%  0.73%
##    57:   1.82%  1.10%  2.77%  2.68%  2.24%  0.73%
##    58:   1.83%  1.10%  2.72%  2.68%  2.29%  0.77%
##    59:   1.82%  1.10%  2.90%  2.63%  2.09%  0.77%
##    60:   1.74%  1.07%  2.68%  2.48%  2.14%  0.73%
##    61:   1.73%  1.13%  2.68%  2.38%  2.04%  0.77%
##    62:   1.73%  1.13%  2.72%  2.43%  2.09%  0.64%
##    63:   1.72%  1.16%  2.51%  2.38%  2.14%  0.73%
##    64:   1.75%  1.19%  2.59%  2.43%  2.09%  0.77%
##    65:   1.71%  1.13%  2.38%  2.48%  2.19%  0.73%
##    66:   1.69%  1.04%  2.51%  2.58%  2.09%  0.64%
##    67:   1.66%  1.04%  2.46%  2.38%  2.14%  0.64%
##    68:   1.69%  1.04%  2.55%  2.43%  2.09%  0.73%
##    69:   1.70%  1.07%  2.59%  2.38%  2.14%  0.68%
##    70:   1.68%  0.94%  2.51%  2.63%  2.14%  0.64%
##    71:   1.68%  0.97%  2.42%  2.58%  2.19%  0.68%
##    72:   1.64%  0.97%  2.51%  2.33%  2.09%  0.68%
##    73:   1.68%  1.00%  2.51%  2.53%  2.09%  0.68%
##    74:   1.66%  0.97%  2.38%  2.48%  2.24%  0.64%
##    75:   1.61%  0.88%  2.33%  2.58%  2.04%  0.68%
##    76:   1.66%  0.97%  2.51%  2.58%  2.09%  0.55%
##    77:   1.63%  0.97%  2.42%  2.48%  2.14%  0.55%
##    78:   1.61%  1.00%  2.46%  2.43%  2.09%  0.46%
##    79:   1.60%  0.88%  2.55%  2.43%  2.04%  0.50%
##    80:   1.60%  0.91%  2.51%  2.48%  1.99%  0.50%
##    81:   1.57%  0.91%  2.42%  2.43%  1.99%  0.50%
##    82:   1.59%  0.91%  2.42%  2.48%  1.99%  0.55%
##    83:   1.57%  0.82%  2.55%  2.38%  1.99%  0.55%
##    84:   1.55%  0.79%  2.42%  2.38%  1.99%  0.59%
##    85:   1.53%  0.82%  2.42%  2.28%  1.99%  0.55%
##    86:   1.49%  0.79%  2.38%  2.18%  1.99%  0.55%
##    87:   1.51%  0.82%  2.25%  2.33%  1.99%  0.59%
##    88:   1.49%  0.76%  2.33%  2.23%  1.94%  0.64%
##    89:   1.49%  0.82%  2.25%  2.23%  1.89%  0.64%
##    90:   1.48%  0.76%  2.29%  2.18%  1.99%  0.59%
##    91:   1.44%  0.73%  2.38%  1.97%  1.99%  0.55%
##    92:   1.49%  0.79%  2.38%  2.13%  1.99%  0.59%
##    93:   1.46%  0.82%  2.29%  2.07%  1.94%  0.55%
##    94:   1.44%  0.79%  2.33%  2.07%  1.79%  0.59%
##    95:   1.44%  0.76%  2.33%  2.18%  1.84%  0.50%
##    96:   1.44%  0.79%  2.33%  2.13%  1.84%  0.50%
##    97:   1.43%  0.76%  2.29%  2.02%  1.89%  0.55%
##    98:   1.42%  0.79%  2.29%  2.07%  1.79%  0.50%
##    99:   1.41%  0.79%  2.29%  2.02%  1.79%  0.50%
##   100:   1.49%  0.85%  2.38%  2.23%  1.79%  0.59%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.84%  4.59% 11.19% 11.84%  8.07%  5.30%
##     2:   7.77%  4.35% 11.62% 10.78%  8.48%  5.28%
##     3:   7.64%  4.65% 12.17%  9.71%  6.98%  6.06%
##     4:   7.26%  4.23% 11.38%  9.42%  7.00%  5.71%
##     5:   6.92%  4.13% 10.75%  8.54%  7.20%  5.34%
##     6:   6.42%  3.70%  9.98%  7.55%  7.47%  4.77%
##     7:   5.73%  3.14%  8.66%  7.29%  6.62%  4.28%
##     8:   5.49%  3.05%  8.54%  6.37%  7.01%  3.74%
##     9:   4.82%  2.69%  7.44%  5.43%  5.77%  3.84%
##    10:   4.57%  2.50%  6.91%  5.40%  5.59%  3.55%
##    11:   4.37%  2.43%  6.71%  4.98%  5.41%  3.30%
##    12:   3.90%  2.05%  6.17%  4.82%  4.55%  2.83%
##    13:   3.78%  2.26%  5.64%  4.51%  4.49%  2.79%
##    14:   3.68%  2.26%  5.24%  4.61%  4.24%  2.83%
##    15:   3.42%  1.95%  5.02%  4.30%  4.19%  2.42%
##    16:   3.33%  1.74%  5.23%  3.80%  4.28%  2.42%
##    17:   3.13%  1.65%  4.97%  3.80%  3.73%  2.28%
##    18:   2.99%  1.43%  4.71%  3.44%  3.88%  2.28%
##    19:   2.73%  1.43%  4.41%  3.14%  3.58%  1.78%
##    20:   2.73%  1.34%  4.41%  3.19%  3.63%  1.78%
##    21:   2.60%  1.16%  4.41%  3.29%  3.38%  1.50%
##    22:   2.59%  1.19%  4.32%  3.14%  3.53%  1.50%
##    23:   2.58%  1.22%  4.15%  3.24%  3.63%  1.41%
##    24:   2.63%  1.34%  4.06%  3.39%  3.58%  1.50%
##    25:   2.51%  1.28%  4.06%  2.94%  3.43%  1.50%
##    26:   2.45%  1.28%  3.72%  3.09%  3.33%  1.46%
##    27:   2.51%  1.19%  4.02%  3.34%  3.33%  1.37%
##    28:   2.35%  1.16%  3.80%  3.19%  3.14%  1.14%
##    29:   2.42%  1.16%  3.80%  3.24%  3.19%  1.41%
##    30:   2.51%  1.04%  3.93%  3.39%  3.48%  1.50%
##    31:   2.37%  0.94%  3.93%  3.09%  3.33%  1.32%
##    32:   2.31%  1.04%  3.76%  2.99%  3.24%  1.23%
##    33:   2.31%  1.16%  3.50%  3.09%  3.19%  1.28%
##    34:   2.34%  1.13%  3.59%  3.19%  3.19%  1.28%
##    35:   2.29%  1.16%  3.37%  3.09%  3.24%  1.28%
##    36:   2.28%  1.13%  3.55%  3.04%  3.04%  1.28%
##    37:   2.21%  1.10%  3.55%  2.94%  2.94%  1.14%
##    38:   2.24%  1.19%  3.55%  2.94%  2.99%  1.14%
##    39:   2.18%  1.16%  3.33%  2.78%  3.04%  1.19%
##    40:   2.20%  1.07%  3.24%  2.99%  3.19%  1.19%
##    41:   2.13%  1.10%  3.11%  2.99%  2.99%  1.09%
##    42:   2.13%  1.04%  3.20%  2.94%  3.04%  1.09%
##    43:   2.12%  1.13%  3.20%  2.73%  3.04%  1.09%
##    44:   2.08%  1.00%  3.16%  2.83%  2.94%  1.09%
##    45:   2.09%  1.04%  3.16%  2.78%  2.99%  1.09%
##    46:   2.01%  0.88%  3.20%  2.88%  2.79%  0.96%
##    47:   2.02%  0.88%  3.11%  2.83%  2.89%  1.05%
##    48:   2.08%  0.97%  3.20%  2.88%  2.94%  1.05%
##    49:   2.03%  0.91%  3.11%  2.88%  2.79%  1.09%
##    50:   2.06%  0.88%  3.11%  2.94%  2.99%  1.05%
##    51:   1.99%  0.82%  3.16%  2.88%  2.74%  1.00%
##    52:   2.00%  0.88%  3.07%  2.83%  2.84%  1.05%
##    53:   2.00%  0.88%  2.98%  2.83%  2.99%  0.96%
##    54:   2.00%  0.94%  3.03%  2.73%  2.89%  1.05%
##    55:   2.00%  0.97%  2.94%  2.83%  2.79%  1.09%
##    56:   1.94%  0.91%  2.77%  2.73%  2.99%  0.96%
##    57:   1.96%  0.85%  2.77%  2.78%  3.04%  1.05%
##    58:   1.97%  0.88%  2.90%  2.68%  3.14%  0.91%
##    59:   1.99%  0.91%  3.03%  2.68%  2.99%  0.96%
##    60:   1.97%  0.79%  2.98%  2.68%  3.04%  1.05%
##    61:   1.94%  0.88%  2.90%  2.63%  2.99%  0.96%
##    62:   1.90%  0.82%  2.90%  2.58%  2.94%  0.91%
##    63:   1.94%  0.82%  2.94%  2.58%  2.99%  1.05%
##    64:   1.91%  0.79%  2.81%  2.63%  2.89%  1.09%
##    65:   1.95%  0.82%  2.98%  2.63%  2.94%  1.05%
##    66:   1.94%  0.91%  2.85%  2.58%  2.94%  1.00%
##    67:   1.94%  0.88%  2.90%  2.58%  3.04%  0.91%
##    68:   1.92%  0.85%  2.68%  2.63%  2.99%  1.09%
##    69:   1.94%  0.88%  2.81%  2.58%  3.04%  1.05%
##    70:   1.97%  0.91%  2.81%  2.63%  3.09%  1.05%
##    71:   1.94%  0.88%  2.77%  2.58%  2.99%  1.09%
##    72:   1.89%  0.85%  2.64%  2.58%  2.99%  1.05%
##    73:   1.90%  0.76%  2.81%  2.58%  3.04%  1.00%
##    74:   1.92%  0.79%  2.81%  2.53%  3.04%  1.09%
##    75:   1.90%  0.79%  2.72%  2.63%  3.04%  1.00%
##    76:   1.89%  0.76%  2.77%  2.58%  2.99%  1.05%
##    77:   1.85%  0.73%  2.68%  2.53%  2.94%  1.05%
##    78:   1.84%  0.70%  2.64%  2.63%  2.89%  1.05%
##    79:   1.87%  0.76%  2.68%  2.63%  2.94%  1.00%
##    80:   1.86%  0.73%  2.72%  2.58%  2.84%  1.09%
##    81:   1.86%  0.82%  2.64%  2.58%  2.84%  1.05%
##    82:   1.83%  0.82%  2.59%  2.48%  2.84%  1.00%
##    83:   1.88%  0.76%  2.68%  2.58%  2.94%  1.09%
##    84:   1.83%  0.76%  2.55%  2.53%  2.89%  1.09%
##    85:   1.83%  0.79%  2.59%  2.53%  2.79%  1.09%
##    86:   1.83%  0.70%  2.64%  2.68%  2.84%  1.00%
##    87:   1.78%  0.70%  2.51%  2.58%  2.79%  1.00%
##    88:   1.76%  0.67%  2.46%  2.48%  2.84%  1.00%
##    89:   1.76%  0.70%  2.42%  2.53%  2.79%  1.00%
##    90:   1.74%  0.70%  2.46%  2.48%  2.79%  0.91%
##    91:   1.71%  0.73%  2.33%  2.38%  2.79%  0.91%
##    92:   1.77%  0.76%  2.46%  2.48%  2.84%  0.91%
##    93:   1.76%  0.82%  2.42%  2.43%  2.79%  0.91%
##    94:   1.75%  0.79%  2.29%  2.48%  2.84%  0.96%
##    95:   1.78%  0.82%  2.38%  2.53%  2.84%  0.96%
##    96:   1.77%  0.79%  2.38%  2.53%  2.84%  0.91%
##    97:   1.76%  0.82%  2.42%  2.43%  2.79%  0.91%
##    98:   1.78%  0.82%  2.33%  2.53%  2.89%  0.96%
##    99:   1.73%  0.79%  2.20%  2.53%  2.79%  0.96%
##   100:   1.76%  0.82%  2.25%  2.53%  2.84%  0.96%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   6.32%  4.31%  8.84%  8.02%  5.27%  6.10%
##     2:   7.14%  4.68%  9.69% 10.19%  6.90%  5.69%
##     3:   7.12%  4.51%  9.25% 10.93%  7.53%  5.00%
##     4:   7.25%  4.83%  9.26% 10.65%  7.96%  4.95%
##     5:   6.64%  5.24%  7.71%  9.78%  7.04%  4.37%
##     6:   6.45%  4.86%  7.89%  9.11%  6.77%  4.58%
##     7:   5.89%  4.49%  7.20%  8.27%  5.86%  4.48%
##     8:   5.19%  3.73%  6.45%  7.58%  5.98%  3.15%
##     9:   5.26%  3.56%  6.57%  7.92%  5.60%  3.68%
##    10:   4.68%  3.17%  5.85%  7.06%  5.12%  3.14%
##    11:   4.18%  2.66%  5.21%  6.37%  4.74%  2.85%
##    12:   3.99%  2.38%  4.90%  5.90%  4.83%  2.93%
##    13:   3.58%  2.26%  4.51%  5.49%  4.33%  2.15%
##    14:   3.64%  2.35%  4.76%  5.58%  4.18%  2.15%
##    15:   3.34%  2.16%  4.07%  4.96%  4.38%  1.92%
##    16:   3.12%  2.16%  3.85%  4.86%  3.73%  1.64%
##    17:   3.04%  1.92%  3.76%  4.81%  3.83%  1.64%
##    18:   2.98%  1.89%  3.63%  4.91%  3.53%  1.69%
##    19:   3.01%  1.89%  3.76%  4.76%  3.88%  1.55%
##    20:   2.79%  1.77%  3.55%  4.05%  3.83%  1.46%
##    21:   2.76%  1.80%  3.59%  4.40%  3.24%  1.41%
##    22:   2.65%  1.77%  3.50%  3.85%  3.38%  1.32%
##    23:   2.67%  1.80%  3.11%  4.10%  3.48%  1.46%
##    24:   2.59%  1.64%  3.29%  4.05%  3.14%  1.46%
##    25:   2.52%  1.58%  2.94%  4.00%  3.29%  1.46%
##    26:   2.48%  1.71%  3.03%  3.80%  2.99%  1.41%
##    27:   2.38%  1.52%  3.03%  3.59%  2.99%  1.32%
##    28:   2.35%  1.52%  2.94%  3.59%  3.09%  1.19%
##    29:   2.34%  1.52%  3.07%  3.24%  3.19%  1.19%
##    30:   2.26%  1.43%  2.98%  3.29%  3.04%  1.09%
##    31:   2.22%  1.37%  3.07%  3.04%  3.04%  1.09%
##    32:   2.30%  1.55%  3.03%  3.19%  3.14%  1.09%
##    33:   2.22%  1.43%  2.85%  3.44%  2.94%  1.00%
##    34:   2.18%  1.40%  2.94%  3.29%  2.74%  1.05%
##    35:   2.19%  1.43%  2.98%  3.14%  2.89%  1.00%
##    36:   2.15%  1.28%  2.98%  3.04%  2.94%  1.05%
##    37:   2.23%  1.28%  3.07%  3.24%  3.09%  1.09%
##    38:   2.20%  1.28%  3.03%  3.19%  2.94%  1.14%
##    39:   2.28%  1.43%  3.07%  3.29%  2.99%  1.19%
##    40:   2.20%  1.40%  2.81%  3.14%  2.89%  1.28%
##    41:   2.20%  1.37%  2.90%  3.14%  2.84%  1.28%
##    42:   2.19%  1.37%  2.77%  3.24%  2.79%  1.32%
##    43:   2.17%  1.34%  2.81%  3.19%  2.79%  1.28%
##    44:   2.16%  1.34%  2.64%  3.14%  2.89%  1.32%
##    45:   2.17%  1.34%  2.72%  3.04%  2.94%  1.37%
##    46:   2.18%  1.40%  2.55%  3.14%  3.04%  1.32%
##    47:   2.17%  1.37%  2.64%  3.14%  2.99%  1.28%
##    48:   2.14%  1.31%  2.42%  3.19%  3.09%  1.28%
##    49:   2.14%  1.28%  2.55%  2.99%  2.99%  1.46%
##    50:   2.13%  1.28%  2.68%  2.73%  3.09%  1.41%
##    51:   2.17%  1.34%  2.72%  2.88%  3.04%  1.41%
##    52:   2.11%  1.28%  2.46%  2.78%  3.19%  1.37%
##    53:   2.16%  1.28%  2.59%  2.78%  3.33%  1.37%
##    54:   2.12%  1.22%  2.51%  2.99%  3.24%  1.28%
##    55:   2.13%  1.19%  2.59%  2.88%  3.24%  1.37%
##    56:   2.17%  1.28%  2.64%  2.88%  3.24%  1.37%
##    57:   2.17%  1.31%  2.64%  2.83%  3.29%  1.32%
##    58:   2.10%  1.25%  2.55%  2.68%  3.29%  1.28%
##    59:   2.10%  1.28%  2.72%  2.58%  3.14%  1.28%
##    60:   2.13%  1.19%  2.64%  2.94%  3.29%  1.23%
##    61:   2.09%  1.19%  2.64%  2.88%  3.09%  1.23%
##    62:   2.04%  1.19%  2.38%  2.78%  3.09%  1.32%
##    63:   2.06%  1.22%  2.46%  2.88%  3.04%  1.28%
##    64:   2.06%  1.22%  2.59%  2.73%  2.99%  1.28%
##    65:   2.11%  1.25%  2.77%  2.78%  2.94%  1.32%
##    66:   2.00%  1.22%  2.51%  2.63%  2.99%  1.19%
##    67:   2.00%  1.16%  2.55%  2.83%  2.84%  1.19%
##    68:   2.03%  1.22%  2.55%  2.73%  2.89%  1.28%
##    69:   2.04%  1.19%  2.59%  2.73%  2.94%  1.28%
##    70:   2.01%  1.16%  2.59%  2.68%  2.99%  1.19%
##    71:   1.98%  1.13%  2.51%  2.68%  2.99%  1.14%
##    72:   2.02%  1.19%  2.51%  2.78%  2.99%  1.19%
##    73:   2.03%  1.19%  2.59%  2.83%  2.94%  1.14%
##    74:   2.00%  1.10%  2.51%  2.83%  2.99%  1.19%
##    75:   2.03%  1.13%  2.68%  2.83%  2.94%  1.14%
##    76:   2.02%  1.10%  2.59%  2.83%  3.04%  1.14%
##    77:   2.06%  1.04%  2.72%  2.88%  3.09%  1.19%
##    78:   2.02%  1.13%  2.46%  2.99%  2.99%  1.14%
##    79:   2.01%  1.07%  2.51%  2.88%  2.99%  1.23%
##    80:   1.96%  1.00%  2.51%  2.78%  2.94%  1.19%
##    81:   1.98%  1.00%  2.68%  2.58%  2.99%  1.23%
##    82:   2.01%  0.97%  2.72%  2.83%  2.99%  1.19%
##    83:   2.00%  1.04%  2.68%  2.68%  2.99%  1.19%
##    84:   1.99%  1.04%  2.64%  2.68%  2.99%  1.19%
##    85:   1.97%  1.07%  2.72%  2.53%  2.94%  1.14%
##    86:   1.94%  1.04%  2.64%  2.63%  2.89%  1.09%
##    87:   1.99%  0.97%  2.64%  2.78%  2.99%  1.19%
##    88:   1.97%  1.00%  2.68%  2.73%  2.89%  1.14%
##    89:   1.97%  0.97%  2.68%  2.73%  2.94%  1.14%
##    90:   1.94%  0.97%  2.59%  2.68%  2.89%  1.14%
##    91:   1.94%  1.04%  2.59%  2.73%  2.79%  1.09%
##    92:   1.96%  1.07%  2.72%  2.63%  2.84%  1.09%
##    93:   1.98%  1.00%  2.64%  2.78%  2.89%  1.19%
##    94:   1.95%  1.07%  2.64%  2.53%  2.89%  1.19%
##    95:   1.93%  1.07%  2.55%  2.58%  2.89%  1.09%
##    96:   1.89%  1.04%  2.55%  2.48%  2.79%  1.14%
##    97:   1.93%  1.00%  2.59%  2.53%  2.94%  1.14%
##    98:   1.95%  1.04%  2.55%  2.63%  2.99%  1.14%
##    99:   1.92%  1.00%  2.68%  2.53%  2.89%  1.05%
##   100:   1.88%  0.97%  2.55%  2.48%  2.84%  1.09%
## ntree      OOB      1      2      3      4      5
##     1:   9.46%  6.90% 10.37% 11.08% 11.08%  9.51%
##     2:  10.45%  7.09% 13.10% 11.63% 11.76% 10.64%
##     3:  10.19%  7.25% 12.19% 12.71% 10.73%  9.93%
##     4:   9.83%  7.00% 12.56% 11.40%  9.77%  9.91%
##     5:   9.08%  6.34% 11.93% 10.39%  9.30%  8.87%
##     6:   8.53%  6.37% 11.53% 10.05%  8.51%  7.26%
##     7:   7.67%  5.58% 10.69%  9.01%  7.75%  6.34%
##     8:   6.95%  5.19%  9.86%  8.24%  6.41%  5.86%
##     9:   6.42%  5.16%  8.48%  7.50%  6.32%  5.27%
##    10:   5.69%  4.01%  7.61%  7.41%  6.17%  4.22%
##    11:   5.10%  3.97%  6.80%  6.37%  5.16%  3.83%
##    12:   4.79%  3.23%  7.00%  5.91%  5.09%  3.55%
##    13:   4.29%  3.02%  5.69%  5.95%  4.41%  3.13%
##    14:   4.01%  3.31%  5.39%  5.35%  3.89%  2.53%
##    15:   3.61%  2.87%  5.08%  4.89%  3.32%  2.25%
##    16:   3.51%  2.63%  4.95%  4.73%  3.42%  2.30%
##    17:   3.10%  2.36%  4.13%  4.18%  3.11%  2.16%
##    18:   3.08%  2.48%  4.13%  4.48%  3.11%  1.61%
##    19:   2.96%  2.36%  4.09%  4.23%  3.16%  1.38%
##    20:   2.70%  2.15%  3.66%  4.17%  2.74%  1.15%
##    21:   2.56%  1.94%  3.57%  3.67%  2.85%  1.19%
##    22:   2.40%  1.94%  3.23%  3.32%  2.43%  1.38%
##    23:   2.40%  2.27%  3.27%  2.61%  2.69%  1.24%
##    24:   2.45%  2.27%  3.31%  3.07%  2.64%  1.10%
##    25:   2.31%  1.91%  3.31%  2.92%  2.64%  1.01%
##    26:   2.29%  2.00%  3.31%  2.92%  2.33%  1.06%
##    27:   2.11%  1.73%  3.10%  2.61%  2.43%  0.92%
##    28:   2.10%  1.76%  3.19%  2.41%  2.33%  0.96%
##    29:   2.06%  1.61%  3.06%  2.51%  2.23%  1.10%
##    30:   2.02%  1.73%  2.76%  2.66%  2.12%  1.01%
##    31:   2.00%  1.70%  2.76%  2.56%  2.12%  1.01%
##    32:   1.83%  1.58%  2.63%  2.21%  2.07%  0.83%
##    33:   1.87%  1.73%  2.67%  2.16%  2.02%  0.83%
##    34:   1.86%  1.52%  2.88%  2.21%  2.07%  0.78%
##    35:   1.81%  1.40%  2.58%  2.31%  2.18%  0.83%
##    36:   1.81%  1.43%  2.67%  2.11%  2.07%  0.96%
##    37:   1.87%  1.52%  2.84%  2.31%  2.07%  0.78%
##    38:   1.84%  1.52%  2.54%  2.56%  1.92%  0.87%
##    39:   1.76%  1.40%  2.37%  2.51%  1.97%  0.78%
##    40:   1.76%  1.37%  2.50%  2.46%  1.92%  0.78%
##    41:   1.76%  1.31%  2.67%  2.41%  1.92%  0.73%
##    42:   1.73%  1.28%  2.50%  2.56%  1.92%  0.69%
##    43:   1.70%  1.16%  2.50%  2.61%  1.81%  0.73%
##    44:   1.63%  1.16%  2.41%  2.36%  1.92%  0.60%
##    45:   1.72%  1.19%  2.50%  2.71%  1.92%  0.60%
##    46:   1.62%  1.10%  2.45%  2.51%  1.81%  0.55%
##    47:   1.57%  1.13%  2.20%  2.41%  1.76%  0.64%
##    48:   1.54%  1.13%  2.37%  2.16%  1.61%  0.64%
##    49:   1.60%  1.22%  2.20%  2.36%  1.76%  0.73%
##    50:   1.51%  1.22%  2.15%  2.21%  1.66%  0.51%
##    51:   1.49%  1.16%  2.24%  1.96%  1.71%  0.55%
##    52:   1.52%  1.13%  2.24%  2.11%  1.76%  0.60%
##    53:   1.55%  1.04%  2.15%  2.26%  1.86%  0.73%
##    54:   1.49%  1.10%  2.07%  2.11%  1.76%  0.64%
##    55:   1.45%  1.01%  2.02%  2.16%  1.71%  0.64%
##    56:   1.45%  1.01%  2.07%  2.11%  1.76%  0.60%
##    57:   1.44%  1.01%  2.20%  1.96%  1.76%  0.55%
##    58:   1.48%  1.13%  2.02%  2.06%  1.81%  0.60%
##    59:   1.41%  0.98%  2.07%  1.91%  1.81%  0.55%
##    60:   1.38%  1.01%  2.02%  1.76%  1.81%  0.51%
##    61:   1.41%  0.98%  2.11%  1.76%  1.86%  0.60%
##    62:   1.42%  1.07%  2.02%  1.81%  1.81%  0.60%
##    63:   1.41%  0.89%  2.24%  1.76%  1.81%  0.64%
##    64:   1.39%  0.89%  2.07%  1.91%  1.76%  0.64%
##    65:   1.36%  0.95%  2.02%  1.81%  1.71%  0.55%
##    66:   1.37%  0.95%  2.02%  1.86%  1.81%  0.46%
##    67:   1.37%  0.83%  2.11%  1.96%  1.71%  0.55%
##    68:   1.29%  0.86%  1.89%  1.76%  1.71%  0.51%
##    69:   1.27%  0.86%  1.89%  1.66%  1.66%  0.51%
##    70:   1.31%  0.83%  1.98%  1.76%  1.76%  0.51%
##    71:   1.30%  0.89%  2.02%  1.66%  1.66%  0.51%
##    72:   1.28%  0.81%  1.85%  1.71%  1.76%  0.60%
##    73:   1.27%  0.86%  1.94%  1.61%  1.66%  0.51%
##    74:   1.23%  0.86%  1.76%  1.71%  1.55%  0.51%
##    75:   1.21%  0.81%  1.81%  1.61%  1.61%  0.51%
##    76:   1.26%  0.86%  1.98%  1.56%  1.61%  0.51%
##    77:   1.27%  0.83%  1.94%  1.61%  1.66%  0.55%
##    78:   1.20%  0.86%  1.94%  1.41%  1.40%  0.55%
##    79:   1.21%  0.83%  1.98%  1.41%  1.50%  0.51%
##    80:   1.20%  0.81%  1.85%  1.51%  1.55%  0.51%
##    81:   1.19%  0.81%  1.94%  1.41%  1.50%  0.51%
##    82:   1.21%  0.86%  1.81%  1.46%  1.55%  0.55%
##    83:   1.18%  0.83%  1.81%  1.36%  1.61%  0.51%
##    84:   1.20%  0.86%  1.89%  1.36%  1.55%  0.51%
##    85:   1.14%  0.81%  1.89%  1.26%  1.45%  0.46%
##    86:   1.20%  0.95%  1.89%  1.31%  1.50%  0.46%
##    87:   1.15%  0.92%  1.68%  1.31%  1.50%  0.46%
##    88:   1.17%  0.95%  1.72%  1.26%  1.61%  0.46%
##    89:   1.17%  0.95%  1.81%  1.26%  1.50%  0.46%
##    90:   1.20%  0.95%  1.94%  1.26%  1.55%  0.41%
##    91:   1.17%  0.98%  1.72%  1.31%  1.55%  0.41%
##    92:   1.20%  0.98%  1.81%  1.26%  1.66%  0.41%
##    93:   1.17%  0.98%  1.76%  1.16%  1.66%  0.41%
##    94:   1.20%  0.92%  1.94%  1.21%  1.66%  0.41%
##    95:   1.16%  0.92%  1.76%  1.26%  1.61%  0.41%
##    96:   1.17%  0.95%  1.76%  1.26%  1.61%  0.41%
##    97:   1.18%  0.92%  1.85%  1.31%  1.55%  0.41%
##    98:   1.21%  0.89%  1.89%  1.41%  1.61%  0.41%
##    99:   1.22%  0.83%  1.98%  1.46%  1.61%  0.46%
##   100:   1.22%  0.89%  1.94%  1.41%  1.61%  0.46%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.10%  5.27% 10.12%  7.33%  9.22%  4.46%
##     2:   7.65%  5.80% 10.43%  8.87%  8.62%  5.62%
##     3:   7.44%  5.86% 10.79%  7.67%  8.15%  5.53%
##     4:   6.97%  4.74% 10.66%  8.05%  7.73%  4.91%
##     5:   6.60%  4.39%  9.40%  7.58%  7.35%  5.48%
##     6:   6.22%  4.22%  8.60%  8.23%  6.49%  4.68%
##     7:   5.91%  3.63%  8.51%  7.35%  7.02%  4.37%
##     8:   5.37%  3.28%  7.96%  6.86%  5.81%  4.12%
##     9:   5.01%  3.16%  7.08%  6.49%  5.61%  3.77%
##    10:   4.41%  2.68%  6.68%  5.64%  4.93%  3.06%
##    11:   4.10%  2.31%  6.61%  5.06%  4.91%  2.59%
##    12:   3.93%  2.27%  6.63%  5.45%  4.02%  2.12%
##    13:   3.64%  2.21%  5.96%  4.53%  3.95%  2.26%
##    14:   3.28%  2.09%  5.52%  4.07%  3.58%  1.75%
##    15:   3.16%  2.03%  5.34%  3.67%  3.53%  1.79%
##    16:   3.01%  2.00%  5.30%  3.42%  3.11%  1.65%
##    17:   2.85%  1.82%  5.04%  3.07%  3.21%  1.56%
##    18:   2.83%  1.76%  4.65%  3.37%  3.26%  1.65%
##    19:   2.69%  1.70%  4.43%  3.17%  3.31%  1.38%
##    20:   2.77%  1.79%  4.18%  3.67%  3.00%  1.74%
##    21:   2.62%  1.49%  4.09%  3.52%  3.00%  1.61%
##    22:   2.63%  1.67%  4.22%  3.57%  2.59%  1.61%
##    23:   2.56%  1.70%  4.09%  3.32%  2.74%  1.42%
##    24:   2.45%  1.46%  3.83%  3.27%  2.85%  1.42%
##    25:   2.35%  1.22%  4.00%  2.81%  2.85%  1.47%
##    26:   2.33%  1.46%  3.79%  2.66%  2.80%  1.38%
##    27:   2.26%  1.40%  3.57%  2.76%  2.95%  1.10%
##    28:   2.22%  1.34%  3.66%  2.66%  2.85%  1.06%
##    29:   2.22%  1.25%  3.75%  2.56%  2.90%  1.15%
##    30:   2.28%  1.43%  3.70%  2.66%  2.95%  1.15%
##    31:   2.17%  1.34%  3.57%  2.46%  2.85%  1.06%
##    32:   2.20%  1.34%  3.83%  2.36%  2.74%  1.15%
##    33:   2.07%  1.19%  3.57%  2.26%  2.85%  0.96%
##    34:   2.03%  1.25%  3.36%  2.31%  2.64%  1.01%
##    35:   2.10%  1.13%  3.44%  2.71%  2.69%  1.06%
##    36:   2.00%  1.10%  3.27%  2.36%  2.64%  1.10%
##    37:   1.99%  1.04%  3.31%  2.61%  2.33%  1.15%
##    38:   1.96%  1.10%  3.31%  2.36%  2.33%  1.15%
##    39:   1.99%  1.10%  3.44%  2.31%  2.38%  1.15%
##    40:   1.92%  1.10%  3.31%  2.26%  2.28%  1.06%
##    41:   1.93%  1.10%  3.40%  2.21%  2.23%  1.10%
##    42:   1.93%  1.16%  3.10%  2.46%  2.23%  1.10%
##    43:   1.88%  1.22%  2.97%  2.21%  2.23%  1.10%
##    44:   1.87%  1.01%  3.10%  2.26%  2.38%  1.06%
##    45:   1.81%  1.01%  3.06%  2.11%  2.23%  1.06%
##    46:   1.86%  1.10%  3.19%  2.16%  2.23%  1.01%
##    47:   1.89%  1.13%  3.19%  2.16%  2.38%  1.01%
##    48:   1.87%  1.07%  3.14%  2.31%  2.28%  0.96%
##    49:   1.88%  0.98%  3.27%  2.31%  2.23%  1.06%
##    50:   1.82%  1.04%  3.31%  1.96%  2.23%  0.92%
##    51:   1.90%  1.10%  3.19%  2.11%  2.54%  1.01%
##    52:   1.89%  1.01%  3.36%  2.16%  2.49%  0.87%
##    53:   1.86%  1.01%  3.31%  2.01%  2.49%  0.92%
##    54:   1.83%  1.07%  3.23%  1.91%  2.38%  0.92%
##    55:   1.76%  1.01%  3.06%  1.91%  2.33%  0.87%
##    56:   1.86%  1.07%  3.23%  2.11%  2.38%  0.92%
##    57:   1.77%  0.95%  3.10%  2.11%  2.23%  0.92%
##    58:   1.73%  0.98%  2.88%  2.11%  2.28%  0.83%
##    59:   1.77%  0.95%  3.23%  2.06%  2.23%  0.83%
##    60:   1.83%  0.98%  3.19%  2.26%  2.28%  0.92%
##    61:   1.80%  1.01%  3.10%  2.16%  2.28%  0.87%
##    62:   1.81%  0.98%  2.93%  2.26%  2.38%  0.96%
##    63:   1.80%  0.98%  3.01%  2.11%  2.43%  0.92%
##    64:   1.79%  0.95%  3.06%  2.06%  2.43%  0.92%
##    65:   1.77%  1.04%  2.93%  1.96%  2.33%  0.96%
##    66:   1.80%  1.07%  2.88%  2.16%  2.33%  0.96%
##    67:   1.77%  0.98%  2.80%  2.16%  2.38%  0.96%
##    68:   1.80%  0.92%  3.06%  2.06%  2.38%  1.06%
##    69:   1.81%  0.95%  3.14%  2.06%  2.38%  0.96%
##    70:   1.80%  0.98%  3.10%  1.96%  2.43%  0.96%
##    71:   1.79%  0.98%  3.14%  1.91%  2.49%  0.87%
##    72:   1.81%  1.01%  3.23%  1.91%  2.33%  0.96%
##    73:   1.73%  0.92%  3.10%  1.96%  2.23%  0.87%
##    74:   1.76%  1.01%  3.06%  1.96%  2.28%  0.87%
##    75:   1.79%  0.95%  3.19%  1.96%  2.28%  1.01%
##    76:   1.81%  0.98%  3.19%  2.01%  2.38%  0.92%
##    77:   1.80%  0.95%  3.14%  2.11%  2.33%  0.92%
##    78:   1.78%  0.92%  3.10%  2.06%  2.38%  0.92%
##    79:   1.76%  0.89%  3.06%  2.11%  2.33%  0.87%
##    80:   1.79%  0.89%  3.10%  2.06%  2.33%  1.06%
##    81:   1.81%  0.89%  3.06%  2.16%  2.49%  0.96%
##    82:   1.74%  0.89%  3.06%  1.96%  2.38%  0.87%
##    83:   1.74%  0.83%  3.01%  1.96%  2.38%  1.01%
##    84:   1.73%  0.83%  3.06%  2.06%  2.28%  0.92%
##    85:   1.77%  0.86%  3.01%  2.11%  2.38%  0.96%
##    86:   1.74%  0.86%  2.97%  2.06%  2.28%  1.01%
##    87:   1.74%  0.83%  2.88%  2.11%  2.43%  0.96%
##    88:   1.77%  0.92%  2.97%  2.01%  2.38%  1.01%
##    89:   1.72%  0.89%  2.93%  1.91%  2.33%  0.96%
##    90:   1.79%  0.95%  3.10%  1.91%  2.54%  0.92%
##    91:   1.77%  0.92%  3.01%  1.96%  2.49%  0.92%
##    92:   1.81%  0.98%  3.01%  1.96%  2.49%  1.06%
##    93:   1.80%  0.98%  3.01%  1.91%  2.49%  1.06%
##    94:   1.77%  1.01%  2.97%  1.96%  2.43%  0.92%
##    95:   1.77%  0.92%  2.97%  2.06%  2.43%  0.96%
##    96:   1.80%  0.95%  2.93%  1.96%  2.59%  1.06%
##    97:   1.77%  0.92%  2.93%  2.01%  2.43%  1.06%
##    98:   1.77%  0.92%  2.88%  1.96%  2.54%  1.06%
##    99:   1.77%  0.92%  2.97%  1.96%  2.43%  1.01%
##   100:   1.77%  0.95%  2.93%  1.96%  2.49%  1.01%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.69%  6.10% 11.85%  7.67%  8.21%  5.19%
##     2:   7.13%  5.41% 11.16%  7.11%  7.59%  5.17%
##     3:   7.13%  4.76% 10.19%  8.71%  8.04%  5.29%
##     4:   6.60%  4.68%  9.55%  7.45%  7.88%  4.49%
##     5:   6.53%  4.20% 10.33%  7.67%  7.79%  3.87%
##     6:   5.89%  3.48%  8.68%  7.40%  7.13%  4.15%
##     7:   5.65%  3.48%  8.68%  6.69%  7.13%  3.51%
##     8:   5.10%  3.37%  7.49%  6.43%  6.22%  3.01%
##     9:   4.65%  3.03%  6.88%  5.52%  5.84%  2.93%
##    10:   4.45%  3.13%  5.98%  5.83%  5.38%  2.78%
##    11:   4.11%  2.97%  5.71%  5.15%  4.81%  2.58%
##    12:   3.78%  2.45%  5.31%  5.09%  4.33%  2.49%
##    13:   3.54%  1.94%  5.39%  4.73%  4.26%  2.30%
##    14:   3.28%  1.82%  5.00%  4.18%  4.31%  1.93%
##    15:   3.07%  1.70%  4.69%  3.97%  3.79%  1.98%
##    16:   2.91%  1.85%  4.18%  3.57%  3.58%  1.98%
##    17:   2.71%  1.85%  4.09%  2.97%  3.37%  1.74%
##    18:   2.65%  1.67%  4.09%  3.17%  3.26%  1.61%
##    19:   2.68%  1.79%  4.00%  3.17%  3.06%  1.84%
##    20:   2.62%  1.70%  3.83%  3.12%  3.42%  1.61%
##    21:   2.54%  1.70%  3.40%  3.02%  3.47%  1.65%
##    22:   2.48%  1.52%  3.62%  2.86%  3.26%  1.70%
##    23:   2.43%  1.52%  3.57%  3.02%  3.26%  1.33%
##    24:   2.35%  1.49%  3.27%  2.96%  3.21%  1.38%
##    25:   2.25%  1.37%  3.06%  2.86%  3.47%  1.10%
##    26:   2.28%  1.43%  3.31%  2.61%  3.26%  1.29%
##    27:   2.28%  1.37%  3.44%  2.66%  3.16%  1.29%
##    28:   2.25%  1.25%  3.44%  2.66%  3.26%  1.24%
##    29:   2.22%  1.37%  3.27%  2.81%  3.11%  1.10%
##    30:   2.10%  1.28%  3.19%  2.46%  2.95%  1.10%
##    31:   2.08%  1.22%  3.10%  2.56%  2.74%  1.29%
##    32:   2.00%  1.19%  2.88%  2.41%  2.90%  1.10%
##    33:   2.00%  1.19%  3.06%  2.36%  2.74%  1.15%
##    34:   2.05%  1.19%  3.06%  2.31%  2.95%  1.24%
##    35:   1.89%  1.10%  3.10%  2.16%  2.49%  1.06%
##    36:   1.94%  1.10%  3.06%  2.26%  2.64%  1.10%
##    37:   1.98%  1.10%  3.14%  2.31%  2.74%  1.10%
##    38:   1.90%  1.13%  3.01%  2.11%  2.59%  1.10%
##    39:   1.87%  1.07%  2.93%  2.26%  2.38%  1.15%
##    40:   1.86%  1.16%  2.84%  2.21%  2.28%  1.19%
##    41:   1.91%  1.13%  3.01%  2.31%  2.64%  0.92%
##    42:   1.92%  1.13%  3.01%  2.36%  2.43%  1.10%
##    43:   1.81%  1.16%  2.80%  2.11%  2.28%  1.06%
##    44:   1.85%  1.28%  2.80%  1.96%  2.38%  1.15%
##    45:   1.94%  1.25%  2.93%  2.11%  2.69%  1.15%
##    46:   1.89%  1.25%  2.80%  2.16%  2.59%  1.01%
##    47:   1.83%  1.13%  2.76%  2.16%  2.49%  1.01%
##    48:   1.87%  1.16%  2.80%  2.21%  2.59%  1.01%
##    49:   1.83%  1.22%  2.80%  2.16%  2.49%  0.87%
##    50:   1.86%  1.16%  2.84%  2.21%  2.54%  0.96%
##    51:   1.89%  1.13%  2.97%  2.26%  2.59%  0.92%
##    52:   1.87%  1.16%  2.93%  2.11%  2.59%  0.96%
##    53:   1.81%  1.04%  2.97%  2.11%  2.43%  0.92%
##    54:   1.79%  1.01%  2.84%  2.06%  2.49%  1.01%
##    55:   1.81%  1.07%  2.84%  2.11%  2.54%  0.92%
##    56:   1.77%  0.98%  2.80%  2.11%  2.54%  0.92%
##    57:   1.75%  0.95%  2.80%  2.16%  2.38%  0.92%
##    58:   1.72%  0.95%  2.76%  1.81%  2.54%  1.01%
##    59:   1.72%  1.01%  2.71%  1.86%  2.64%  0.78%
##    60:   1.72%  0.95%  2.84%  1.71%  2.54%  0.96%
##    61:   1.74%  0.98%  2.93%  1.81%  2.54%  0.87%
##    62:   1.74%  0.95%  2.88%  1.96%  2.59%  0.78%
##    63:   1.77%  0.98%  2.97%  1.91%  2.59%  0.83%
##    64:   1.78%  1.04%  2.88%  2.01%  2.49%  0.92%
##    65:   1.77%  1.01%  2.97%  1.96%  2.54%  0.78%
##    66:   1.76%  1.04%  3.06%  1.81%  2.49%  0.78%
##    67:   1.72%  1.04%  3.01%  1.76%  2.33%  0.78%
##    68:   1.78%  1.04%  3.06%  2.06%  2.38%  0.78%
##    69:   1.73%  1.01%  3.01%  1.81%  2.38%  0.83%
##    70:   1.77%  1.01%  3.06%  1.86%  2.54%  0.78%
##    71:   1.71%  1.04%  2.88%  1.76%  2.49%  0.73%
##    72:   1.72%  1.01%  2.93%  1.86%  2.49%  0.69%
##    73:   1.71%  1.04%  3.10%  1.71%  2.18%  0.83%
##    74:   1.72%  0.98%  2.93%  1.76%  2.43%  0.92%
##    75:   1.70%  0.95%  2.93%  1.86%  2.33%  0.83%
##    76:   1.72%  0.98%  2.97%  1.81%  2.38%  0.83%
##    77:   1.73%  0.95%  2.97%  1.86%  2.43%  0.87%
##    78:   1.73%  1.01%  2.97%  1.86%  2.28%  0.92%
##    79:   1.72%  1.01%  2.93%  1.86%  2.43%  0.78%
##    80:   1.74%  0.98%  2.97%  1.96%  2.28%  0.92%
##    81:   1.70%  1.04%  2.84%  1.76%  2.33%  0.87%
##    82:   1.74%  1.04%  2.84%  1.96%  2.38%  0.87%
##    83:   1.70%  1.01%  2.88%  1.71%  2.38%  0.87%
##    84:   1.70%  0.98%  2.88%  1.91%  2.33%  0.78%
##    85:   1.72%  0.95%  2.97%  2.01%  2.28%  0.78%
##    86:   1.77%  0.95%  3.06%  2.11%  2.38%  0.83%
##    87:   1.76%  0.95%  3.01%  2.06%  2.28%  0.92%
##    88:   1.75%  0.95%  3.06%  1.96%  2.23%  0.96%
##    89:   1.73%  0.98%  3.10%  1.96%  2.12%  0.87%
##    90:   1.72%  0.98%  2.93%  2.06%  2.28%  0.73%
##    91:   1.72%  1.01%  3.01%  1.86%  2.23%  0.83%
##    92:   1.74%  1.04%  3.06%  1.86%  2.28%  0.83%
##    93:   1.70%  1.04%  2.93%  1.86%  2.18%  0.83%
##    94:   1.69%  1.01%  2.80%  1.96%  2.23%  0.83%
##    95:   1.66%  0.98%  2.84%  1.86%  2.18%  0.83%
##    96:   1.67%  1.04%  2.76%  1.86%  2.23%  0.83%
##    97:   1.67%  1.01%  2.80%  1.86%  2.23%  0.83%
##    98:   1.66%  0.98%  2.88%  1.86%  2.07%  0.87%
##    99:   1.70%  1.04%  2.84%  1.96%  2.18%  0.83%
##   100:   1.69%  1.04%  2.84%  1.86%  2.18%  0.87%
## ntree      OOB      1      2      3      4      5
##     1:  10.06%  5.56% 14.06% 13.43% 11.39%  8.89%
##     2:   9.84%  6.11% 13.73% 13.08% 11.42%  7.35%
##     3:   9.94%  6.68% 13.61% 12.51% 11.70%  7.15%
##     4:   9.59%  5.95% 12.42% 13.69% 11.71%  6.50%
##     5:   8.68%  6.06% 11.25% 11.27%  9.93%  6.45%
##     6:   7.57%  4.77% 10.78%  9.53%  8.95%  5.43%
##     7:   6.74%  4.21%  9.23%  8.45%  8.06%  5.25%
##     8:   6.03%  3.82%  8.36%  7.71%  6.94%  4.66%
##     9:   5.73%  3.75%  8.01%  7.46%  6.35%  4.21%
##    10:   5.26%  3.37%  7.13%  6.86%  5.96%  4.09%
##    11:   4.56%  2.85%  6.39%  6.35%  5.54%  2.71%
##    12:   4.18%  2.78%  5.34%  5.75%  5.28%  2.62%
##    13:   3.95%  2.63%  5.34%  5.70%  4.66%  2.24%
##    14:   3.64%  2.21%  5.60%  5.06%  4.25%  1.91%
##    15:   3.44%  2.09%  4.88%  4.52%  4.40%  2.14%
##    16:   3.14%  2.00%  4.52%  4.13%  3.84%  1.91%
##    17:   2.79%  1.79%  4.16%  3.45%  3.49%  1.63%
##    18:   2.79%  2.00%  3.89%  3.99%  3.09%  1.44%
##    19:   2.58%  1.82%  3.31%  3.50%  3.29%  1.49%
##    20:   2.53%  1.73%  3.62%  3.31%  3.03%  1.44%
##    21:   2.46%  1.55%  3.85%  2.87%  3.13%  1.44%
##    22:   2.22%  1.31%  3.22%  2.97%  2.68%  1.44%
##    23:   2.20%  1.28%  3.53%  2.82%  2.83%  1.07%
##    24:   2.17%  1.34%  3.40%  2.58%  2.88%  1.16%
##    25:   2.11%  1.13%  3.40%  2.68%  2.53%  1.35%
##    26:   2.02%  1.37%  3.22%  2.38%  2.38%  1.12%
##    27:   1.94%  1.04%  3.44%  2.33%  2.33%  1.02%
##    28:   1.94%  1.22%  3.18%  2.33%  2.28%  1.07%
##    29:   1.96%  1.16%  3.18%  2.33%  2.53%  1.07%
##    30:   1.93%  1.01%  3.09%  2.04%  2.58%  1.44%
##    31:   1.87%  0.95%  3.22%  2.09%  2.28%  1.30%
##    32:   1.83%  1.10%  2.95%  1.99%  2.22%  1.26%
##    33:   1.84%  1.10%  2.95%  2.04%  2.33%  1.21%
##    34:   1.83%  1.01%  2.91%  2.24%  2.22%  1.26%
##    35:   1.79%  1.10%  3.04%  1.80%  2.22%  1.16%
##    36:   1.70%  1.01%  2.91%  1.85%  2.07%  1.02%
##    37:   1.77%  1.13%  2.91%  1.85%  2.17%  1.12%
##    38:   1.73%  1.07%  2.95%  1.65%  2.22%  1.12%
##    39:   1.65%  0.95%  2.82%  1.80%  2.12%  0.93%
##    40:   1.69%  0.95%  3.04%  1.65%  2.17%  1.02%
##    41:   1.65%  0.98%  2.82%  1.65%  2.22%  0.93%
##    42:   1.69%  0.95%  2.91%  1.95%  2.17%  0.88%
##    43:   1.63%  0.98%  2.91%  1.46%  2.22%  0.93%
##    44:   1.66%  0.92%  2.86%  1.70%  2.28%  0.98%
##    45:   1.67%  0.95%  2.95%  1.46%  2.38%  1.02%
##    46:   1.65%  1.01%  2.95%  1.56%  2.22%  0.84%
##    47:   1.66%  1.01%  2.86%  1.36%  2.28%  1.16%
##    48:   1.60%  0.98%  2.68%  1.46%  2.17%  1.02%
##    49:   1.55%  0.89%  2.55%  1.51%  2.22%  0.98%
##    50:   1.56%  0.92%  2.68%  1.36%  2.28%  0.93%
##    51:   1.53%  0.86%  2.50%  1.61%  2.07%  0.98%
##    52:   1.50%  0.83%  2.46%  1.56%  2.07%  0.98%
##    53:   1.53%  0.89%  2.42%  1.56%  2.07%  1.07%
##    54:   1.59%  0.89%  2.55%  1.80%  2.07%  1.02%
##    55:   1.58%  0.86%  2.68%  1.75%  2.07%  0.93%
##    56:   1.59%  0.80%  2.77%  1.65%  2.17%  0.98%
##    57:   1.55%  0.86%  2.73%  1.61%  2.02%  0.93%
##    58:   1.55%  0.80%  2.73%  1.51%  2.02%  1.12%
##    59:   1.50%  0.86%  2.64%  1.61%  1.72%  1.02%
##    60:   1.50%  0.89%  2.64%  1.61%  1.72%  0.98%
##    61:   1.44%  0.92%  2.42%  1.46%  1.67%  0.98%
##    62:   1.43%  0.80%  2.37%  1.56%  1.77%  0.98%
##    63:   1.47%  0.98%  2.37%  1.56%  1.72%  0.98%
##    64:   1.42%  0.92%  2.19%  1.56%  1.72%  0.98%
##    65:   1.44%  0.86%  2.33%  1.56%  1.77%  0.98%
##    66:   1.42%  0.80%  2.46%  1.41%  1.72%  1.02%
##    67:   1.48%  0.92%  2.55%  1.51%  1.82%  0.88%
##    68:   1.44%  0.86%  2.59%  1.41%  1.77%  0.84%
##    69:   1.42%  0.92%  2.46%  1.31%  1.77%  0.88%
##    70:   1.40%  0.74%  2.59%  1.36%  1.77%  0.88%
##    71:   1.45%  0.77%  2.64%  1.36%  1.82%  1.02%
##    72:   1.42%  0.83%  2.50%  1.36%  1.77%  0.93%
##    73:   1.43%  0.89%  2.42%  1.46%  1.77%  0.88%
##    74:   1.36%  0.77%  2.46%  1.26%  1.77%  0.84%
##    75:   1.39%  0.80%  2.37%  1.41%  1.82%  0.88%
##    76:   1.38%  0.83%  2.33%  1.31%  1.82%  0.88%
##    77:   1.39%  0.83%  2.37%  1.46%  1.72%  0.88%
##    78:   1.40%  0.86%  2.42%  1.26%  1.82%  0.93%
##    79:   1.42%  0.77%  2.55%  1.31%  1.87%  0.93%
##    80:   1.35%  0.69%  2.28%  1.51%  1.87%  0.79%
##    81:   1.39%  0.74%  2.37%  1.41%  2.07%  0.74%
##    82:   1.37%  0.69%  2.42%  1.36%  1.97%  0.79%
##    83:   1.35%  0.74%  2.24%  1.36%  2.07%  0.70%
##    84:   1.36%  0.69%  2.46%  1.36%  1.97%  0.70%
##    85:   1.36%  0.74%  2.19%  1.51%  1.92%  0.79%
##    86:   1.35%  0.69%  2.33%  1.41%  1.97%  0.74%
##    87:   1.42%  0.69%  2.33%  1.61%  2.12%  0.79%
##    88:   1.34%  0.66%  2.06%  1.56%  2.07%  0.79%
##    89:   1.38%  0.72%  2.19%  1.51%  2.07%  0.84%
##    90:   1.36%  0.72%  2.15%  1.56%  2.02%  0.74%
##    91:   1.33%  0.74%  2.10%  1.46%  2.02%  0.70%
##    92:   1.33%  0.72%  2.06%  1.41%  2.07%  0.79%
##    93:   1.33%  0.72%  2.10%  1.51%  1.92%  0.79%
##    94:   1.32%  0.69%  2.24%  1.36%  1.97%  0.74%
##    95:   1.28%  0.72%  2.06%  1.41%  1.97%  0.60%
##    96:   1.29%  0.69%  2.15%  1.46%  1.92%  0.60%
##    97:   1.27%  0.66%  2.06%  1.46%  1.87%  0.65%
##    98:   1.23%  0.69%  1.92%  1.31%  1.92%  0.65%
##    99:   1.27%  0.63%  2.01%  1.41%  2.02%  0.65%
##   100:   1.25%  0.60%  2.10%  1.41%  1.92%  0.60%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.60%  3.43% 11.81% 11.24%  7.80%  5.93%
##     2:   7.68%  5.10% 10.53% 10.82%  7.97%  5.57%
##     3:   7.43%  4.88% 10.22% 10.23%  7.82%  5.52%
##     4:   6.99%  4.50%  9.95%  9.37%  7.85%  4.76%
##     5:   6.29%  3.92%  9.11%  8.49%  6.93%  4.37%
##     6:   6.12%  4.04%  9.14%  8.36%  6.57%  3.70%
##     7:   5.70%  3.38%  8.53%  7.40%  6.24%  4.26%
##     8:   5.22%  3.00%  7.70%  7.16%  5.44%  4.05%
##     9:   4.79%  2.85%  6.64%  6.37%  5.63%  3.64%
##    10:   4.62%  2.59%  6.64%  6.62%  5.17%  3.29%
##    11:   4.40%  2.67%  6.16%  5.86%  4.59%  3.70%
##    12:   3.89%  1.98%  5.79%  5.17%  4.33%  3.27%
##    13:   3.65%  1.79%  5.47%  5.46%  3.81%  2.80%
##    14:   3.40%  1.82%  4.97%  5.11%  3.25%  2.70%
##    15:   3.35%  1.85%  5.10%  5.07%  3.29%  2.28%
##    16:   3.28%  1.64%  4.79%  4.92%  3.34%  2.65%
##    17:   3.06%  1.73%  4.38%  4.43%  3.14%  2.37%
##    18:   3.13%  1.88%  4.65%  4.77%  2.88%  2.19%
##    19:   2.85%  1.73%  4.11%  3.89%  2.93%  2.23%
##    20:   2.71%  1.49%  4.16%  3.55%  2.88%  2.14%
##    21:   2.78%  1.43%  4.43%  3.79%  2.88%  2.09%
##    22:   2.70%  1.46%  3.94%  3.75%  3.03%  2.05%
##    23:   2.64%  1.58%  3.85%  3.70%  2.83%  1.86%
##    24:   2.49%  1.46%  3.71%  3.70%  2.53%  1.63%
##    25:   2.39%  1.52%  3.53%  3.26%  2.48%  1.67%
##    26:   2.43%  1.43%  3.31%  3.45%  2.73%  1.81%
##    27:   2.43%  1.37%  3.44%  3.65%  2.53%  1.77%
##    28:   2.36%  1.37%  3.26%  3.50%  2.53%  1.72%
##    29:   2.31%  1.37%  3.22%  3.26%  2.58%  1.67%
##    30:   2.28%  1.25%  3.22%  3.11%  2.83%  1.63%
##    31:   2.14%  1.28%  2.91%  2.97%  2.53%  1.53%
##    32:   2.23%  1.31%  3.22%  3.02%  2.63%  1.53%
##    33:   2.17%  1.28%  3.18%  3.02%  2.38%  1.53%
##    34:   2.18%  1.25%  2.91%  3.11%  2.63%  1.58%
##    35:   2.13%  1.28%  2.77%  3.06%  2.48%  1.58%
##    36:   2.09%  1.28%  2.82%  2.97%  2.53%  1.35%
##    37:   2.06%  1.28%  2.77%  2.92%  2.38%  1.44%
##    38:   2.11%  1.28%  2.91%  2.77%  2.48%  1.63%
##    39:   2.06%  1.25%  2.77%  2.72%  2.53%  1.49%
##    40:   1.99%  1.19%  2.64%  2.48%  2.43%  1.67%
##    41:   2.02%  1.19%  2.64%  2.63%  2.53%  1.63%
##    42:   2.05%  1.25%  2.73%  2.68%  2.48%  1.58%
##    43:   2.03%  1.28%  2.68%  2.68%  2.33%  1.63%
##    44:   2.05%  1.34%  2.73%  2.72%  2.33%  1.53%
##    45:   2.00%  1.22%  2.64%  2.82%  2.43%  1.35%
##    46:   2.05%  1.28%  2.64%  2.77%  2.53%  1.49%
##    47:   2.00%  1.19%  2.64%  2.58%  2.63%  1.49%
##    48:   2.05%  1.22%  2.86%  2.68%  2.48%  1.49%
##    49:   1.99%  1.22%  2.77%  2.48%  2.53%  1.40%
##    50:   2.04%  1.22%  2.68%  2.68%  2.58%  1.53%
##    51:   2.00%  1.16%  2.73%  2.53%  2.58%  1.53%
##    52:   2.05%  1.16%  2.73%  2.72%  2.63%  1.53%
##    53:   1.99%  1.10%  2.68%  2.68%  2.53%  1.49%
##    54:   2.06%  1.25%  2.68%  2.82%  2.63%  1.44%
##    55:   2.04%  1.16%  2.73%  2.77%  2.58%  1.49%
##    56:   2.08%  1.16%  2.86%  2.82%  2.63%  1.49%
##    57:   2.08%  1.16%  2.82%  2.63%  2.83%  1.53%
##    58:   2.04%  1.10%  2.91%  2.58%  2.73%  1.44%
##    59:   2.07%  1.07%  2.82%  2.82%  2.78%  1.49%
##    60:   2.01%  1.07%  2.86%  2.72%  2.58%  1.40%
##    61:   1.94%  1.07%  2.82%  2.38%  2.58%  1.40%
##    62:   1.94%  1.04%  2.68%  2.48%  2.63%  1.40%
##    63:   1.95%  1.10%  2.82%  2.43%  2.58%  1.35%
##    64:   1.91%  1.04%  2.73%  2.38%  2.53%  1.40%
##    65:   1.91%  1.04%  2.64%  2.38%  2.53%  1.49%
##    66:   1.89%  1.01%  2.64%  2.43%  2.43%  1.44%
##    67:   1.90%  0.95%  2.59%  2.58%  2.53%  1.44%
##    68:   1.92%  0.95%  2.73%  2.58%  2.48%  1.44%
##    69:   1.96%  1.01%  2.77%  2.63%  2.63%  1.35%
##    70:   1.92%  1.04%  2.73%  2.48%  2.48%  1.40%
##    71:   1.91%  1.07%  2.77%  2.29%  2.53%  1.40%
##    72:   1.87%  1.04%  2.64%  2.24%  2.53%  1.40%
##    73:   1.94%  1.10%  2.77%  2.33%  2.53%  1.44%
##    74:   1.86%  0.98%  2.77%  2.33%  2.43%  1.30%
##    75:   1.84%  0.98%  2.68%  2.29%  2.38%  1.40%
##    76:   1.83%  0.86%  2.73%  2.33%  2.43%  1.35%
##    77:   1.84%  0.89%  2.68%  2.38%  2.38%  1.44%
##    78:   1.85%  0.86%  2.77%  2.33%  2.38%  1.49%
##    79:   1.83%  0.86%  2.59%  2.33%  2.53%  1.44%
##    80:   1.89%  0.89%  2.68%  2.48%  2.63%  1.40%
##    81:   1.87%  0.89%  2.64%  2.38%  2.58%  1.44%
##    82:   1.82%  0.89%  2.64%  2.29%  2.48%  1.35%
##    83:   1.86%  0.95%  2.64%  2.38%  2.48%  1.40%
##    84:   1.85%  0.95%  2.64%  2.38%  2.48%  1.35%
##    85:   1.84%  0.92%  2.64%  2.43%  2.48%  1.30%
##    86:   1.83%  0.89%  2.59%  2.38%  2.48%  1.35%
##    87:   1.85%  0.95%  2.64%  2.38%  2.48%  1.35%
##    88:   1.87%  0.86%  2.55%  2.48%  2.63%  1.44%
##    89:   1.88%  0.89%  2.59%  2.53%  2.63%  1.35%
##    90:   1.83%  0.83%  2.46%  2.58%  2.58%  1.30%
##    91:   1.83%  0.89%  2.46%  2.48%  2.58%  1.35%
##    92:   1.82%  0.89%  2.42%  2.43%  2.58%  1.35%
##    93:   1.79%  0.83%  2.46%  2.48%  2.43%  1.35%
##    94:   1.81%  0.89%  2.46%  2.43%  2.48%  1.35%
##    95:   1.83%  0.89%  2.46%  2.53%  2.53%  1.35%
##    96:   1.83%  0.89%  2.46%  2.48%  2.53%  1.35%
##    97:   1.80%  0.86%  2.42%  2.33%  2.63%  1.35%
##    98:   1.81%  0.86%  2.46%  2.43%  2.53%  1.35%
##    99:   1.79%  0.89%  2.42%  2.29%  2.58%  1.35%
##   100:   1.77%  0.89%  2.33%  2.38%  2.53%  1.30%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   6.91%  3.58%  9.98%  9.95%  8.94%  4.10%
##     2:   7.06%  2.98%  9.60% 10.47%  9.22%  5.44%
##     3:   6.96%  3.51%  9.55% 10.34%  8.37%  5.07%
##     4:   6.49%  3.82%  8.70%  9.06%  7.52%  4.91%
##     5:   6.23%  3.46%  8.99%  7.98%  7.46%  4.83%
##     6:   5.79%  2.76%  8.16%  8.00%  7.48%  4.30%
##     7:   5.54%  3.17%  7.56%  7.02%  7.11%  4.22%
##     8:   5.30%  2.79%  8.23%  6.35%  6.54%  4.03%
##     9:   4.89%  2.76%  7.01%  5.96%  6.27%  3.70%
##    10:   4.49%  2.32%  6.29%  5.98%  5.71%  3.44%
##    11:   4.17%  2.16%  5.92%  5.41%  5.44%  3.14%
##    12:   3.81%  2.36%  5.51%  4.85%  4.52%  2.67%
##    13:   3.66%  2.18%  5.10%  5.08%  4.41%  2.43%
##    14:   3.38%  2.00%  4.74%  4.58%  4.45%  2.00%
##    15:   3.27%  1.76%  4.52%  4.58%  4.20%  2.23%
##    16:   3.02%  1.94%  4.25%  3.99%  3.64%  1.91%
##    17:   3.07%  1.73%  4.20%  4.24%  3.79%  2.19%
##    18:   2.88%  1.85%  3.67%  4.04%  3.44%  2.05%
##    19:   2.88%  1.58%  3.76%  4.23%  3.69%  1.95%
##    20:   2.79%  1.55%  3.62%  3.79%  3.74%  2.00%
##    21:   2.60%  1.58%  3.35%  3.40%  3.54%  1.77%
##    22:   2.67%  1.58%  3.58%  3.55%  3.54%  1.81%
##    23:   2.73%  1.73%  3.53%  3.45%  3.74%  1.81%
##    24:   2.57%  1.61%  3.44%  3.21%  3.54%  1.67%
##    25:   2.62%  1.49%  3.22%  3.31%  4.15%  1.72%
##    26:   2.62%  1.61%  3.49%  3.31%  3.64%  1.72%
##    27:   2.56%  1.46%  3.13%  3.50%  3.89%  1.58%
##    28:   2.47%  1.43%  3.22%  3.31%  3.44%  1.63%
##    29:   2.46%  1.43%  3.18%  3.40%  3.34%  1.63%
##    30:   2.44%  1.34%  3.18%  3.36%  3.29%  1.72%
##    31:   2.48%  1.46%  3.26%  3.26%  3.44%  1.63%
##    32:   2.43%  1.49%  3.44%  3.02%  3.29%  1.49%
##    33:   2.37%  1.34%  3.40%  3.02%  3.19%  1.53%
##    34:   2.39%  1.40%  3.31%  2.92%  3.34%  1.58%
##    35:   2.35%  1.46%  3.40%  2.87%  3.08%  1.49%
##    36:   2.35%  1.37%  3.31%  2.92%  3.13%  1.63%
##    37:   2.28%  1.28%  3.13%  2.97%  3.08%  1.53%
##    38:   2.28%  1.31%  3.13%  2.82%  3.08%  1.63%
##    39:   2.23%  1.34%  3.18%  2.77%  2.93%  1.49%
##    40:   2.11%  1.19%  3.04%  2.68%  2.73%  1.49%
##    41:   2.08%  1.16%  2.91%  2.68%  2.83%  1.40%
##    42:   2.00%  1.22%  2.73%  2.53%  2.68%  1.35%
##    43:   2.04%  1.10%  2.95%  2.63%  2.73%  1.35%
##    44:   2.01%  1.19%  2.77%  2.58%  2.78%  1.26%
##    45:   2.07%  1.10%  3.00%  2.68%  2.83%  1.35%
##    46:   2.07%  1.19%  2.95%  2.72%  2.83%  1.21%
##    47:   2.07%  1.19%  3.04%  2.77%  2.73%  1.16%
##    48:   2.11%  1.16%  3.04%  2.77%  2.93%  1.26%
##    49:   2.05%  1.16%  2.82%  2.63%  2.88%  1.30%
##    50:   2.00%  1.13%  2.73%  2.58%  2.83%  1.26%
##    51:   2.00%  1.16%  2.91%  2.38%  2.78%  1.30%
##    52:   1.97%  1.13%  2.82%  2.43%  2.83%  1.16%
##    53:   2.04%  1.04%  3.13%  2.58%  2.88%  1.16%
##    54:   1.95%  1.04%  2.82%  2.63%  2.73%  1.12%
##    55:   1.93%  1.01%  2.82%  2.53%  2.68%  1.16%
##    56:   1.97%  1.04%  2.95%  2.53%  2.68%  1.21%
##    57:   2.00%  1.01%  3.00%  2.68%  2.63%  1.26%
##    58:   1.94%  0.98%  2.91%  2.48%  2.63%  1.26%
##    59:   1.95%  1.01%  2.91%  2.53%  2.63%  1.26%
##    60:   1.90%  1.01%  2.77%  2.43%  2.58%  1.26%
##    61:   1.89%  0.92%  2.73%  2.48%  2.68%  1.26%
##    62:   1.86%  0.92%  2.82%  2.43%  2.43%  1.26%
##    63:   1.83%  0.92%  2.68%  2.43%  2.43%  1.26%
##    64:   1.82%  0.92%  2.59%  2.38%  2.53%  1.21%
##    65:   1.84%  0.92%  2.77%  2.43%  2.38%  1.26%
##    66:   1.83%  0.95%  2.82%  2.43%  2.33%  1.16%
##    67:   1.78%  0.86%  2.64%  2.48%  2.28%  1.21%
##    68:   1.89%  1.07%  2.82%  2.58%  2.22%  1.21%
##    69:   1.89%  1.10%  2.77%  2.48%  2.38%  1.16%
##    70:   1.80%  0.95%  2.73%  2.43%  2.17%  1.21%
##    71:   1.86%  1.07%  2.77%  2.48%  2.28%  1.16%
##    72:   1.86%  1.04%  2.77%  2.53%  2.33%  1.12%
##    73:   1.85%  1.10%  2.59%  2.58%  2.22%  1.21%
##    74:   1.76%  0.92%  2.55%  2.38%  2.33%  1.12%
##    75:   1.79%  0.95%  2.68%  2.38%  2.33%  1.12%
##    76:   1.81%  0.95%  2.73%  2.48%  2.22%  1.16%
##    77:   1.76%  0.83%  2.50%  2.48%  2.28%  1.26%
##    78:   1.81%  0.89%  2.59%  2.48%  2.38%  1.26%
##    79:   1.83%  0.95%  2.68%  2.38%  2.43%  1.21%
##    80:   1.84%  0.98%  2.73%  2.43%  2.33%  1.26%
##    81:   1.79%  0.95%  2.68%  2.33%  2.22%  1.26%
##    82:   1.81%  1.01%  2.55%  2.33%  2.38%  1.26%
##    83:   1.77%  0.95%  2.68%  2.24%  2.33%  1.12%
##    84:   1.78%  0.98%  2.73%  2.33%  2.22%  1.12%
##    85:   1.75%  1.01%  2.68%  2.19%  2.12%  1.16%
##    86:   1.81%  1.10%  2.73%  2.14%  2.33%  1.16%
##    87:   1.76%  1.01%  2.82%  2.09%  2.12%  1.16%
##    88:   1.77%  1.13%  2.73%  2.09%  2.07%  1.16%
##    89:   1.73%  1.01%  2.68%  2.09%  2.12%  1.16%
##    90:   1.72%  1.01%  2.68%  2.04%  2.12%  1.16%
##    91:   1.71%  0.98%  2.55%  2.04%  2.28%  1.12%
##    92:   1.75%  1.04%  2.50%  2.09%  2.43%  1.12%
##    93:   1.70%  1.04%  2.50%  2.09%  2.17%  1.07%
##    94:   1.68%  1.04%  2.46%  2.09%  2.12%  1.07%
##    95:   1.71%  1.13%  2.50%  2.09%  2.07%  1.07%
##    96:   1.70%  1.07%  2.50%  2.09%  2.17%  1.02%
##    97:   1.72%  1.01%  2.59%  2.19%  2.17%  1.02%
##    98:   1.72%  1.01%  2.68%  2.14%  2.12%  1.07%
##    99:   1.68%  1.07%  2.59%  2.04%  2.02%  1.02%
##   100:   1.66%  1.04%  2.55%  1.99%  1.97%  1.07%
## ntree      OOB      1      2      3      4      5
##     1:   9.61%  5.92% 12.01% 12.97% 10.79%  8.30%
##     2:   9.74%  6.25% 11.95% 12.85%  9.75%  9.73%
##     3:   9.59%  5.87% 12.24% 11.36% 10.27% 10.23%
##     4:   8.85%  5.68% 11.35% 11.00%  9.16%  8.71%
##     5:   8.63%  5.62% 11.73% 10.22%  9.17%  7.94%
##     6:   7.79%  5.29% 10.56%  9.19%  8.10%  7.07%
##     7:   7.21%  4.81%  9.90%  8.23%  7.43%  6.82%
##     8:   6.50%  4.42%  9.12%  7.80%  6.61%  5.51%
##     9:   5.89%  4.35%  8.12%  6.93%  5.89%  4.80%
##    10:   5.49%  4.15%  7.58%  6.50%  5.15%  4.62%
##    11:   4.88%  3.74%  6.97%  5.43%  4.73%  3.97%
##    12:   4.62%  3.25%  7.16%  5.66%  4.67%  2.86%
##    13:   4.17%  3.04%  5.95%  4.87%  4.56%  2.91%
##    14:   3.97%  2.92%  5.56%  4.44%  4.35%  3.05%
##    15:   3.71%  2.86%  5.04%  4.24%  4.39%  2.36%
##    16:   3.53%  2.86%  5.04%  3.61%  3.99%  2.36%
##    17:   3.36%  2.97%  4.36%  3.66%  3.78%  2.16%
##    18:   3.04%  2.53%  4.66%  3.33%  3.17%  1.62%
##    19:   2.78%  2.23%  3.93%  2.99%  3.07%  1.87%
##    20:   2.78%  2.20%  3.89%  3.18%  3.17%  1.67%
##    21:   2.63%  2.38%  3.59%  2.94%  2.76%  1.52%
##    22:   2.55%  2.41%  3.42%  2.51%  2.71%  1.67%
##    23:   2.27%  2.02%  3.03%  2.46%  2.45%  1.42%
##    24:   2.33%  1.87%  3.50%  2.46%  2.60%  1.33%
##    25:   2.19%  1.93%  3.08%  2.27%  2.65%  1.08%
##    26:   2.14%  1.93%  2.99%  2.27%  2.50%  1.03%
##    27:   2.04%  1.63%  3.12%  2.17%  2.30%  1.08%
##    28:   2.03%  1.75%  2.73%  2.02%  2.45%  1.28%
##    29:   2.07%  1.66%  3.20%  2.07%  2.25%  1.28%
##    30:   1.97%  1.63%  2.82%  2.02%  2.19%  1.28%
##    31:   2.02%  1.60%  2.95%  2.17%  2.40%  1.13%
##    32:   2.08%  1.72%  3.03%  2.17%  2.45%  1.13%
##    33:   2.02%  1.72%  2.95%  2.07%  2.30%  1.13%
##    34:   1.95%  1.69%  3.16%  1.83%  2.09%  0.98%
##    35:   1.84%  1.60%  2.65%  1.98%  2.04%  0.98%
##    36:   1.86%  1.63%  2.73%  1.98%  1.99%  0.98%
##    37:   1.87%  1.69%  2.69%  1.93%  2.04%  0.98%
##    38:   1.85%  1.58%  2.65%  2.02%  2.04%  1.03%
##    39:   1.84%  1.58%  2.65%  2.02%  2.04%  0.98%
##    40:   1.81%  1.63%  2.39%  2.12%  2.04%  0.88%
##    41:   1.76%  1.55%  2.39%  1.88%  2.14%  0.88%
##    42:   1.81%  1.63%  2.56%  1.88%  2.09%  0.88%
##    43:   1.77%  1.58%  2.39%  1.88%  1.99%  1.03%
##    44:   1.76%  1.52%  2.35%  2.02%  1.94%  1.03%
##    45:   1.72%  1.55%  2.22%  1.78%  2.19%  0.93%
##    46:   1.71%  1.52%  2.39%  1.69%  1.99%  0.98%
##    47:   1.74%  1.58%  2.31%  1.93%  1.94%  0.98%
##    48:   1.72%  1.52%  2.35%  1.78%  1.99%  1.03%
##    49:   1.65%  1.43%  2.31%  1.78%  1.94%  0.83%
##    50:   1.60%  1.31%  2.26%  1.69%  1.89%  0.93%
##    51:   1.68%  1.37%  2.48%  1.83%  1.89%  0.93%
##    52:   1.66%  1.37%  2.43%  1.83%  1.79%  0.98%
##    53:   1.67%  1.43%  2.43%  1.78%  1.84%  0.93%
##    54:   1.66%  1.40%  2.43%  1.73%  1.74%  1.03%
##    55:   1.63%  1.40%  2.43%  1.64%  1.74%  0.98%
##    56:   1.65%  1.40%  2.56%  1.69%  1.68%  0.93%
##    57:   1.60%  1.28%  2.48%  1.64%  1.79%  0.93%
##    58:   1.58%  1.22%  2.43%  1.73%  1.68%  0.93%
##    59:   1.60%  1.22%  2.48%  1.59%  1.84%  0.98%
##    60:   1.55%  1.16%  2.35%  1.59%  1.84%  0.93%
##    61:   1.62%  1.25%  2.43%  1.73%  1.89%  0.93%
##    62:   1.61%  1.22%  2.35%  1.78%  1.94%  0.93%
##    63:   1.68%  1.31%  2.56%  1.83%  1.89%  0.93%
##    64:   1.60%  1.13%  2.61%  1.59%  1.94%  0.93%
##    65:   1.66%  1.34%  2.52%  1.69%  1.94%  0.88%
##    66:   1.63%  1.28%  2.39%  1.59%  2.09%  0.93%
##    67:   1.54%  1.19%  2.22%  1.59%  1.89%  0.93%
##    68:   1.56%  1.13%  2.35%  1.69%  1.94%  0.88%
##    69:   1.55%  1.16%  2.31%  1.64%  1.89%  0.88%
##    70:   1.53%  1.13%  2.31%  1.64%  1.84%  0.88%
##    71:   1.49%  1.16%  2.31%  1.54%  1.68%  0.88%
##    72:   1.48%  1.10%  2.31%  1.54%  1.74%  0.83%
##    73:   1.47%  1.04%  2.18%  1.64%  1.84%  0.83%
##    74:   1.48%  1.01%  2.18%  1.73%  1.79%  0.88%
##    75:   1.48%  1.07%  2.18%  1.69%  1.74%  0.88%
##    76:   1.49%  1.13%  2.22%  1.64%  1.74%  0.83%
##    77:   1.46%  1.13%  2.26%  1.45%  1.74%  0.83%
##    78:   1.44%  1.16%  2.14%  1.49%  1.63%  0.83%
##    79:   1.46%  1.22%  2.09%  1.54%  1.74%  0.79%
##    80:   1.46%  1.22%  2.14%  1.49%  1.68%  0.83%
##    81:   1.45%  1.13%  2.14%  1.49%  1.74%  0.88%
##    82:   1.50%  1.22%  2.26%  1.49%  1.74%  0.88%
##    83:   1.47%  1.19%  2.09%  1.59%  1.74%  0.83%
##    84:   1.48%  1.16%  2.18%  1.64%  1.63%  0.88%
##    85:   1.49%  1.16%  2.14%  1.73%  1.63%  0.88%
##    86:   1.43%  1.07%  2.18%  1.54%  1.53%  0.93%
##    87:   1.43%  1.13%  2.22%  1.49%  1.53%  0.83%
##    88:   1.44%  1.13%  2.18%  1.54%  1.68%  0.79%
##    89:   1.43%  1.07%  2.05%  1.73%  1.58%  0.83%
##    90:   1.49%  1.16%  2.26%  1.69%  1.58%  0.88%
##    91:   1.51%  1.16%  2.26%  1.73%  1.63%  0.88%
##    92:   1.46%  1.16%  2.01%  1.69%  1.68%  0.88%
##    93:   1.50%  1.19%  2.14%  1.73%  1.74%  0.83%
##    94:   1.46%  1.13%  2.18%  1.59%  1.68%  0.83%
##    95:   1.45%  1.22%  2.22%  1.59%  1.48%  0.79%
##    96:   1.43%  1.16%  2.14%  1.45%  1.68%  0.79%
##    97:   1.46%  1.25%  2.14%  1.54%  1.63%  0.79%
##    98:   1.43%  1.19%  2.14%  1.54%  1.53%  0.79%
##    99:   1.46%  1.19%  2.22%  1.59%  1.58%  0.79%
##   100:   1.40%  1.10%  2.22%  1.45%  1.58%  0.74%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   6.54%  4.68%  7.48%  8.91%  5.87%  6.88%
##     2:   6.81%  4.77%  7.94%  9.83%  6.77%  5.88%
##     3:   7.14%  5.02%  8.52%  9.11%  7.78%  6.42%
##     4:   6.79%  4.34%  8.92%  8.00%  6.94%  6.97%
##     5:   6.32%  4.22%  8.36%  7.50%  6.29%  6.25%
##     6:   5.74%  3.78%  7.57%  7.32%  5.76%  5.27%
##     7:   5.51%  3.74%  7.23%  6.81%  5.85%  4.84%
##     8:   5.11%  3.80%  7.22%  5.97%  5.27%  3.82%
##     9:   4.69%  3.56%  6.47%  5.93%  4.92%  3.04%
##    10:   4.31%  3.15%  6.44%  4.98%  4.33%  3.08%
##    11:   4.09%  3.17%  5.64%  4.86%  4.21%  2.92%
##    12:   3.86%  3.01%  5.46%  4.51%  4.09%  2.56%
##    13:   3.54%  2.71%  5.22%  4.41%  3.43%  2.22%
##    14:   3.35%  2.74%  4.96%  3.82%  3.58%  1.82%
##    15:   3.27%  2.47%  4.92%  3.92%  3.52%  1.82%
##    16:   3.15%  2.26%  4.53%  3.72%  3.78%  1.82%
##    17:   2.98%  2.14%  4.36%  3.72%  3.27%  1.77%
##    18:   2.87%  2.23%  4.06%  3.47%  3.22%  1.62%
##    19:   2.79%  2.17%  4.14%  3.33%  2.96%  1.57%
##    20:   2.68%  1.93%  4.02%  3.04%  2.91%  1.82%
##    21:   2.62%  1.87%  3.97%  3.13%  2.71%  1.67%
##    22:   2.51%  1.90%  3.67%  3.04%  2.50%  1.62%
##    23:   2.50%  1.90%  3.55%  2.99%  2.71%  1.57%
##    24:   2.49%  1.78%  3.72%  3.04%  2.55%  1.62%
##    25:   2.47%  1.84%  3.59%  2.99%  2.55%  1.62%
##    26:   2.39%  1.75%  3.50%  2.89%  2.60%  1.47%
##    27:   2.38%  1.78%  3.33%  2.84%  2.71%  1.47%
##    28:   2.29%  1.75%  3.25%  2.75%  2.35%  1.57%
##    29:   2.28%  1.69%  3.29%  2.89%  2.40%  1.38%
##    30:   2.27%  1.75%  3.16%  2.75%  2.35%  1.52%
##    31:   2.27%  1.72%  3.33%  2.80%  2.19%  1.47%
##    32:   2.28%  1.81%  3.46%  2.60%  2.35%  1.33%
##    33:   2.25%  1.81%  3.42%  2.65%  2.19%  1.28%
##    34:   2.24%  1.81%  3.33%  2.51%  2.40%  1.28%
##    35:   2.21%  1.78%  3.29%  2.51%  2.40%  1.18%
##    36:   2.16%  1.75%  3.25%  2.27%  2.45%  1.18%
##    37:   2.14%  1.58%  3.16%  2.41%  2.40%  1.38%
##    38:   2.16%  1.55%  3.42%  2.36%  2.35%  1.33%
##    39:   2.17%  1.58%  3.37%  2.36%  2.40%  1.33%
##    40:   2.07%  1.52%  3.33%  2.02%  2.50%  1.18%
##    41:   2.06%  1.58%  3.16%  2.17%  2.45%  1.08%
##    42:   2.03%  1.49%  3.37%  1.98%  2.30%  1.18%
##    43:   2.05%  1.55%  3.12%  2.12%  2.35%  1.28%
##    44:   1.96%  1.46%  3.16%  1.88%  2.19%  1.28%
##    45:   1.92%  1.49%  3.08%  1.88%  2.14%  1.13%
##    46:   2.01%  1.58%  3.25%  1.98%  2.14%  1.23%
##    47:   1.89%  1.49%  2.99%  1.93%  2.04%  1.13%
##    48:   1.89%  1.49%  3.03%  1.83%  2.04%  1.13%
##    49:   1.85%  1.46%  2.99%  1.83%  2.04%  1.03%
##    50:   1.83%  1.46%  2.95%  1.69%  2.19%  0.98%
##    51:   1.83%  1.40%  2.90%  1.78%  2.19%  0.98%
##    52:   1.91%  1.52%  3.08%  2.02%  1.99%  1.03%
##    53:   1.85%  1.49%  2.82%  1.98%  1.99%  1.08%
##    54:   1.85%  1.46%  2.86%  1.98%  1.99%  1.08%
##    55:   1.81%  1.40%  2.82%  1.88%  2.04%  1.03%
##    56:   1.89%  1.60%  2.82%  2.02%  2.09%  0.98%
##    57:   1.89%  1.52%  2.90%  2.02%  2.04%  1.08%
##    58:   1.89%  1.49%  2.95%  1.98%  2.14%  0.98%
##    59:   1.87%  1.58%  2.86%  1.93%  2.04%  0.98%
##    60:   1.86%  1.55%  2.86%  1.98%  2.04%  0.93%
##    61:   1.85%  1.58%  2.82%  1.93%  2.04%  0.93%
##    62:   1.87%  1.58%  2.86%  1.93%  2.09%  0.93%
##    63:   1.86%  1.55%  2.73%  2.02%  2.14%  0.93%
##    64:   1.90%  1.58%  2.99%  1.93%  2.09%  0.98%
##    65:   1.84%  1.40%  2.82%  1.93%  2.14%  1.08%
##    66:   1.82%  1.40%  2.69%  1.93%  2.14%  1.08%
##    67:   1.82%  1.46%  2.82%  1.83%  2.04%  1.03%
##    68:   1.86%  1.46%  2.86%  1.93%  2.14%  1.03%
##    69:   1.84%  1.52%  2.86%  1.88%  2.09%  0.93%
##    70:   1.89%  1.55%  2.86%  1.83%  2.19%  1.08%
##    71:   1.79%  1.49%  2.78%  1.78%  1.99%  0.98%
##    72:   1.81%  1.49%  2.82%  1.73%  1.99%  1.08%
##    73:   1.77%  1.46%  2.56%  1.78%  2.09%  1.03%
##    74:   1.76%  1.49%  2.61%  1.64%  2.09%  1.03%
##    75:   1.77%  1.52%  2.69%  1.69%  1.99%  0.98%
##    76:   1.74%  1.46%  2.56%  1.69%  2.14%  0.93%
##    77:   1.73%  1.43%  2.52%  1.73%  2.04%  1.03%
##    78:   1.74%  1.37%  2.56%  1.69%  2.14%  1.08%
##    79:   1.70%  1.37%  2.43%  1.73%  1.99%  1.08%
##    80:   1.74%  1.37%  2.56%  1.73%  2.09%  1.08%
##    81:   1.74%  1.31%  2.78%  1.69%  2.04%  1.03%
##    82:   1.74%  1.40%  2.65%  1.64%  2.04%  1.08%
##    83:   1.76%  1.31%  2.78%  1.73%  2.04%  1.08%
##    84:   1.73%  1.37%  2.56%  1.78%  2.04%  1.03%
##    85:   1.75%  1.40%  2.61%  1.78%  2.04%  1.03%
##    86:   1.72%  1.34%  2.61%  1.69%  2.04%  1.08%
##    87:   1.70%  1.31%  2.65%  1.73%  1.94%  0.98%
##    88:   1.74%  1.37%  2.61%  1.83%  1.99%  1.03%
##    89:   1.73%  1.37%  2.52%  1.83%  2.04%  1.03%
##    90:   1.77%  1.31%  2.69%  1.83%  2.14%  1.08%
##    91:   1.77%  1.34%  2.61%  1.83%  2.14%  1.13%
##    92:   1.77%  1.31%  2.61%  1.83%  2.14%  1.13%
##    93:   1.78%  1.34%  2.56%  1.88%  2.19%  1.13%
##    94:   1.76%  1.34%  2.52%  1.83%  2.14%  1.13%
##    95:   1.80%  1.40%  2.61%  1.88%  2.14%  1.13%
##    96:   1.77%  1.40%  2.69%  1.83%  1.94%  1.13%
##    97:   1.72%  1.37%  2.48%  1.83%  1.94%  1.13%
##    98:   1.74%  1.40%  2.56%  1.73%  2.04%  1.08%
##    99:   1.75%  1.43%  2.56%  1.78%  2.04%  1.03%
##   100:   1.72%  1.40%  2.52%  1.78%  1.99%  1.03%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.25%  5.88%  8.72%  8.94%  8.30%  5.09%
##     2:   7.13%  5.08%  9.01% 10.43%  6.64%  5.59%
##     3:   6.88%  4.76%  9.53%  9.42%  7.29%  4.50%
##     4:   6.51%  4.55%  9.31%  8.77%  6.46%  4.38%
##     5:   6.02%  4.07%  8.59%  8.56%  5.46%  4.26%
##     6:   5.86%  4.05%  7.66%  8.48%  6.11%  3.87%
##     7:   5.50%  3.73%  7.86%  7.65%  5.13%  3.89%
##     8:   5.10%  3.74%  6.64%  7.35%  4.97%  3.43%
##     9:   4.62%  3.10%  6.28%  6.78%  4.56%  3.11%
##    10:   4.42%  3.18%  5.76%  6.30%  4.33%  3.09%
##    11:   4.02%  2.72%  5.17%  6.07%  4.42%  2.38%
##    12:   3.72%  2.72%  4.77%  5.71%  3.74%  2.13%
##    13:   3.58%  2.44%  4.76%  5.41%  3.79%  2.02%
##    14:   3.29%  2.29%  4.32%  4.49%  3.79%  2.07%
##    15:   3.23%  2.29%  4.15%  4.34%  3.74%  2.12%
##    16:   3.04%  2.20%  3.85%  4.05%  3.47%  2.07%
##    17:   2.91%  2.17%  3.67%  3.71%  3.12%  2.26%
##    18:   2.88%  2.17%  3.67%  3.66%  3.32%  1.92%
##    19:   2.78%  2.14%  3.55%  3.57%  3.27%  1.67%
##    20:   2.66%  1.96%  3.29%  3.57%  3.12%  1.72%
##    21:   2.59%  2.14%  3.03%  3.28%  3.06%  1.67%
##    22:   2.62%  2.20%  3.16%  3.28%  3.01%  1.67%
##    23:   2.46%  1.99%  3.16%  3.08%  2.76%  1.52%
##    24:   2.45%  1.93%  3.08%  2.94%  3.22%  1.33%
##    25:   2.35%  1.99%  2.90%  2.89%  2.86%  1.28%
##    26:   2.43%  1.96%  3.12%  3.04%  2.81%  1.42%
##    27:   2.34%  1.93%  3.03%  2.80%  2.76%  1.33%
##    28:   2.34%  2.02%  2.99%  2.70%  2.71%  1.42%
##    29:   2.39%  1.93%  3.08%  2.84%  2.76%  1.52%
##    30:   2.29%  1.84%  2.82%  3.04%  2.65%  1.33%
##    31:   2.20%  1.90%  2.69%  2.60%  2.65%  1.28%
##    32:   2.20%  1.81%  2.82%  2.70%  2.60%  1.23%
##    33:   2.20%  1.84%  2.56%  2.80%  2.71%  1.28%
##    34:   2.22%  1.87%  2.65%  2.94%  2.50%  1.28%
##    35:   2.14%  1.69%  2.65%  2.75%  2.60%  1.23%
##    36:   2.09%  1.75%  2.61%  2.60%  2.40%  1.23%
##    37:   2.15%  1.69%  2.61%  2.75%  2.50%  1.42%
##    38:   2.09%  1.63%  2.56%  2.70%  2.40%  1.38%
##    39:   2.06%  1.66%  2.56%  2.51%  2.55%  1.23%
##    40:   2.14%  1.66%  2.90%  2.55%  2.55%  1.23%
##    41:   2.06%  1.66%  2.65%  2.55%  2.50%  1.13%
##    42:   2.02%  1.63%  2.52%  2.36%  2.55%  1.23%
##    43:   2.03%  1.58%  2.52%  2.51%  2.55%  1.23%
##    44:   1.97%  1.55%  2.39%  2.41%  2.60%  1.13%
##    45:   2.06%  1.63%  2.56%  2.51%  2.71%  1.13%
##    46:   2.03%  1.55%  2.43%  2.60%  2.65%  1.18%
##    47:   2.00%  1.63%  2.48%  2.46%  2.50%  1.13%
##    48:   2.00%  1.63%  2.39%  2.46%  2.45%  1.23%
##    49:   1.93%  1.60%  2.39%  2.22%  2.45%  1.13%
##    50:   2.00%  1.78%  2.35%  2.36%  2.40%  1.23%
##    51:   2.02%  1.78%  2.39%  2.31%  2.45%  1.28%
##    52:   1.98%  1.72%  2.39%  2.22%  2.50%  1.18%
##    53:   1.99%  1.66%  2.52%  2.31%  2.50%  1.08%
##    54:   1.96%  1.69%  2.39%  2.27%  2.45%  1.13%
##    55:   1.90%  1.60%  2.43%  2.02%  2.50%  1.08%
##    56:   1.91%  1.66%  2.31%  2.02%  2.55%  1.13%
##    57:   1.86%  1.60%  2.31%  1.93%  2.55%  1.03%
##    58:   1.83%  1.63%  2.26%  1.88%  2.40%  1.03%
##    59:   1.79%  1.58%  2.26%  1.78%  2.45%  0.98%
##    60:   1.83%  1.60%  2.31%  1.83%  2.45%  1.03%
##    61:   1.81%  1.63%  2.26%  1.78%  2.45%  0.98%
##    62:   1.76%  1.63%  2.18%  1.59%  2.40%  1.03%
##    63:   1.84%  1.66%  2.39%  1.64%  2.55%  1.03%
##    64:   1.83%  1.63%  2.35%  1.64%  2.45%  1.13%
##    65:   1.79%  1.58%  2.35%  1.59%  2.45%  1.08%
##    66:   1.80%  1.60%  2.26%  1.54%  2.55%  1.13%
##    67:   1.79%  1.52%  2.26%  1.59%  2.60%  1.13%
##    68:   1.74%  1.55%  2.14%  1.54%  2.55%  1.03%
##    69:   1.73%  1.52%  2.09%  1.54%  2.65%  0.98%
##    70:   1.77%  1.55%  2.18%  1.54%  2.71%  1.03%
##    71:   1.80%  1.58%  2.22%  1.64%  2.71%  0.98%
##    72:   1.78%  1.63%  2.18%  1.59%  2.65%  0.93%
##    73:   1.77%  1.58%  2.09%  1.64%  2.71%  0.98%
##    74:   1.78%  1.52%  2.14%  1.73%  2.71%  0.98%
##    75:   1.78%  1.55%  2.18%  1.69%  2.65%  0.98%
##    76:   1.76%  1.52%  2.18%  1.59%  2.60%  1.03%
##    77:   1.79%  1.58%  2.14%  1.69%  2.60%  1.08%
##    78:   1.77%  1.58%  2.14%  1.69%  2.55%  0.98%
##    79:   1.82%  1.60%  2.14%  1.78%  2.60%  1.08%
##    80:   1.81%  1.63%  2.09%  1.69%  2.60%  1.13%
##    81:   1.83%  1.58%  2.18%  1.78%  2.55%  1.18%
##    82:   1.78%  1.55%  2.14%  1.73%  2.55%  1.08%
##    83:   1.80%  1.55%  2.14%  1.83%  2.55%  1.08%
##    84:   1.82%  1.55%  2.14%  1.83%  2.55%  1.18%
##    85:   1.82%  1.63%  2.14%  1.78%  2.50%  1.13%
##    86:   1.81%  1.58%  2.14%  1.83%  2.50%  1.13%
##    87:   1.81%  1.66%  2.14%  1.73%  2.45%  1.13%
##    88:   1.81%  1.58%  2.14%  1.83%  2.50%  1.13%
##    89:   1.80%  1.58%  2.14%  1.83%  2.45%  1.13%
##    90:   1.81%  1.58%  2.18%  1.78%  2.45%  1.18%
##    91:   1.81%  1.58%  2.18%  1.78%  2.45%  1.18%
##    92:   1.81%  1.52%  2.22%  1.83%  2.45%  1.18%
##    93:   1.77%  1.49%  2.22%  1.69%  2.40%  1.18%
##    94:   1.79%  1.58%  2.18%  1.73%  2.45%  1.13%
##    95:   1.75%  1.52%  2.22%  1.64%  2.45%  1.03%
##    96:   1.74%  1.55%  2.09%  1.73%  2.40%  1.03%
##    97:   1.77%  1.55%  2.22%  1.73%  2.45%  1.03%
##    98:   1.74%  1.55%  2.09%  1.73%  2.40%  1.03%
##    99:   1.72%  1.49%  2.05%  1.78%  2.40%  1.03%
##   100:   1.77%  1.58%  2.18%  1.78%  2.45%  0.98%
## ntree      OOB      1      2      3      4      5
##     1:   9.88%  6.45% 12.91% 12.30% 11.25%  8.26%
##     2:   9.91%  6.19% 12.47% 12.02% 11.89%  9.06%
##     3:   9.89%  6.53% 12.81% 12.41% 11.40%  8.22%
##     4:   9.29%  5.89% 12.54% 11.23% 10.23%  8.37%
##     5:   8.78%  5.21% 11.90% 11.24%  9.39%  8.10%
##     6:   7.57%  4.82% 10.01%  9.71%  7.90%  6.82%
##     7:   6.88%  4.27%  9.41%  8.73%  7.82%  5.57%
##     8:   6.54%  4.08%  8.48%  8.36%  7.62%  5.56%
##     9:   5.61%  3.54%  7.13%  7.23%  6.90%  4.51%
##    10:   5.05%  3.22%  7.00%  6.27%  5.74%  4.02%
##    11:   4.71%  3.02%  6.42%  5.96%  5.50%  3.55%
##    12:   4.40%  2.71%  5.67%  5.75%  5.54%  3.35%
##    13:   4.27%  3.01%  5.03%  5.69%  5.06%  3.34%
##    14:   4.05%  2.98%  4.89%  5.20%  4.69%  3.11%
##    15:   3.70%  2.74%  4.64%  4.72%  4.58%  2.41%
##    16:   3.45%  2.31%  4.72%  4.52%  4.21%  2.13%
##    17:   3.30%  2.31%  4.71%  3.84%  3.84%  2.32%
##    18:   3.15%  2.25%  4.24%  4.04%  3.84%  1.90%
##    19:   2.99%  2.10%  4.16%  3.69%  3.84%  1.67%
##    20:   3.00%  2.07%  3.94%  3.60%  4.05%  1.90%
##    21:   2.72%  1.83%  3.81%  3.11%  3.63%  1.72%
##    22:   2.55%  1.80%  3.21%  3.11%  3.47%  1.62%
##    23:   2.51%  1.80%  3.43%  2.97%  3.47%  1.30%
##    24:   2.51%  1.77%  3.34%  3.16%  3.42%  1.30%
##    25:   2.39%  1.95%  3.00%  2.92%  3.16%  1.21%
##    26:   2.37%  1.86%  2.91%  2.97%  3.32%  1.16%
##    27:   2.23%  1.74%  2.66%  2.77%  3.26%  1.11%
##    28:   2.22%  1.74%  2.53%  2.77%  3.16%  1.25%
##    29:   2.17%  1.65%  2.53%  2.72%  3.11%  1.25%
##    30:   2.19%  1.83%  2.57%  2.58%  3.11%  1.16%
##    31:   2.06%  1.65%  2.57%  2.48%  2.84%  1.02%
##    32:   1.92%  1.59%  2.36%  2.28%  2.74%  0.88%
##    33:   1.98%  1.65%  2.40%  2.53%  2.63%  0.93%
##    34:   1.92%  1.56%  2.14%  2.33%  2.74%  1.11%
##    35:   1.89%  1.59%  2.10%  2.19%  2.79%  1.02%
##    36:   1.77%  1.50%  1.97%  2.28%  2.53%  0.83%
##    37:   1.82%  1.41%  2.14%  2.28%  2.68%  0.88%
##    38:   1.73%  1.35%  2.06%  2.14%  2.58%  0.83%
##    39:   1.77%  1.38%  2.40%  2.24%  2.26%  0.79%
##    40:   1.80%  1.35%  2.53%  2.14%  2.47%  0.79%
##    41:   1.69%  1.35%  2.27%  1.94%  2.37%  0.74%
##    42:   1.82%  1.47%  2.40%  2.19%  2.47%  0.79%
##    43:   1.66%  1.35%  2.06%  1.99%  2.37%  0.79%
##    44:   1.68%  1.38%  1.93%  2.09%  2.42%  0.83%
##    45:   1.69%  1.32%  2.01%  2.19%  2.37%  0.83%
##    46:   1.66%  1.38%  1.80%  2.09%  2.37%  0.88%
##    47:   1.68%  1.35%  2.01%  2.14%  2.26%  0.88%
##    48:   1.66%  1.38%  1.84%  2.09%  2.32%  0.88%
##    49:   1.62%  1.23%  1.97%  2.04%  2.37%  0.79%
##    50:   1.62%  1.23%  1.97%  2.04%  2.32%  0.83%
##    51:   1.60%  1.17%  2.06%  1.99%  2.21%  0.83%
##    52:   1.58%  1.14%  1.93%  1.99%  2.16%  0.97%
##    53:   1.55%  1.17%  1.71%  1.99%  2.32%  0.88%
##    54:   1.54%  1.17%  1.97%  1.94%  2.16%  0.70%
##    55:   1.60%  1.17%  2.06%  2.09%  2.16%  0.83%
##    56:   1.51%  1.08%  1.93%  1.94%  2.16%  0.74%
##    57:   1.51%  1.08%  1.88%  1.99%  2.16%  0.74%
##    58:   1.49%  1.14%  1.80%  1.94%  2.16%  0.65%
##    59:   1.46%  1.08%  1.80%  1.80%  2.16%  0.74%
##    60:   1.46%  1.08%  1.88%  1.94%  2.05%  0.60%
##    61:   1.49%  1.08%  1.88%  2.04%  2.11%  0.60%
##    62:   1.51%  1.11%  1.93%  1.90%  2.21%  0.70%
##    63:   1.50%  1.23%  1.97%  1.70%  2.05%  0.74%
##    64:   1.49%  1.02%  1.93%  1.85%  2.16%  0.79%
##    65:   1.42%  0.93%  1.97%  1.60%  2.21%  0.70%
##    66:   1.49%  0.99%  2.06%  1.75%  2.16%  0.79%
##    67:   1.49%  0.96%  2.06%  1.70%  2.26%  0.79%
##    68:   1.46%  0.99%  2.06%  1.75%  2.16%  0.65%
##    69:   1.52%  1.02%  2.01%  1.94%  2.11%  0.83%
##    70:   1.44%  0.93%  1.84%  1.80%  2.21%  0.74%
##    71:   1.46%  0.99%  1.93%  1.70%  2.26%  0.74%
##    72:   1.42%  0.93%  1.93%  1.75%  2.16%  0.65%
##    73:   1.43%  0.90%  1.97%  1.75%  2.21%  0.65%
##    74:   1.46%  0.93%  2.01%  1.80%  2.16%  0.74%
##    75:   1.36%  0.93%  1.80%  1.70%  2.00%  0.65%
##    76:   1.40%  0.90%  1.80%  1.80%  2.16%  0.70%
##    77:   1.42%  0.84%  1.97%  1.85%  2.11%  0.70%
##    78:   1.38%  0.87%  1.97%  1.60%  2.11%  0.70%
##    79:   1.38%  0.87%  1.93%  1.60%  2.21%  0.60%
##    80:   1.39%  0.81%  1.88%  1.75%  2.21%  0.70%
##    81:   1.41%  0.84%  1.93%  1.85%  2.21%  0.60%
##    82:   1.41%  0.84%  2.10%  1.75%  2.11%  0.60%
##    83:   1.35%  0.84%  2.01%  1.70%  2.00%  0.51%
##    84:   1.33%  0.81%  1.97%  1.56%  2.05%  0.60%
##    85:   1.32%  0.84%  1.93%  1.56%  2.05%  0.56%
##    86:   1.35%  0.75%  2.01%  1.65%  2.16%  0.56%
##    87:   1.35%  0.81%  2.01%  1.56%  2.16%  0.56%
##    88:   1.35%  0.78%  2.01%  1.56%  2.21%  0.56%
##    89:   1.34%  0.81%  2.06%  1.46%  2.16%  0.56%
##    90:   1.36%  0.81%  2.10%  1.56%  2.11%  0.56%
##    91:   1.33%  0.81%  2.01%  1.51%  2.11%  0.56%
##    92:   1.35%  0.81%  2.10%  1.46%  2.16%  0.56%
##    93:   1.33%  0.84%  1.97%  1.51%  2.11%  0.56%
##    94:   1.34%  0.81%  2.06%  1.51%  2.00%  0.65%
##    95:   1.33%  0.84%  1.97%  1.51%  2.05%  0.60%
##    96:   1.36%  0.81%  2.01%  1.65%  2.05%  0.60%
##    97:   1.37%  0.78%  2.10%  1.65%  2.05%  0.60%
##    98:   1.35%  0.78%  2.06%  1.60%  2.05%  0.60%
##    99:   1.38%  0.84%  2.10%  1.60%  2.11%  0.56%
##   100:   1.38%  0.84%  2.10%  1.60%  2.11%  0.60%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   8.23%  4.93%  9.95% 12.37% 10.32%  5.67%
##     2:   7.67%  4.79%  9.26% 10.12%  9.78%  6.34%
##     3:   7.33%  4.04%  9.27%  9.82%  9.20%  6.41%
##     4:   6.68%  3.92%  8.71%  8.76%  8.23%  5.39%
##     5:   6.01%  3.79%  7.35%  8.09%  7.39%  4.81%
##     6:   5.68%  3.59%  7.00%  7.69%  7.20%  4.23%
##     7:   5.00%  3.21%  6.38%  6.98%  5.71%  3.75%
##     8:   4.73%  2.97%  6.24%  6.53%  5.52%  3.37%
##     9:   4.29%  2.69%  5.67%  5.83%  5.36%  2.87%
##    10:   4.06%  2.58%  5.33%  5.49%  5.01%  2.76%
##    11:   3.77%  2.39%  4.74%  5.27%  4.77%  2.52%
##    12:   3.50%  2.23%  4.26%  4.97%  4.54%  2.33%
##    13:   3.28%  1.96%  4.42%  4.53%  4.38%  1.91%
##    14:   3.35%  1.93%  4.67%  4.48%  4.43%  2.09%
##    15:   3.15%  2.17%  4.24%  4.04%  3.95%  1.90%
##    16:   2.84%  1.78%  3.64%  3.80%  3.63%  1.99%
##    17:   2.85%  1.81%  3.56%  3.79%  3.74%  2.04%
##    18:   2.78%  1.68%  3.73%  3.21%  3.89%  2.04%
##    19:   2.66%  1.68%  3.68%  3.26%  3.42%  1.81%
##    20:   2.60%  1.62%  3.43%  3.16%  3.63%  1.76%
##    21:   2.51%  1.65%  3.38%  3.01%  3.47%  1.58%
##    22:   2.44%  1.56%  3.30%  2.67%  3.47%  1.72%
##    23:   2.45%  1.68%  3.17%  2.82%  3.32%  1.76%
##    24:   2.38%  1.56%  3.00%  2.77%  3.53%  1.58%
##    25:   2.17%  1.41%  2.74%  2.58%  3.11%  1.53%
##    26:   2.31%  1.44%  3.13%  2.58%  3.37%  1.58%
##    27:   2.24%  1.38%  3.13%  2.67%  3.05%  1.48%
##    28:   2.19%  1.38%  2.87%  2.53%  3.21%  1.48%
##    29:   2.05%  1.26%  2.70%  2.33%  3.11%  1.35%
##    30:   2.00%  1.17%  2.74%  2.24%  3.11%  1.30%
##    31:   2.02%  1.26%  2.74%  2.38%  3.00%  1.21%
##    32:   1.94%  1.20%  2.40%  2.24%  3.05%  1.30%
##    33:   2.00%  1.17%  2.61%  2.48%  3.16%  1.11%
##    34:   1.94%  1.14%  2.53%  2.28%  3.16%  1.11%
##    35:   1.92%  1.29%  2.48%  2.19%  2.95%  1.11%
##    36:   1.96%  1.35%  2.48%  2.28%  3.05%  1.07%
##    37:   1.89%  1.29%  2.61%  2.14%  2.74%  1.07%
##    38:   1.91%  1.32%  2.53%  2.09%  2.89%  1.11%
##    39:   1.94%  1.26%  2.70%  2.14%  3.05%  1.02%
##    40:   1.93%  1.32%  2.53%  2.04%  3.05%  1.11%
##    41:   1.89%  1.29%  2.44%  2.19%  2.89%  1.02%
##    42:   1.89%  1.29%  2.44%  2.24%  2.89%  1.02%
##    43:   1.88%  1.17%  2.40%  2.33%  3.00%  0.97%
##    44:   1.90%  1.29%  2.40%  2.24%  2.89%  1.11%
##    45:   1.88%  1.20%  2.53%  2.24%  2.84%  1.02%
##    46:   1.92%  1.20%  2.61%  2.28%  2.89%  1.07%
##    47:   1.90%  1.26%  2.53%  2.14%  2.95%  1.07%
##    48:   1.91%  1.26%  2.53%  2.28%  2.89%  1.02%
##    49:   1.86%  1.20%  2.48%  2.28%  2.74%  1.02%
##    50:   1.87%  1.20%  2.53%  2.14%  2.84%  1.07%
##    51:   1.88%  1.23%  2.61%  2.19%  2.74%  1.02%
##    52:   1.77%  1.20%  2.27%  2.14%  2.58%  1.07%
##    53:   1.84%  1.20%  2.53%  2.19%  2.68%  1.02%
##    54:   1.77%  1.17%  2.53%  1.99%  2.42%  1.07%
##    55:   1.77%  1.11%  2.36%  2.14%  2.63%  1.07%
##    56:   1.74%  1.05%  2.36%  2.04%  2.68%  1.02%
##    57:   1.74%  1.11%  2.31%  1.99%  2.63%  1.07%
##    58:   1.77%  1.08%  2.44%  2.14%  2.68%  0.97%
##    59:   1.77%  1.08%  2.44%  2.14%  2.63%  1.02%
##    60:   1.79%  1.11%  2.48%  2.14%  2.68%  0.97%
##    61:   1.82%  1.14%  2.44%  2.24%  2.68%  1.02%
##    62:   1.82%  1.14%  2.40%  2.28%  2.63%  1.07%
##    63:   1.81%  1.05%  2.53%  2.19%  2.74%  1.02%
##    64:   1.84%  1.11%  2.48%  2.33%  2.74%  1.02%
##    65:   1.79%  1.08%  2.36%  2.28%  2.63%  1.07%
##    66:   1.81%  1.11%  2.48%  2.24%  2.58%  1.07%
##    67:   1.83%  1.17%  2.53%  2.28%  2.58%  1.02%
##    68:   1.83%  1.14%  2.48%  2.33%  2.68%  0.97%
##    69:   1.79%  1.11%  2.36%  2.28%  2.68%  0.97%
##    70:   1.80%  1.08%  2.40%  2.33%  2.68%  0.97%
##    71:   1.84%  1.23%  2.40%  2.33%  2.68%  0.97%
##    72:   1.83%  1.11%  2.48%  2.38%  2.63%  0.97%
##    73:   1.80%  1.17%  2.36%  2.38%  2.58%  0.93%
##    74:   1.80%  1.11%  2.44%  2.28%  2.58%  1.02%
##    75:   1.84%  1.20%  2.36%  2.43%  2.63%  1.02%
##    76:   1.81%  1.17%  2.27%  2.43%  2.74%  0.88%
##    77:   1.81%  1.14%  2.36%  2.38%  2.63%  0.97%
##    78:   1.83%  1.11%  2.44%  2.38%  2.74%  0.93%
##    79:   1.77%  1.11%  2.18%  2.33%  2.68%  0.97%
##    80:   1.76%  1.08%  2.18%  2.33%  2.63%  1.02%
##    81:   1.77%  1.05%  2.27%  2.28%  2.74%  0.97%
##    82:   1.76%  1.02%  2.31%  2.24%  2.74%  0.97%
##    83:   1.75%  1.08%  2.18%  2.19%  2.74%  1.02%
##    84:   1.77%  1.02%  2.31%  2.28%  2.79%  0.93%
##    85:   1.77%  1.08%  2.23%  2.38%  2.68%  0.97%
##    86:   1.72%  0.99%  2.18%  2.28%  2.74%  0.88%
##    87:   1.76%  1.05%  2.14%  2.33%  2.89%  0.88%
##    88:   1.75%  1.05%  2.14%  2.33%  2.89%  0.83%
##    89:   1.75%  1.02%  2.18%  2.38%  2.84%  0.83%
##    90:   1.77%  0.99%  2.23%  2.38%  2.95%  0.83%
##    91:   1.77%  1.08%  2.14%  2.38%  2.89%  0.88%
##    92:   1.75%  1.02%  2.10%  2.43%  2.89%  0.83%
##    93:   1.77%  1.02%  2.14%  2.43%  2.95%  0.83%
##    94:   1.74%  1.02%  2.06%  2.33%  2.95%  0.88%
##    95:   1.70%  0.96%  1.97%  2.33%  2.84%  0.93%
##    96:   1.76%  1.02%  2.06%  2.33%  2.95%  0.97%
##    97:   1.77%  1.05%  2.14%  2.38%  2.84%  0.93%
##    98:   1.77%  1.02%  2.14%  2.43%  2.84%  0.97%
##    99:   1.76%  1.05%  2.06%  2.38%  2.89%  0.93%
##   100:   1.73%  1.02%  2.10%  2.38%  2.68%  0.97%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.11%  4.94% 10.36% 10.90%  5.87%  4.52%
##     2:   7.56%  5.30%  9.37% 11.60%  7.73%  5.03%
##     3:   7.12%  5.48%  9.33% 10.76%  6.80%  3.97%
##     4:   6.54%  4.81%  8.27%  9.75%  6.81%  4.04%
##     5:   6.01%  4.53%  7.01%  8.61%  7.01%  3.87%
##     6:   5.75%  4.19%  7.01%  8.04%  6.72%  3.79%
##     7:   5.26%  3.66%  6.91%  7.41%  6.18%  3.12%
##     8:   4.89%  3.73%  6.02%  7.01%  5.38%  3.04%
##     9:   4.65%  3.54%  5.39%  6.17%  6.02%  2.96%
##    10:   4.21%  3.06%  4.86%  5.97%  5.33%  2.62%
##    11:   4.00%  2.78%  4.46%  6.10%  5.21%  2.33%
##    12:   3.69%  2.44%  4.05%  5.22%  5.40%  2.28%
##    13:   3.53%  2.47%  4.30%  4.97%  4.81%  1.86%
##    14:   3.41%  2.17%  4.21%  4.77%  4.81%  1.95%
##    15:   3.30%  1.96%  4.08%  4.52%  4.80%  2.04%
##    16:   3.11%  1.98%  3.60%  4.28%  4.53%  1.95%
##    17:   3.06%  1.86%  3.68%  4.33%  4.48%  1.76%
##    18:   2.87%  1.98%  3.30%  3.69%  4.27%  1.76%
##    19:   2.90%  1.89%  3.38%  4.08%  4.27%  1.58%
##    20:   2.87%  1.80%  3.51%  3.89%  4.11%  1.76%
##    21:   2.67%  1.65%  3.34%  3.60%  4.00%  1.44%
##    22:   2.68%  1.59%  3.21%  3.50%  4.27%  1.58%
##    23:   2.54%  1.53%  3.04%  3.50%  3.84%  1.48%
##    24:   2.48%  1.53%  2.91%  3.16%  4.11%  1.39%
##    25:   2.39%  1.38%  2.91%  3.21%  3.95%  1.21%
##    26:   2.34%  1.47%  2.66%  3.40%  3.69%  1.11%
##    27:   2.22%  1.29%  2.66%  3.11%  3.74%  0.97%
##    28:   2.28%  1.41%  2.74%  2.97%  3.79%  1.16%
##    29:   2.25%  1.32%  2.78%  2.87%  3.74%  1.21%
##    30:   2.22%  1.35%  2.40%  3.06%  3.63%  1.30%
##    31:   2.22%  1.41%  2.48%  2.82%  3.79%  1.21%
##    32:   2.20%  1.44%  2.61%  2.87%  3.58%  1.07%
##    33:   2.20%  1.38%  2.57%  2.67%  3.68%  1.30%
##    34:   2.18%  1.44%  2.66%  2.67%  3.58%  1.11%
##    35:   2.11%  1.32%  2.66%  2.53%  3.53%  1.07%
##    36:   2.06%  1.26%  2.57%  2.63%  3.47%  0.97%
##    37:   2.09%  1.29%  2.57%  2.58%  3.47%  1.11%
##    38:   2.06%  1.20%  2.57%  2.72%  3.42%  0.97%
##    39:   2.06%  1.32%  2.40%  2.77%  3.42%  0.93%
##    40:   1.98%  1.17%  2.40%  2.53%  3.42%  0.97%
##    41:   1.94%  1.11%  2.40%  2.43%  3.37%  0.97%
##    42:   1.89%  1.05%  2.31%  2.33%  3.32%  1.02%
##    43:   1.92%  1.20%  2.36%  2.28%  3.37%  0.93%
##    44:   1.98%  1.14%  2.48%  2.33%  3.42%  1.11%
##    45:   1.99%  1.26%  2.40%  2.43%  3.26%  1.11%
##    46:   1.94%  1.17%  2.31%  2.43%  3.21%  1.11%
##    47:   1.94%  1.14%  2.36%  2.53%  3.21%  1.02%
##    48:   1.94%  1.20%  2.27%  2.33%  3.37%  1.07%
##    49:   1.90%  1.23%  2.40%  2.14%  3.21%  1.02%
##    50:   1.94%  1.26%  2.31%  2.19%  3.42%  1.07%
##    51:   1.97%  1.29%  2.44%  2.24%  3.32%  1.07%
##    52:   1.94%  1.20%  2.48%  2.14%  3.37%  1.07%
##    53:   1.96%  1.20%  2.48%  2.19%  3.47%  1.02%
##    54:   1.92%  1.17%  2.44%  2.28%  3.21%  1.02%
##    55:   1.89%  1.17%  2.40%  2.19%  3.21%  1.02%
##    56:   1.95%  1.20%  2.66%  2.28%  3.11%  1.02%
##    57:   1.93%  1.23%  2.57%  2.19%  3.16%  0.97%
##    58:   1.83%  1.14%  2.40%  1.94%  3.05%  1.07%
##    59:   1.88%  1.23%  2.57%  1.90%  3.05%  1.07%
##    60:   1.85%  1.23%  2.44%  1.94%  2.95%  1.11%
##    61:   1.76%  1.08%  2.36%  1.80%  3.00%  1.02%
##    62:   1.80%  1.20%  2.44%  1.90%  2.79%  1.07%
##    63:   1.79%  1.11%  2.40%  1.94%  2.95%  1.02%
##    64:   1.83%  1.14%  2.44%  1.94%  3.11%  0.97%
##    65:   1.77%  1.05%  2.40%  1.80%  3.05%  1.07%
##    66:   1.77%  1.17%  2.23%  1.85%  3.00%  1.07%
##    67:   1.76%  0.99%  2.36%  1.75%  3.11%  1.11%
##    68:   1.77%  1.08%  2.31%  1.75%  3.11%  1.11%
##    69:   1.71%  0.99%  2.27%  1.85%  2.89%  1.02%
##    70:   1.72%  0.96%  2.40%  1.80%  3.00%  0.97%
##    71:   1.70%  1.02%  2.36%  1.85%  2.84%  0.88%
##    72:   1.67%  0.99%  2.23%  1.80%  2.89%  0.93%
##    73:   1.71%  1.05%  2.31%  1.75%  3.00%  0.88%
##    74:   1.72%  1.11%  2.27%  1.85%  2.95%  0.88%
##    75:   1.72%  1.05%  2.36%  1.75%  2.95%  0.97%
##    76:   1.66%  1.02%  2.23%  1.75%  2.79%  0.97%
##    77:   1.66%  0.99%  2.23%  1.80%  2.84%  0.88%
##    78:   1.68%  1.05%  2.31%  1.80%  2.79%  0.88%
##    79:   1.69%  1.02%  2.31%  1.85%  2.84%  0.88%
##    80:   1.70%  1.08%  2.27%  1.85%  2.79%  0.93%
##    81:   1.68%  1.08%  2.23%  1.85%  2.74%  0.93%
##    82:   1.71%  1.11%  2.27%  1.85%  2.79%  0.93%
##    83:   1.68%  1.08%  2.18%  1.85%  2.84%  0.88%
##    84:   1.69%  1.11%  2.18%  1.90%  2.79%  0.88%
##    85:   1.67%  1.11%  2.10%  1.90%  2.79%  0.88%
##    86:   1.68%  1.08%  2.14%  1.90%  2.84%  0.88%
##    87:   1.64%  1.08%  2.06%  1.85%  2.74%  0.88%
##    88:   1.66%  1.08%  2.01%  1.85%  2.95%  0.88%
##    89:   1.69%  1.11%  2.23%  1.85%  2.79%  0.88%
##    90:   1.66%  1.11%  2.14%  1.85%  2.74%  0.88%
##    91:   1.67%  1.08%  2.10%  1.99%  2.74%  0.88%
##    92:   1.64%  1.11%  2.06%  1.90%  2.63%  0.88%
##    93:   1.66%  1.17%  2.10%  1.94%  2.63%  0.83%
##    94:   1.65%  1.11%  2.01%  1.99%  2.63%  0.88%
##    95:   1.62%  1.08%  1.93%  1.94%  2.74%  0.83%
##    96:   1.65%  1.08%  1.97%  2.04%  2.74%  0.83%
##    97:   1.68%  1.11%  2.06%  2.09%  2.74%  0.83%
##    98:   1.65%  1.08%  2.10%  1.90%  2.74%  0.83%
##    99:   1.63%  1.08%  2.10%  1.90%  2.68%  0.79%
##   100:   1.70%  1.20%  2.14%  1.99%  2.68%  0.83%
## ntree      OOB      1      2      3      4      5
##     1:  10.74%  5.96% 16.93% 14.27%  8.47% 10.42%
##     2:  10.87%  6.64% 16.67% 13.05%  9.58% 10.47%
##     3:  10.17%  6.43% 13.97% 13.77%  8.81%  9.79%
##     4:   9.61%  5.45% 13.64% 12.46%  9.61%  9.15%
##     5:   8.93%  5.20% 12.30% 11.59%  9.31%  8.32%
##     6:   8.29%  5.09% 11.43% 10.45%  8.17%  8.01%
##     7:   7.63%  4.97% 10.56%  8.65%  8.17%  7.25%
##     8:   6.77%  4.49%  9.44%  7.48%  6.93%  6.71%
##     9:   6.25%  4.10%  8.31%  7.47%  6.71%  5.87%
##    10:   5.74%  4.01%  7.49%  7.03%  5.81%  5.32%
##    11:   5.46%  3.57%  7.47%  6.37%  6.25%  4.75%
##    12:   4.71%  3.15%  6.15%  6.20%  5.04%  3.97%
##    13:   4.53%  3.17%  5.87%  5.85%  5.08%  3.51%
##    14:   4.42%  3.02%  5.91%  5.80%  4.72%  3.50%
##    15:   4.15%  2.99%  6.04%  5.10%  3.99%  3.23%
##    16:   3.88%  2.81%  4.84%  5.29%  4.30%  2.86%
##    17:   3.64%  2.69%  4.75%  4.56%  4.20%  2.64%
##    18:   3.33%  2.45%  4.26%  4.31%  3.89%  2.32%
##    19:   3.01%  2.18%  3.41%  4.41%  3.52%  2.14%
##    20:   2.89%  2.12%  3.73%  3.82%  3.47%  1.82%
##    21:   2.74%  1.97%  3.64%  3.58%  3.00%  2.00%
##    22:   2.62%  2.03%  3.24%  3.33%  3.16%  1.73%
##    23:   2.46%  1.88%  3.10%  3.14%  2.95%  1.64%
##    24:   2.47%  1.88%  3.06%  3.33%  2.80%  1.68%
##    25:   2.30%  1.73%  2.97%  3.04%  2.74%  1.41%
##    26:   2.29%  1.70%  3.37%  2.84%  2.48%  1.41%
##    27:   2.28%  1.55%  3.02%  2.74%  2.74%  1.77%
##    28:   2.13%  1.34%  2.71%  2.89%  2.90%  1.36%
##    29:   2.13%  1.37%  3.06%  2.55%  2.64%  1.50%
##    30:   2.11%  1.37%  2.84%  2.79%  2.64%  1.41%
##    31:   2.07%  1.55%  2.93%  2.50%  2.43%  1.27%
##    32:   2.00%  1.31%  2.79%  2.55%  2.38%  1.41%
##    33:   1.97%  1.20%  2.88%  2.35%  2.43%  1.45%
##    34:   2.06%  1.25%  3.15%  2.35%  2.54%  1.50%
##    35:   1.94%  1.25%  2.79%  2.25%  2.43%  1.41%
##    36:   1.87%  1.17%  2.79%  2.20%  2.43%  1.18%
##    37:   1.83%  1.14%  2.79%  2.11%  2.38%  1.14%
##    38:   1.83%  1.22%  2.75%  2.06%  2.38%  1.09%
##    39:   1.86%  1.31%  2.75%  2.30%  2.28%  1.00%
##    40:   1.78%  1.08%  2.53%  2.40%  2.33%  1.04%
##    41:   1.73%  1.02%  2.48%  2.30%  2.33%  1.00%
##    42:   1.81%  1.08%  2.62%  2.45%  2.43%  0.95%
##    43:   1.79%  1.14%  2.44%  2.25%  2.54%  1.04%
##    44:   1.74%  1.11%  2.31%  2.30%  2.33%  1.09%
##    45:   1.70%  1.02%  2.44%  2.16%  2.33%  1.00%
##    46:   1.67%  1.05%  2.39%  2.11%  2.33%  0.91%
##    47:   1.61%  0.96%  2.35%  1.91%  2.43%  0.86%
##    48:   1.65%  1.02%  2.39%  2.06%  2.28%  0.91%
##    49:   1.69%  1.17%  2.35%  2.11%  2.28%  0.91%
##    50:   1.69%  1.14%  2.44%  2.16%  2.17%  0.91%
##    51:   1.70%  1.20%  2.26%  2.06%  2.43%  0.91%
##    52:   1.68%  1.17%  2.26%  2.16%  2.28%  0.91%
##    53:   1.67%  1.11%  2.26%  2.01%  2.43%  0.95%
##    54:   1.66%  1.17%  2.35%  1.96%  2.38%  0.82%
##    55:   1.65%  1.20%  2.35%  2.06%  2.23%  0.73%
##    56:   1.64%  1.17%  2.26%  2.01%  2.28%  0.82%
##    57:   1.63%  1.20%  2.13%  1.96%  2.33%  0.86%
##    58:   1.58%  1.08%  2.13%  2.06%  2.23%  0.77%
##    59:   1.61%  1.14%  2.26%  2.01%  2.23%  0.77%
##    60:   1.65%  1.08%  2.26%  2.11%  2.38%  0.82%
##    61:   1.61%  1.14%  2.17%  2.01%  2.17%  0.91%
##    62:   1.62%  1.08%  2.31%  1.96%  2.28%  0.86%
##    63:   1.61%  1.14%  2.22%  1.96%  2.28%  0.82%
##    64:   1.68%  1.22%  2.26%  2.20%  2.23%  0.82%
##    65:   1.67%  1.17%  2.31%  2.16%  2.17%  0.91%
##    66:   1.63%  1.11%  2.26%  2.25%  2.12%  0.77%
##    67:   1.61%  1.02%  2.35%  2.16%  2.12%  0.82%
##    68:   1.62%  1.05%  2.35%  2.11%  2.17%  0.82%
##    69:   1.59%  1.02%  2.31%  1.96%  2.12%  0.91%
##    70:   1.59%  1.02%  2.39%  2.11%  1.97%  0.82%
##    71:   1.60%  1.02%  2.35%  2.06%  2.17%  0.82%
##    72:   1.60%  1.02%  2.31%  1.96%  2.23%  0.86%
##    73:   1.54%  0.93%  2.48%  1.76%  2.02%  0.86%
##    74:   1.59%  1.02%  2.39%  1.96%  2.07%  0.86%
##    75:   1.60%  0.99%  2.48%  1.96%  2.02%  0.91%
##    76:   1.61%  0.96%  2.48%  2.01%  2.07%  0.95%
##    77:   1.55%  0.96%  2.44%  1.91%  1.97%  0.86%
##    78:   1.57%  0.99%  2.31%  1.96%  2.02%  0.95%
##    79:   1.57%  0.93%  2.39%  2.01%  1.97%  0.95%
##    80:   1.53%  0.93%  2.31%  1.91%  1.97%  0.91%
##    81:   1.55%  0.96%  2.35%  1.86%  2.02%  0.95%
##    82:   1.49%  0.90%  2.31%  1.67%  2.02%  0.91%
##    83:   1.47%  0.93%  2.31%  1.67%  1.97%  0.82%
##    84:   1.49%  0.90%  2.22%  1.81%  2.02%  0.86%
##    85:   1.54%  0.93%  2.35%  1.86%  2.02%  0.91%
##    86:   1.51%  0.93%  2.31%  1.81%  1.97%  0.91%
##    87:   1.51%  0.93%  2.26%  1.96%  1.92%  0.86%
##    88:   1.45%  0.87%  2.08%  1.91%  1.97%  0.82%
##    89:   1.46%  0.87%  2.26%  1.81%  1.92%  0.82%
##    90:   1.40%  0.84%  2.08%  1.81%  1.86%  0.77%
##    91:   1.41%  0.87%  2.13%  1.71%  1.92%  0.77%
##    92:   1.39%  0.84%  1.95%  1.76%  1.97%  0.82%
##    93:   1.36%  0.81%  1.91%  1.71%  1.92%  0.82%
##    94:   1.41%  0.90%  1.95%  1.81%  1.92%  0.82%
##    95:   1.44%  0.93%  1.95%  1.81%  2.07%  0.82%
##    96:   1.44%  0.90%  2.04%  1.81%  1.92%  0.86%
##    97:   1.44%  0.90%  2.08%  1.91%  1.86%  0.82%
##    98:   1.44%  0.84%  2.13%  1.91%  1.92%  0.82%
##    99:   1.41%  0.84%  2.00%  1.86%  1.97%  0.77%
##   100:   1.43%  0.93%  2.04%  1.67%  1.92%  0.91%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   8.42%  5.67% 12.05% 10.10%  8.48%  7.09%
##     2:   7.87%  4.82% 10.37% 10.68%  9.27%  5.94%
##     3:   7.72%  5.16% 11.02% 10.05%  8.03%  5.70%
##     4:   7.56%  5.19% 10.69%  9.68%  8.13%  5.40%
##     5:   6.69%  4.57% 10.05%  8.00%  7.38%  4.60%
##     6:   6.44%  4.58%  9.18%  7.69%  7.06%  4.74%
##     7:   6.10%  4.56%  8.16%  7.04%  7.49%  4.21%
##     8:   5.28%  3.87%  7.38%  6.31%  6.29%  3.45%
##     9:   4.87%  3.67%  7.26%  5.12%  5.71%  3.27%
##    10:   4.55%  3.01%  6.28%  5.68%  5.01%  3.67%
##    11:   4.42%  3.03%  6.43%  5.17%  5.05%  3.24%
##    12:   3.93%  2.82%  5.52%  4.87%  4.63%  2.51%
##    13:   3.85%  2.55%  5.60%  4.62%  4.57%  2.69%
##    14:   3.72%  2.39%  5.77%  4.76%  4.10%  2.32%
##    15:   3.54%  2.51%  5.19%  4.36%  4.35%  1.96%
##    16:   3.40%  2.45%  4.92%  4.70%  3.52%  1.95%
##    17:   3.12%  2.06%  4.75%  3.87%  3.83%  1.73%
##    18:   3.18%  2.09%  4.88%  4.26%  4.04%  1.36%
##    19:   3.07%  2.03%  4.66%  4.16%  3.62%  1.55%
##    20:   2.98%  1.91%  4.75%  4.07%  3.47%  1.36%
##    21:   2.81%  1.73%  4.39%  3.67%  3.36%  1.54%
##    22:   2.78%  1.76%  4.30%  3.82%  3.31%  1.32%
##    23:   2.78%  1.85%  4.48%  3.63%  3.26%  1.23%
##    24:   2.70%  1.79%  4.30%  3.53%  3.21%  1.23%
##    25:   2.54%  1.61%  4.21%  3.23%  3.16%  1.04%
##    26:   2.56%  1.64%  4.04%  3.23%  3.21%  1.23%
##    27:   2.44%  1.64%  3.81%  2.84%  3.11%  1.27%
##    28:   2.52%  1.61%  3.90%  3.43%  3.00%  1.23%
##    29:   2.41%  1.49%  3.77%  3.33%  2.85%  1.18%
##    30:   2.45%  1.61%  3.86%  3.04%  3.11%  1.14%
##    31:   2.51%  1.52%  3.99%  3.33%  3.11%  1.18%
##    32:   2.50%  1.64%  3.95%  3.18%  3.00%  1.23%
##    33:   2.47%  1.58%  4.04%  2.94%  3.00%  1.32%
##    34:   2.47%  1.43%  4.12%  2.99%  3.00%  1.41%
##    35:   2.48%  1.61%  3.99%  2.99%  3.05%  1.27%
##    36:   2.46%  1.64%  3.90%  3.14%  2.95%  1.18%
##    37:   2.45%  1.61%  3.81%  3.14%  3.00%  1.23%
##    38:   2.44%  1.61%  3.95%  2.99%  2.95%  1.18%
##    39:   2.46%  1.61%  3.86%  3.18%  3.00%  1.18%
##    40:   2.47%  1.82%  3.95%  3.28%  2.69%  1.00%
##    41:   2.40%  1.67%  3.77%  3.18%  2.80%  1.04%
##    42:   2.42%  1.58%  3.95%  3.09%  2.90%  1.09%
##    43:   2.42%  1.58%  4.12%  3.09%  2.74%  1.04%
##    44:   2.28%  1.46%  3.73%  2.89%  2.74%  1.04%
##    45:   2.24%  1.49%  3.73%  2.89%  2.59%  0.95%
##    46:   2.24%  1.49%  3.77%  2.74%  2.69%  0.95%
##    47:   2.31%  1.52%  3.86%  2.84%  2.74%  1.04%
##    48:   2.26%  1.43%  3.77%  2.79%  2.74%  1.04%
##    49:   2.26%  1.46%  3.73%  2.84%  2.64%  1.09%
##    50:   2.28%  1.43%  3.86%  2.79%  2.74%  1.04%
##    51:   2.22%  1.43%  3.55%  2.79%  2.74%  1.04%
##    52:   2.28%  1.52%  3.77%  2.79%  2.69%  1.09%
##    53:   2.28%  1.43%  3.86%  2.79%  2.80%  1.04%
##    54:   2.25%  1.43%  3.73%  2.79%  2.69%  1.09%
##    55:   2.22%  1.34%  3.77%  2.84%  2.59%  1.04%
##    56:   2.19%  1.40%  3.73%  2.74%  2.54%  1.00%
##    57:   2.13%  1.34%  3.64%  2.74%  2.43%  0.95%
##    58:   2.17%  1.40%  3.68%  2.69%  2.38%  1.09%
##    59:   2.11%  1.37%  3.46%  2.74%  2.33%  1.09%
##    60:   2.11%  1.37%  3.46%  2.65%  2.43%  1.09%
##    61:   2.09%  1.31%  3.41%  2.79%  2.38%  1.00%
##    62:   2.10%  1.37%  3.55%  2.65%  2.38%  0.95%
##    63:   2.12%  1.37%  3.50%  2.79%  2.38%  1.00%
##    64:   2.07%  1.34%  3.41%  2.65%  2.48%  0.91%
##    65:   2.06%  1.37%  3.41%  2.60%  2.43%  0.91%
##    66:   2.08%  1.34%  3.50%  2.74%  2.33%  0.91%
##    67:   2.13%  1.43%  3.55%  2.69%  2.48%  0.91%
##    68:   2.00%  1.31%  3.24%  2.65%  2.28%  0.95%
##    69:   2.05%  1.34%  3.33%  2.74%  2.33%  0.91%
##    70:   2.04%  1.31%  3.28%  2.65%  2.38%  1.00%
##    71:   2.06%  1.31%  3.41%  2.65%  2.38%  0.95%
##    72:   2.05%  1.25%  3.28%  2.84%  2.33%  1.00%
##    73:   2.02%  1.25%  3.28%  2.69%  2.38%  0.95%
##    74:   2.00%  1.20%  3.33%  2.69%  2.38%  0.91%
##    75:   2.00%  1.22%  3.28%  2.65%  2.38%  0.95%
##    76:   1.99%  1.28%  3.24%  2.60%  2.38%  0.86%
##    77:   1.94%  1.22%  3.24%  2.50%  2.23%  0.91%
##    78:   1.94%  1.20%  3.28%  2.55%  2.23%  0.91%
##    79:   2.00%  1.22%  3.24%  2.60%  2.48%  0.95%
##    80:   1.97%  1.25%  3.19%  2.55%  2.33%  0.95%
##    81:   2.03%  1.37%  3.24%  2.50%  2.43%  1.00%
##    82:   1.99%  1.31%  3.10%  2.50%  2.48%  0.95%
##    83:   1.95%  1.28%  3.15%  2.60%  2.28%  0.86%
##    84:   1.97%  1.28%  3.10%  2.60%  2.43%  0.86%
##    85:   1.93%  1.25%  3.15%  2.55%  2.28%  0.82%
##    86:   1.94%  1.28%  3.15%  2.50%  2.33%  0.82%
##    87:   1.94%  1.22%  3.10%  2.60%  2.33%  0.91%
##    88:   1.94%  1.22%  3.19%  2.50%  2.38%  0.86%
##    89:   1.95%  1.22%  3.24%  2.55%  2.33%  0.86%
##    90:   1.94%  1.22%  3.19%  2.55%  2.38%  0.82%
##    91:   1.98%  1.28%  3.24%  2.60%  2.38%  0.82%
##    92:   1.95%  1.28%  3.15%  2.65%  2.28%  0.82%
##    93:   1.95%  1.28%  3.19%  2.55%  2.33%  0.82%
##    94:   1.94%  1.22%  3.15%  2.45%  2.43%  0.91%
##    95:   1.96%  1.22%  3.24%  2.55%  2.38%  0.86%
##    96:   2.01%  1.28%  3.24%  2.60%  2.48%  0.91%
##    97:   1.97%  1.25%  3.24%  2.55%  2.48%  0.77%
##    98:   1.96%  1.22%  3.24%  2.55%  2.43%  0.82%
##    99:   1.99%  1.25%  3.28%  2.45%  2.54%  0.86%
##   100:   1.98%  1.22%  3.28%  2.50%  2.54%  0.82%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.48%  5.91% 10.58%  8.89%  8.80%  4.38%
##     2:   7.71%  5.46% 11.37%  9.13%  9.09%  4.92%
##     3:   7.38%  5.21% 10.83%  8.46%  8.94%  4.86%
##     4:   7.36%  5.21% 10.37%  8.38%  9.36%  4.89%
##     5:   6.81%  4.76%  9.63%  8.75%  7.47%  4.63%
##     6:   6.38%  4.35%  9.81%  8.11%  7.00%  3.82%
##     7:   5.75%  3.88%  8.86%  7.14%  6.04%  3.87%
##     8:   5.25%  3.59%  7.64%  6.60%  5.84%  3.57%
##     9:   4.97%  3.58%  7.36%  6.41%  5.09%  3.23%
##    10:   4.53%  3.08%  6.76%  5.63%  4.95%  3.07%
##    11:   4.26%  2.89%  6.60%  5.45%  4.74%  2.38%
##    12:   4.05%  2.76%  5.96%  5.26%  4.73%  2.33%
##    13:   3.75%  2.70%  5.59%  4.86%  3.74%  2.46%
##    14:   3.66%  2.51%  5.33%  4.76%  3.99%  2.36%
##    15:   3.32%  2.27%  5.15%  4.36%  3.42%  2.00%
##    16:   3.30%  2.18%  5.06%  4.31%  3.52%  2.09%
##    17:   3.17%  2.15%  4.79%  4.31%  3.16%  2.00%
##    18:   3.03%  2.15%  4.48%  4.12%  3.31%  1.64%
##    19:   2.89%  2.18%  4.52%  3.58%  2.95%  1.59%
##    20:   2.89%  1.97%  4.35%  3.58%  3.47%  1.64%
##    21:   2.79%  2.03%  4.04%  3.48%  3.31%  1.54%
##    22:   2.84%  1.94%  4.04%  3.72%  3.52%  1.54%
##    23:   2.82%  1.88%  4.04%  3.63%  3.52%  1.64%
##    24:   2.71%  1.70%  4.04%  3.38%  3.52%  1.54%
##    25:   2.77%  1.88%  4.08%  3.53%  3.36%  1.54%
##    26:   2.62%  1.79%  3.81%  3.38%  3.21%  1.45%
##    27:   2.67%  1.85%  3.86%  3.43%  3.21%  1.50%
##    28:   2.68%  1.85%  4.04%  3.53%  3.11%  1.41%
##    29:   2.56%  1.76%  3.77%  3.43%  3.00%  1.36%
##    30:   2.51%  1.67%  3.77%  3.28%  3.00%  1.32%
##    31:   2.44%  1.55%  3.64%  3.14%  3.05%  1.36%
##    32:   2.47%  1.79%  3.64%  3.09%  3.00%  1.27%
##    33:   2.31%  1.49%  3.46%  3.04%  2.85%  1.23%
##    34:   2.39%  1.61%  3.59%  3.04%  3.05%  1.18%
##    35:   2.45%  1.79%  3.59%  3.23%  2.85%  1.23%
##    36:   2.41%  1.55%  3.64%  3.23%  3.05%  1.14%
##    37:   2.37%  1.49%  3.68%  3.04%  2.90%  1.27%
##    38:   2.41%  1.64%  3.50%  3.23%  2.95%  1.23%
##    39:   2.34%  1.55%  3.37%  3.28%  2.80%  1.23%
##    40:   2.30%  1.43%  3.37%  3.48%  2.80%  1.00%
##    41:   2.22%  1.46%  3.15%  3.23%  2.74%  1.04%
##    42:   2.25%  1.40%  3.19%  3.33%  2.85%  1.04%
##    43:   2.29%  1.37%  3.41%  3.33%  2.85%  1.09%
##    44:   2.20%  1.28%  3.46%  3.18%  2.74%  0.91%
##    45:   2.24%  1.28%  3.55%  3.28%  2.64%  1.04%
##    46:   2.25%  1.37%  3.59%  3.14%  2.74%  0.95%
##    47:   2.25%  1.25%  3.59%  3.23%  2.90%  0.91%
##    48:   2.25%  1.25%  3.73%  3.14%  2.85%  0.91%
##    49:   2.18%  1.28%  3.50%  3.04%  2.74%  0.91%
##    50:   2.17%  1.31%  3.50%  3.09%  2.59%  0.91%
##    51:   2.11%  1.31%  3.46%  2.94%  2.48%  0.86%
##    52:   2.17%  1.37%  3.37%  3.04%  2.69%  0.91%
##    53:   2.19%  1.28%  3.59%  3.18%  2.59%  0.86%
##    54:   2.16%  1.22%  3.37%  3.09%  2.69%  1.00%
##    55:   2.16%  1.25%  3.46%  2.99%  2.64%  1.00%
##    56:   2.21%  1.31%  3.46%  3.18%  2.64%  1.00%
##    57:   2.15%  1.37%  3.19%  3.18%  2.54%  0.95%
##    58:   2.17%  1.31%  3.37%  3.04%  2.59%  1.04%
##    59:   2.20%  1.34%  3.33%  3.18%  2.69%  1.00%
##    60:   2.14%  1.31%  3.28%  3.09%  2.64%  0.91%
##    61:   2.11%  1.31%  3.15%  3.09%  2.59%  0.91%
##    62:   2.15%  1.31%  3.15%  3.18%  2.69%  0.95%
##    63:   2.11%  1.25%  3.15%  3.09%  2.64%  0.95%
##    64:   2.12%  1.28%  3.15%  3.18%  2.54%  1.00%
##    65:   2.10%  1.22%  3.19%  3.09%  2.54%  1.00%
##    66:   2.04%  1.20%  3.06%  2.94%  2.59%  0.95%
##    67:   2.03%  1.17%  3.10%  2.94%  2.54%  0.95%
##    68:   2.05%  1.20%  3.15%  2.89%  2.59%  0.95%
##    69:   2.06%  1.25%  3.06%  2.99%  2.54%  0.95%
##    70:   2.07%  1.28%  3.24%  2.74%  2.64%  0.95%
##    71:   2.05%  1.25%  3.15%  2.79%  2.59%  0.95%
##    72:   2.00%  1.20%  3.10%  2.69%  2.64%  0.91%
##    73:   1.99%  1.17%  3.10%  2.65%  2.69%  0.86%
##    74:   2.02%  1.25%  3.06%  2.74%  2.59%  0.95%
##    75:   2.04%  1.25%  3.10%  2.69%  2.64%  1.00%
##    76:   2.07%  1.31%  3.15%  2.84%  2.64%  0.91%
##    77:   2.00%  1.22%  3.06%  2.74%  2.59%  0.86%
##    78:   2.05%  1.22%  3.15%  2.74%  2.69%  0.95%
##    79:   2.02%  1.25%  3.02%  2.79%  2.64%  0.91%
##    80:   2.05%  1.22%  3.19%  2.84%  2.59%  0.91%
##    81:   2.06%  1.28%  3.15%  2.69%  2.74%  0.91%
##    82:   2.04%  1.22%  3.10%  2.89%  2.64%  0.86%
##    83:   2.03%  1.17%  3.19%  2.84%  2.54%  0.95%
##    84:   1.99%  1.20%  3.19%  2.74%  2.43%  0.86%
##    85:   2.03%  1.20%  3.19%  2.84%  2.59%  0.86%
##    86:   2.03%  1.22%  3.10%  2.89%  2.59%  0.86%
##    87:   2.01%  1.20%  3.02%  2.84%  2.64%  0.91%
##    88:   1.99%  1.17%  2.97%  2.84%  2.59%  0.91%
##    89:   1.98%  1.17%  3.02%  2.74%  2.54%  0.95%
##    90:   2.00%  1.17%  3.10%  2.84%  2.48%  0.95%
##    91:   2.05%  1.17%  3.19%  2.89%  2.59%  0.95%
##    92:   2.02%  1.20%  3.19%  2.74%  2.54%  0.95%
##    93:   2.00%  1.14%  3.10%  2.74%  2.59%  0.95%
##    94:   2.03%  1.20%  3.19%  2.79%  2.54%  0.95%
##    95:   2.01%  1.17%  3.02%  2.94%  2.54%  0.95%
##    96:   2.03%  1.20%  3.06%  2.94%  2.54%  0.95%
##    97:   2.02%  1.14%  3.15%  2.89%  2.54%  0.95%
##    98:   2.00%  1.20%  3.06%  2.74%  2.54%  0.95%
##    99:   2.00%  1.14%  3.15%  2.74%  2.59%  0.95%
##   100:   2.00%  1.17%  3.19%  2.74%  2.48%  0.95%
## ntree      OOB      1      2      3      4      5
##     1:  11.12%  7.04% 14.55% 13.76% 12.66%  9.86%
##     2:  10.22%  7.01% 13.55% 13.69% 10.79%  7.95%
##     3:  10.17%  6.94% 13.00% 13.80% 10.55%  8.41%
##     4:   9.35%  6.44% 11.66% 12.90%  9.64%  7.85%
##     5:   8.80%  5.68% 11.88% 12.02%  9.27%  6.95%
##     6:   8.23%  5.11% 10.47% 11.73%  8.46%  7.25%
##     7:   7.34%  4.79%  9.94% 10.13%  7.65%  5.68%
##     8:   6.94%  4.38%  9.01% 10.19%  7.11%  5.56%
##     9:   6.29%  4.26%  8.26%  9.58%  6.18%  4.42%
##    10:   5.51%  3.42%  7.42%  8.30%  5.77%  3.96%
##    11:   4.89%  3.25%  6.25%  7.60%  5.17%  3.26%
##    12:   4.72%  2.76%  6.62%  7.16%  4.68%  3.52%
##    13:   4.26%  2.79%  6.10%  6.09%  4.36%  2.83%
##    14:   3.86%  2.43%  5.23%  5.79%  4.05%  2.69%
##    15:   3.63%  2.16%  5.31%  5.54%  3.94%  2.10%
##    16:   3.20%  1.92%  4.14%  5.33%  3.27%  2.14%
##    17:   3.18%  1.83%  4.19%  4.92%  3.21%  2.55%
##    18:   2.84%  1.56%  4.40%  4.07%  3.01%  1.87%
##    19:   2.67%  1.44%  3.97%  3.77%  3.01%  1.87%
##    20:   2.51%  1.20%  4.10%  3.52%  2.85%  1.64%
##    21:   2.49%  1.35%  3.92%  3.47%  2.64%  1.69%
##    22:   2.34%  1.35%  3.54%  3.11%  2.64%  1.59%
##    23:   2.22%  1.17%  3.41%  3.11%  2.54%  1.50%
##    24:   2.20%  1.17%  3.28%  2.96%  2.75%  1.46%
##    25:   2.17%  1.14%  3.54%  2.86%  2.44%  1.41%
##    26:   2.11%  1.17%  3.23%  2.81%  2.49%  1.41%
##    27:   2.05%  1.20%  3.28%  2.81%  2.33%  1.09%
##    28:   1.99%  1.23%  2.93%  2.51%  2.44%  1.28%
##    29:   1.94%  1.02%  3.06%  2.61%  2.44%  1.09%
##    30:   1.86%  0.96%  3.06%  2.41%  2.18%  1.18%
##    31:   1.89%  0.90%  2.93%  2.66%  2.38%  1.14%
##    32:   1.90%  1.05%  3.10%  2.46%  2.28%  1.09%
##    33:   1.86%  1.08%  2.93%  2.41%  2.28%  1.05%
##    34:   1.84%  0.99%  3.02%  2.21%  2.33%  1.14%
##    35:   1.79%  0.99%  2.80%  2.41%  2.38%  0.87%
##    36:   1.75%  0.99%  2.72%  2.21%  2.33%  0.96%
##    37:   1.69%  0.99%  2.59%  2.06%  2.23%  1.00%
##    38:   1.69%  1.02%  2.59%  2.21%  2.23%  0.82%
##    39:   1.66%  1.08%  2.41%  1.96%  2.28%  0.91%
##    40:   1.66%  0.90%  2.46%  2.21%  2.28%  0.96%
##    41:   1.68%  1.08%  2.50%  2.26%  2.18%  0.77%
##    42:   1.66%  0.99%  2.50%  2.16%  2.18%  0.87%
##    43:   1.60%  0.96%  2.41%  2.01%  2.23%  0.77%
##    44:   1.59%  0.87%  2.41%  1.96%  2.18%  0.96%
##    45:   1.60%  0.96%  2.29%  2.06%  2.18%  0.91%
##    46:   1.59%  0.93%  2.20%  2.11%  2.23%  0.91%
##    47:   1.52%  0.84%  2.11%  2.16%  2.07%  0.87%
##    48:   1.52%  0.93%  2.11%  2.01%  2.07%  0.87%
##    49:   1.49%  0.90%  2.07%  1.91%  2.13%  0.87%
##    50:   1.44%  0.78%  2.03%  1.86%  2.02%  0.96%
##    51:   1.45%  0.84%  2.03%  1.81%  2.02%  0.96%
##    52:   1.44%  0.69%  2.16%  1.76%  2.18%  0.91%
##    53:   1.44%  0.69%  2.20%  1.81%  2.13%  0.87%
##    54:   1.41%  0.66%  2.16%  1.81%  2.07%  0.82%
##    55:   1.38%  0.60%  1.98%  1.91%  2.07%  0.82%
##    56:   1.39%  0.57%  2.07%  1.76%  2.13%  0.96%
##    57:   1.38%  0.63%  2.03%  1.71%  2.07%  0.96%
##    58:   1.40%  0.60%  2.20%  1.76%  2.07%  0.87%
##    59:   1.42%  0.57%  2.11%  2.06%  2.13%  0.77%
##    60:   1.44%  0.60%  2.20%  2.06%  2.07%  0.82%
##    61:   1.42%  0.57%  2.11%  1.96%  2.13%  0.87%
##    62:   1.46%  0.69%  2.03%  2.01%  2.23%  0.87%
##    63:   1.44%  0.63%  2.20%  1.91%  2.13%  0.82%
##    64:   1.39%  0.63%  2.07%  1.81%  2.07%  0.87%
##    65:   1.42%  0.69%  2.07%  1.86%  2.18%  0.77%
##    66:   1.41%  0.69%  2.07%  1.86%  2.13%  0.77%
##    67:   1.34%  0.60%  2.03%  1.71%  2.13%  0.73%
##    68:   1.36%  0.60%  2.11%  1.71%  2.07%  0.77%
##    69:   1.34%  0.60%  1.94%  1.76%  2.13%  0.77%
##    70:   1.33%  0.63%  1.85%  1.66%  2.13%  0.87%
##    71:   1.28%  0.54%  1.85%  1.76%  1.87%  0.87%
##    72:   1.29%  0.57%  1.98%  1.61%  1.87%  0.87%
##    73:   1.35%  0.63%  1.94%  1.71%  2.02%  0.91%
##    74:   1.29%  0.60%  1.98%  1.51%  1.92%  0.87%
##    75:   1.29%  0.60%  2.03%  1.51%  1.87%  0.87%
##    76:   1.33%  0.57%  2.20%  1.56%  1.92%  0.87%
##    77:   1.34%  0.63%  2.07%  1.71%  1.87%  0.87%
##    78:   1.28%  0.54%  1.98%  1.61%  1.87%  0.87%
##    79:   1.29%  0.51%  2.03%  1.61%  1.87%  0.91%
##    80:   1.29%  0.51%  2.03%  1.56%  1.92%  0.91%
##    81:   1.28%  0.54%  1.98%  1.61%  1.92%  0.82%
##    82:   1.29%  0.60%  2.03%  1.46%  1.92%  0.87%
##    83:   1.28%  0.51%  1.98%  1.56%  1.97%  0.87%
##    84:   1.31%  0.60%  1.98%  1.71%  1.87%  0.82%
##    85:   1.29%  0.57%  2.03%  1.56%  1.87%  0.87%
##    86:   1.28%  0.63%  1.98%  1.66%  1.81%  0.73%
##    87:   1.32%  0.63%  2.03%  1.71%  1.92%  0.73%
##    88:   1.31%  0.57%  2.16%  1.56%  1.81%  0.87%
##    89:   1.30%  0.60%  1.98%  1.61%  1.97%  0.77%
##    90:   1.33%  0.60%  2.11%  1.66%  1.87%  0.87%
##    91:   1.34%  0.60%  2.16%  1.61%  1.92%  0.87%
##    92:   1.34%  0.60%  2.20%  1.56%  1.92%  0.87%
##    93:   1.31%  0.57%  2.16%  1.51%  1.92%  0.82%
##    94:   1.34%  0.57%  2.24%  1.56%  1.92%  0.87%
##    95:   1.32%  0.57%  2.20%  1.51%  1.92%  0.87%
##    96:   1.34%  0.60%  2.16%  1.61%  1.97%  0.82%
##    97:   1.31%  0.57%  2.20%  1.46%  1.92%  0.82%
##    98:   1.30%  0.57%  2.16%  1.51%  1.87%  0.82%
##    99:   1.32%  0.54%  2.20%  1.61%  1.92%  0.82%
##   100:   1.34%  0.57%  2.20%  1.56%  2.07%  0.77%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   8.28%  3.98% 13.02% 12.42%  7.03%  6.83%
##     2:   7.87%  4.54% 11.47% 11.59%  7.02%  6.51%
##     3:   7.72%  4.81%  9.73% 11.76%  7.91%  6.17%
##     4:   7.27%  4.99%  9.10% 10.84%  6.84%  5.97%
##     5:   6.82%  4.33%  9.81%  9.39%  6.47%  5.43%
##     6:   6.18%  3.61%  8.81%  8.70%  6.23%  4.99%
##     7:   5.72%  3.53%  8.47%  8.01%  5.23%  4.50%
##     8:   5.45%  3.56%  7.69%  8.28%  4.65%  4.09%
##     9:   4.96%  3.04%  7.26%  7.64%  4.65%  3.32%
##    10:   4.74%  2.87%  6.66%  7.02%  4.61%  3.58%
##    11:   4.39%  2.71%  6.52%  6.43%  3.97%  3.20%
##    12:   4.17%  2.68%  5.81%  6.66%  3.96%  2.65%
##    13:   4.02%  2.31%  5.79%  6.40%  4.11%  2.51%
##    14:   3.59%  2.16%  5.10%  5.63%  3.90%  2.05%
##    15:   3.44%  1.83%  5.01%  5.63%  3.58%  2.14%
##    16:   3.15%  1.98%  4.45%  4.52%  3.68%  1.87%
##    17:   3.06%  1.65%  4.36%  4.37%  3.89%  1.91%
##    18:   3.01%  1.77%  4.23%  4.37%  3.78%  1.68%
##    19:   2.90%  1.83%  4.06%  4.27%  3.42%  1.64%
##    20:   2.85%  1.89%  4.23%  4.07%  3.37%  1.28%
##    21:   2.68%  1.86%  3.84%  3.92%  3.11%  1.23%
##    22:   2.60%  1.80%  3.49%  3.92%  3.16%  1.18%
##    23:   2.63%  1.68%  3.88%  3.97%  3.01%  1.23%
##    24:   2.59%  1.77%  3.62%  3.92%  3.06%  1.14%
##    25:   2.54%  1.71%  3.54%  3.77%  3.01%  1.23%
##    26:   2.47%  1.59%  3.41%  3.92%  2.85%  1.18%
##    27:   2.45%  1.62%  3.41%  3.67%  2.95%  1.18%
##    28:   2.34%  1.56%  3.23%  3.37%  2.90%  1.14%
##    29:   2.39%  1.56%  3.19%  3.47%  3.06%  1.28%
##    30:   2.43%  1.59%  3.28%  3.47%  3.11%  1.28%
##    31:   2.22%  1.44%  3.02%  3.26%  2.75%  1.14%
##    32:   2.27%  1.50%  3.15%  3.21%  2.95%  1.05%
##    33:   2.29%  1.50%  2.89%  3.37%  3.01%  1.28%
##    34:   2.14%  1.38%  2.89%  2.96%  2.95%  1.05%
##    35:   2.08%  1.38%  2.89%  2.96%  2.70%  0.96%
##    36:   2.00%  1.23%  3.02%  2.66%  2.70%  0.91%
##    37:   1.99%  1.32%  3.02%  2.61%  2.59%  0.82%
##    38:   1.95%  1.23%  2.93%  2.71%  2.54%  0.82%
##    39:   2.00%  1.23%  2.98%  2.96%  2.54%  0.77%
##    40:   2.03%  1.20%  3.10%  2.91%  2.49%  0.96%
##    41:   1.99%  1.20%  2.80%  2.96%  2.54%  0.96%
##    42:   1.99%  1.11%  2.85%  2.96%  2.64%  0.96%
##    43:   2.02%  1.17%  2.93%  2.96%  2.70%  0.91%
##    44:   2.01%  1.20%  2.98%  2.96%  2.54%  0.91%
##    45:   2.06%  1.08%  3.10%  3.16%  2.70%  0.87%
##    46:   2.04%  1.20%  2.85%  3.11%  2.75%  0.87%
##    47:   1.95%  1.05%  2.85%  2.96%  2.59%  0.91%
##    48:   1.93%  1.14%  2.72%  3.01%  2.44%  0.87%
##    49:   1.89%  1.05%  2.72%  2.91%  2.44%  0.87%
##    50:   1.84%  1.17%  2.54%  2.61%  2.49%  0.87%
##    51:   1.95%  1.14%  2.89%  2.81%  2.54%  0.91%
##    52:   1.89%  1.20%  2.59%  2.86%  2.49%  0.82%
##    53:   1.89%  1.11%  2.59%  2.91%  2.54%  0.82%
##    54:   1.91%  1.11%  2.67%  2.91%  2.59%  0.82%
##    55:   1.86%  1.08%  2.67%  2.81%  2.49%  0.77%
##    56:   1.89%  1.11%  2.59%  2.86%  2.59%  0.82%
##    57:   1.83%  1.08%  2.50%  2.71%  2.54%  0.82%
##    58:   1.81%  1.08%  2.54%  2.66%  2.54%  0.73%
##    59:   1.76%  1.02%  2.37%  2.66%  2.49%  0.77%
##    60:   1.78%  1.08%  2.41%  2.76%  2.44%  0.73%
##    61:   1.78%  0.99%  2.37%  2.86%  2.59%  0.68%
##    62:   1.76%  1.02%  2.37%  2.76%  2.49%  0.68%
##    63:   1.82%  1.02%  2.50%  2.76%  2.59%  0.77%
##    64:   1.82%  1.02%  2.54%  2.76%  2.54%  0.77%
##    65:   1.76%  1.02%  2.41%  2.71%  2.49%  0.68%
##    66:   1.80%  1.08%  2.54%  2.61%  2.54%  0.73%
##    67:   1.78%  1.08%  2.54%  2.66%  2.44%  0.68%
##    68:   1.85%  1.14%  2.59%  2.86%  2.44%  0.73%
##    69:   1.80%  1.11%  2.54%  2.66%  2.49%  0.68%
##    70:   1.77%  1.05%  2.59%  2.56%  2.49%  0.68%
##    71:   1.77%  1.14%  2.63%  2.36%  2.49%  0.64%
##    72:   1.77%  1.11%  2.63%  2.36%  2.54%  0.64%
##    73:   1.74%  1.11%  2.50%  2.46%  2.44%  0.64%
##    74:   1.72%  1.11%  2.46%  2.46%  2.38%  0.59%
##    75:   1.72%  1.11%  2.41%  2.41%  2.49%  0.64%
##    76:   1.75%  1.11%  2.41%  2.51%  2.44%  0.73%
##    77:   1.73%  1.11%  2.50%  2.36%  2.44%  0.68%
##    78:   1.71%  1.02%  2.46%  2.51%  2.38%  0.64%
##    79:   1.70%  0.99%  2.46%  2.36%  2.49%  0.68%
##    80:   1.73%  1.11%  2.50%  2.41%  2.49%  0.59%
##    81:   1.70%  1.11%  2.50%  2.26%  2.33%  0.68%
##    82:   1.71%  1.02%  2.59%  2.46%  2.23%  0.68%
##    83:   1.66%  1.08%  2.59%  2.11%  2.18%  0.68%
##    84:   1.72%  1.08%  2.50%  2.46%  2.23%  0.73%
##    85:   1.65%  1.05%  2.41%  2.16%  2.28%  0.73%
##    86:   1.65%  0.99%  2.46%  2.31%  2.18%  0.73%
##    87:   1.66%  0.96%  2.41%  2.31%  2.33%  0.73%
##    88:   1.68%  0.99%  2.50%  2.31%  2.38%  0.68%
##    89:   1.63%  1.05%  2.33%  2.16%  2.33%  0.68%
##    90:   1.64%  1.11%  2.37%  2.16%  2.23%  0.68%
##    91:   1.62%  1.02%  2.37%  2.21%  2.23%  0.68%
##    92:   1.61%  1.05%  2.29%  2.11%  2.33%  0.68%
##    93:   1.64%  1.05%  2.33%  2.16%  2.38%  0.68%
##    94:   1.66%  1.05%  2.46%  2.11%  2.44%  0.68%
##    95:   1.62%  0.99%  2.37%  2.01%  2.49%  0.68%
##    96:   1.63%  1.05%  2.41%  2.11%  2.28%  0.68%
##    97:   1.60%  1.02%  2.33%  2.06%  2.28%  0.73%
##    98:   1.59%  1.05%  2.29%  2.01%  2.28%  0.68%
##    99:   1.61%  1.05%  2.37%  2.01%  2.33%  0.68%
##   100:   1.59%  1.11%  2.29%  1.91%  2.23%  0.73%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   9.08%  5.62% 10.99% 12.82% 11.41%  6.80%
##     2:   8.68%  5.36% 10.77% 12.24%  9.52%  7.47%
##     3:   8.46%  5.50% 10.60% 12.18%  8.45%  7.25%
##     4:   8.15%  4.95% 10.27% 12.37%  7.50%  7.45%
##     5:   7.60%  4.87%  9.93% 10.89%  7.00%  6.77%
##     6:   6.97%  4.56%  8.54%  9.77%  6.95%  6.45%
##     7:   6.32%  3.97%  7.93%  9.22%  6.59%  5.30%
##     8:   5.88%  3.98%  7.23%  8.58%  6.09%  4.71%
##     9:   5.18%  3.40%  6.72%  7.96%  4.87%  3.97%
##    10:   4.91%  3.31%  6.13%  7.35%  5.00%  3.77%
##    11:   4.45%  3.08%  5.81%  6.61%  4.29%  3.26%
##    12:   4.26%  2.98%  5.63%  6.55%  4.06%  2.84%
##    13:   4.16%  2.94%  5.62%  6.24%  4.00%  2.70%
##    14:   3.83%  2.64%  5.22%  5.78%  3.63%  2.55%
##    15:   3.82%  2.55%  5.48%  5.93%  3.42%  2.46%
##    16:   3.70%  2.61%  4.75%  5.93%  3.63%  2.28%
##    17:   3.51%  2.30%  4.66%  5.83%  3.42%  2.10%
##    18:   3.38%  2.21%  4.61%  5.32%  3.47%  2.00%
##    19:   3.28%  2.07%  4.66%  5.12%  3.37%  1.91%
##    20:   3.09%  2.07%  4.14%  5.07%  3.06%  1.78%
##    21:   3.01%  1.92%  4.10%  4.87%  2.85%  1.96%
##    22:   2.92%  1.83%  4.05%  4.72%  2.75%  1.91%
##    23:   2.61%  1.68%  3.54%  4.07%  2.64%  1.68%
##    24:   2.52%  1.65%  3.36%  3.77%  2.80%  1.59%
##    25:   2.50%  1.80%  3.23%  3.72%  2.80%  1.41%
##    26:   2.52%  1.80%  3.32%  3.92%  2.54%  1.50%
##    27:   2.48%  1.65%  3.49%  3.77%  2.75%  1.28%
##    28:   2.46%  1.53%  3.45%  3.72%  2.75%  1.46%
##    29:   2.42%  1.56%  3.32%  3.57%  2.75%  1.46%
##    30:   2.33%  1.47%  3.41%  3.37%  2.75%  1.18%
##    31:   2.29%  1.56%  3.23%  3.16%  2.75%  1.23%
##    32:   2.22%  1.35%  3.23%  3.16%  2.70%  1.23%
##    33:   2.18%  1.38%  3.02%  3.21%  2.75%  1.09%
##    34:   2.15%  1.59%  2.67%  3.26%  2.70%  0.96%
##    35:   2.06%  1.47%  2.89%  3.01%  2.44%  0.87%
##    36:   2.10%  1.38%  2.93%  3.11%  2.59%  0.96%
##    37:   2.07%  1.41%  2.89%  3.06%  2.44%  1.00%
##    38:   2.06%  1.38%  2.89%  3.31%  2.33%  0.82%
##    39:   2.02%  1.38%  2.93%  3.06%  2.49%  0.68%
##    40:   1.98%  1.35%  2.72%  3.37%  2.28%  0.64%
##    41:   1.94%  1.32%  2.63%  3.01%  2.33%  0.82%
##    42:   1.97%  1.41%  2.67%  3.06%  2.18%  0.91%
##    43:   1.99%  1.47%  2.89%  2.86%  2.23%  0.82%
##    44:   1.93%  1.44%  2.59%  2.96%  2.18%  0.82%
##    45:   1.94%  1.53%  2.59%  2.96%  2.23%  0.73%
##    46:   1.88%  1.32%  2.41%  3.06%  2.23%  0.77%
##    47:   1.99%  1.47%  2.63%  3.01%  2.33%  0.87%
##    48:   1.92%  1.44%  2.67%  2.91%  2.13%  0.77%
##    49:   1.93%  1.32%  2.76%  2.81%  2.33%  0.82%
##    50:   1.92%  1.38%  2.72%  2.81%  2.23%  0.82%
##    51:   1.94%  1.32%  2.80%  2.96%  2.18%  0.82%
##    52:   1.94%  1.38%  2.89%  2.86%  2.18%  0.77%
##    53:   1.89%  1.35%  2.85%  2.71%  2.13%  0.73%
##    54:   1.88%  1.26%  2.72%  2.91%  2.18%  0.73%
##    55:   1.88%  1.32%  2.63%  2.91%  2.28%  0.64%
##    56:   1.80%  1.32%  2.50%  2.76%  2.13%  0.64%
##    57:   1.78%  1.32%  2.41%  2.71%  2.13%  0.68%
##    58:   1.83%  1.29%  2.59%  2.61%  2.33%  0.73%
##    59:   1.86%  1.32%  2.54%  2.81%  2.18%  0.82%
##    60:   1.83%  1.32%  2.37%  2.81%  2.18%  0.82%
##    61:   1.85%  1.32%  2.59%  2.81%  2.13%  0.77%
##    62:   1.87%  1.32%  2.67%  2.76%  2.13%  0.82%
##    63:   1.89%  1.41%  2.50%  2.86%  2.18%  0.87%
##    64:   1.83%  1.32%  2.37%  2.81%  2.18%  0.87%
##    65:   1.85%  1.29%  2.46%  2.81%  2.18%  0.91%
##    66:   1.83%  1.26%  2.41%  2.81%  2.13%  0.96%
##    67:   1.82%  1.26%  2.46%  2.71%  2.18%  0.87%
##    68:   1.77%  1.20%  2.41%  2.71%  2.07%  0.87%
##    69:   1.79%  1.17%  2.41%  2.71%  2.23%  0.87%
##    70:   1.82%  1.23%  2.37%  2.81%  2.23%  0.87%
##    71:   1.79%  1.17%  2.33%  2.71%  2.28%  0.91%
##    72:   1.78%  1.23%  2.24%  2.81%  2.23%  0.82%
##    73:   1.77%  1.23%  2.33%  2.61%  2.28%  0.77%
##    74:   1.80%  1.29%  2.41%  2.66%  2.18%  0.82%
##    75:   1.80%  1.26%  2.37%  2.71%  2.23%  0.82%
##    76:   1.82%  1.29%  2.33%  2.76%  2.23%  0.87%
##    77:   1.83%  1.32%  2.33%  2.86%  2.23%  0.77%
##    78:   1.86%  1.32%  2.37%  2.86%  2.33%  0.82%
##    79:   1.88%  1.35%  2.46%  2.86%  2.33%  0.77%
##    80:   1.85%  1.29%  2.33%  2.91%  2.28%  0.87%
##    81:   1.84%  1.32%  2.20%  3.01%  2.28%  0.82%
##    82:   1.78%  1.29%  2.24%  2.76%  2.28%  0.73%
##    83:   1.78%  1.26%  2.29%  2.76%  2.33%  0.68%
##    84:   1.77%  1.26%  2.24%  2.66%  2.33%  0.73%
##    85:   1.74%  1.23%  2.24%  2.66%  2.28%  0.68%
##    86:   1.82%  1.26%  2.33%  2.76%  2.38%  0.77%
##    87:   1.79%  1.29%  2.24%  2.61%  2.44%  0.77%
##    88:   1.77%  1.29%  2.20%  2.61%  2.38%  0.77%
##    89:   1.73%  1.20%  2.24%  2.46%  2.38%  0.77%
##    90:   1.70%  1.20%  2.16%  2.56%  2.33%  0.64%
##    91:   1.73%  1.26%  2.24%  2.46%  2.38%  0.68%
##    92:   1.74%  1.26%  2.29%  2.51%  2.33%  0.68%
##    93:   1.70%  1.23%  2.29%  2.41%  2.28%  0.64%
##    94:   1.72%  1.26%  2.33%  2.36%  2.28%  0.68%
##    95:   1.72%  1.26%  2.20%  2.46%  2.33%  0.68%
##    96:   1.69%  1.20%  2.24%  2.41%  2.28%  0.68%
##    97:   1.73%  1.26%  2.33%  2.46%  2.33%  0.64%
##    98:   1.70%  1.26%  2.24%  2.31%  2.38%  0.64%
##    99:   1.68%  1.20%  2.24%  2.31%  2.38%  0.64%
##   100:   1.69%  1.20%  2.37%  2.26%  2.28%  0.68%
## ntree      OOB      1      2      3      4      5
##     1:  10.38%  7.06% 12.01% 13.21% 11.97%  9.69%
##     2:  10.42%  6.60% 12.92% 13.70% 12.22%  8.96%
##     3:  10.17%  5.99% 14.38% 13.00% 10.96%  8.80%
##     4:   9.53%  5.74% 12.71% 12.25% 10.63%  8.39%
##     5:   9.01%  5.47% 11.52% 11.12% 10.83%  8.11%
##     6:   8.14%  4.79% 10.29% 11.03%  8.91%  7.49%
##     7:   7.42%  3.96% 10.32%  9.26%  8.52%  6.88%
##     8:   6.61%  4.11%  9.00%  8.69%  7.59%  4.95%
##     9:   6.10%  3.85%  8.09%  7.89%  7.35%  4.56%
##    10:   5.52%  3.41%  7.73%  7.10%  6.48%  4.01%
##    11:   4.96%  2.61%  7.70%  6.17%  5.73%  3.75%
##    12:   4.71%  2.73%  7.10%  6.11%  5.58%  3.02%
##    13:   4.62%  2.81%  6.51%  5.63%  5.66%  3.40%
##    14:   4.26%  2.72%  6.19%  5.30%  5.09%  2.77%
##    15:   4.00%  2.60%  5.70%  4.92%  4.88%  2.63%
##    16:   3.60%  2.18%  5.52%  4.36%  4.10%  2.53%
##    17:   3.49%  2.09%  5.43%  4.17%  4.15%  2.28%
##    18:   3.42%  2.09%  5.43%  4.22%  4.00%  1.99%
##    19:   3.04%  1.70%  4.81%  3.56%  3.64%  2.14%
##    20:   3.00%  1.79%  5.07%  3.46%  3.54%  1.65%
##    21:   3.07%  1.94%  4.99%  3.65%  3.38%  1.90%
##    22:   2.82%  1.76%  4.46%  3.37%  3.38%  1.60%
##    23:   2.77%  1.82%  4.29%  3.46%  3.23%  1.46%
##    24:   2.64%  1.64%  4.11%  3.37%  3.03%  1.51%
##    25:   2.60%  1.73%  4.07%  3.04%  2.97%  1.56%
##    26:   2.49%  1.67%  3.72%  3.14%  2.87%  1.41%
##    27:   2.43%  1.70%  3.67%  2.76%  2.92%  1.41%
##    28:   2.47%  1.67%  3.67%  2.85%  2.87%  1.65%
##    29:   2.39%  1.58%  3.32%  3.04%  2.82%  1.56%
##    30:   2.31%  1.58%  3.41%  2.71%  2.77%  1.41%
##    31:   2.14%  1.46%  3.15%  2.71%  2.56%  1.12%
##    32:   2.07%  1.20%  2.97%  2.81%  2.77%  1.07%
##    33:   2.02%  1.17%  3.11%  2.85%  2.46%  0.92%
##    34:   1.99%  1.23%  3.06%  2.53%  2.56%  0.92%
##    35:   1.93%  1.17%  2.93%  2.76%  2.46%  0.68%
##    36:   1.85%  1.26%  2.71%  2.43%  2.46%  0.68%
##    37:   1.77%  1.05%  2.67%  2.43%  2.46%  0.63%
##    38:   1.73%  1.05%  2.84%  2.06%  2.41%  0.63%
##    39:   1.74%  1.02%  2.71%  2.15%  2.51%  0.68%
##    40:   1.69%  1.02%  2.67%  2.01%  2.36%  0.73%
##    41:   1.74%  1.11%  2.58%  2.25%  2.56%  0.53%
##    42:   1.72%  1.05%  2.76%  2.01%  2.36%  0.73%
##    43:   1.72%  1.11%  2.93%  1.97%  2.26%  0.63%
##    44:   1.68%  1.14%  2.76%  1.97%  2.26%  0.53%
##    45:   1.65%  1.11%  2.58%  1.92%  2.36%  0.53%
##    46:   1.63%  1.08%  2.41%  1.92%  2.41%  0.63%
##    47:   1.55%  1.05%  2.23%  2.06%  2.15%  0.53%
##    48:   1.57%  1.20%  2.27%  1.92%  2.20%  0.44%
##    49:   1.55%  1.20%  2.23%  1.82%  2.20%  0.49%
##    50:   1.59%  1.14%  2.36%  1.97%  2.15%  0.53%
##    51:   1.53%  1.11%  2.23%  1.73%  2.31%  0.49%
##    52:   1.61%  1.20%  2.36%  1.78%  2.31%  0.63%
##    53:   1.49%  1.11%  2.23%  1.64%  2.20%  0.44%
##    54:   1.44%  1.14%  2.10%  1.64%  2.15%  0.34%
##    55:   1.43%  1.05%  2.14%  1.59%  2.15%  0.39%
##    56:   1.48%  1.17%  2.10%  1.68%  2.26%  0.34%
##    57:   1.43%  1.14%  2.06%  1.59%  2.10%  0.39%
##    58:   1.42%  1.14%  1.97%  1.64%  2.05%  0.44%
##    59:   1.50%  1.17%  2.23%  1.68%  2.10%  0.49%
##    60:   1.44%  1.05%  2.27%  1.64%  2.05%  0.39%
##    61:   1.49%  1.20%  2.06%  1.73%  2.10%  0.49%
##    62:   1.47%  1.08%  2.10%  1.78%  2.10%  0.49%
##    63:   1.52%  1.11%  2.23%  1.78%  2.15%  0.53%
##    64:   1.44%  1.11%  2.06%  1.59%  2.15%  0.44%
##    65:   1.46%  1.17%  1.97%  1.73%  2.10%  0.49%
##    66:   1.44%  1.14%  2.06%  1.64%  2.05%  0.49%
##    67:   1.39%  1.02%  2.01%  1.68%  2.00%  0.44%
##    68:   1.39%  0.99%  1.92%  1.68%  2.10%  0.49%
##    69:   1.42%  1.05%  1.97%  1.68%  2.15%  0.44%
##    70:   1.40%  0.96%  2.06%  1.64%  2.10%  0.49%
##    71:   1.34%  0.99%  1.97%  1.54%  2.00%  0.39%
##    72:   1.38%  0.99%  2.01%  1.68%  2.05%  0.39%
##    73:   1.40%  1.08%  1.97%  1.59%  2.15%  0.39%
##    74:   1.40%  1.02%  2.06%  1.59%  2.10%  0.44%
##    75:   1.40%  1.05%  2.01%  1.64%  2.05%  0.44%
##    76:   1.38%  1.08%  1.92%  1.54%  2.00%  0.49%
##    77:   1.38%  1.02%  1.92%  1.64%  2.05%  0.49%
##    78:   1.39%  0.93%  2.01%  1.73%  2.00%  0.53%
##    79:   1.44%  0.99%  2.19%  1.73%  1.95%  0.53%
##    80:   1.44%  0.99%  2.06%  1.78%  2.05%  0.58%
##    81:   1.44%  0.99%  2.14%  1.78%  1.95%  0.53%
##    82:   1.44%  1.02%  2.19%  1.68%  2.00%  0.49%
##    83:   1.44%  1.02%  2.14%  1.68%  2.00%  0.58%
##    84:   1.41%  1.02%  2.10%  1.59%  2.00%  0.53%
##    85:   1.44%  0.93%  2.10%  1.64%  2.15%  0.63%
##    86:   1.45%  0.99%  2.19%  1.68%  2.15%  0.49%
##    87:   1.39%  0.99%  2.01%  1.59%  2.00%  0.58%
##    88:   1.44%  0.99%  2.06%  1.68%  2.10%  0.58%
##    89:   1.38%  0.96%  1.97%  1.59%  2.05%  0.58%
##    90:   1.39%  0.99%  1.97%  1.50%  2.10%  0.63%
##    91:   1.43%  1.02%  2.01%  1.64%  2.10%  0.58%
##    92:   1.42%  0.96%  2.06%  1.59%  2.10%  0.63%
##    93:   1.41%  0.99%  2.06%  1.64%  2.00%  0.58%
##    94:   1.38%  0.99%  1.88%  1.64%  2.00%  0.58%
##    95:   1.38%  0.99%  1.92%  1.64%  2.05%  0.53%
##    96:   1.37%  0.99%  1.88%  1.54%  1.95%  0.68%
##    97:   1.35%  0.99%  1.88%  1.64%  1.90%  0.53%
##    98:   1.34%  0.96%  1.92%  1.59%  1.90%  0.53%
##    99:   1.35%  0.99%  1.84%  1.59%  1.95%  0.58%
##   100:   1.39%  0.99%  2.01%  1.59%  2.00%  0.58%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   8.33%  5.56% 11.00% 10.84%  9.99%  5.56%
##     2:   7.95%  4.86% 11.76% 10.24%  9.37%  4.92%
##     3:   7.38%  4.24% 10.57%  9.11%  8.36%  6.20%
##     4:   7.23%  4.39% 10.90%  8.67%  8.10%  5.50%
##     5:   6.79%  4.31%  9.73%  8.41%  7.76%  4.96%
##     6:   6.31%  3.99%  8.99%  7.62%  7.61%  4.50%
##     7:   5.60%  3.09%  8.25%  7.26%  6.98%  3.67%
##     8:   5.33%  3.31%  7.28%  7.12%  6.42%  3.56%
##     9:   4.73%  2.83%  6.56%  5.96%  6.05%  3.27%
##    10:   4.36%  2.57%  6.65%  4.89%  5.43%  3.19%
##    11:   4.22%  2.65%  6.13%  4.96%  5.01%  3.12%
##    12:   4.04%  2.26%  5.76%  5.08%  5.04%  2.97%
##    13:   3.75%  2.28%  5.09%  4.56%  4.87%  2.73%
##    14:   3.51%  2.31%  4.86%  4.27%  4.57%  2.19%
##    15:   3.32%  2.06%  4.77%  3.79%  4.25%  2.38%
##    16:   3.24%  2.21%  4.72%  3.23%  4.31%  2.24%
##    17:   3.04%  2.00%  4.46%  3.09%  4.10%  2.09%
##    18:   2.90%  1.70%  4.51%  3.04%  3.69%  2.19%
##    19:   2.91%  1.67%  4.46%  3.23%  4.00%  1.85%
##    20:   2.79%  1.67%  4.11%  3.14%  3.74%  1.90%
##    21:   2.81%  1.67%  4.29%  2.76%  3.84%  2.09%
##    22:   2.68%  1.49%  3.85%  2.71%  4.00%  2.04%
##    23:   2.64%  1.43%  3.98%  2.76%  3.74%  1.94%
##    24:   2.55%  1.46%  3.72%  2.76%  3.54%  1.85%
##    25:   2.67%  1.46%  3.94%  2.76%  3.95%  1.94%
##    26:   2.54%  1.49%  3.76%  2.57%  3.64%  1.80%
##    27:   2.56%  1.49%  3.67%  2.62%  3.69%  1.90%
##    28:   2.46%  1.38%  3.46%  2.43%  3.74%  1.94%
##    29:   2.45%  1.52%  3.24%  2.53%  3.64%  1.90%
##    30:   2.45%  1.35%  3.19%  2.81%  3.74%  1.85%
##    31:   2.45%  1.35%  3.41%  2.62%  3.69%  1.85%
##    32:   2.43%  1.38%  3.11%  2.71%  3.69%  1.90%
##    33:   2.33%  1.41%  3.02%  2.53%  3.49%  1.75%
##    34:   2.28%  1.23%  2.89%  2.43%  3.59%  1.90%
##    35:   2.28%  1.23%  3.06%  2.29%  3.59%  1.85%
##    36:   2.37%  1.38%  3.15%  2.43%  3.69%  1.80%
##    37:   2.19%  1.11%  2.84%  2.29%  3.59%  1.80%
##    38:   2.31%  1.32%  2.97%  2.39%  3.64%  1.85%
##    39:   2.23%  1.17%  3.02%  2.29%  3.54%  1.80%
##    40:   2.12%  1.14%  2.80%  2.15%  3.43%  1.70%
##    41:   2.14%  1.14%  2.71%  2.29%  3.43%  1.75%
##    42:   2.07%  1.02%  2.80%  2.20%  3.38%  1.60%
##    43:   2.09%  1.05%  2.84%  2.34%  3.18%  1.65%
##    44:   2.10%  1.11%  2.89%  2.25%  3.18%  1.65%
##    45:   2.06%  1.14%  2.67%  2.20%  3.28%  1.60%
##    46:   2.08%  1.14%  2.84%  2.11%  3.33%  1.56%
##    47:   2.08%  1.08%  2.93%  2.01%  3.28%  1.70%
##    48:   2.08%  1.11%  2.71%  2.29%  3.18%  1.70%
##    49:   2.05%  1.08%  2.58%  2.25%  3.28%  1.65%
##    50:   1.97%  1.05%  2.67%  2.01%  3.08%  1.60%
##    51:   1.93%  1.02%  2.54%  1.97%  3.08%  1.60%
##    52:   1.91%  1.11%  2.49%  1.87%  2.97%  1.60%
##    53:   1.96%  1.11%  2.54%  1.92%  3.13%  1.65%
##    54:   1.93%  1.08%  2.58%  2.01%  2.92%  1.56%
##    55:   1.94%  1.11%  2.58%  2.01%  2.97%  1.56%
##    56:   1.91%  1.08%  2.54%  1.92%  2.97%  1.56%
##    57:   1.90%  1.08%  2.62%  1.87%  2.92%  1.51%
##    58:   1.93%  1.11%  2.58%  1.92%  3.08%  1.46%
##    59:   1.94%  1.11%  2.71%  1.97%  2.97%  1.41%
##    60:   1.94%  1.11%  2.67%  1.92%  2.97%  1.51%
##    61:   1.96%  1.11%  2.76%  2.01%  3.02%  1.41%
##    62:   1.95%  1.05%  2.76%  2.01%  3.08%  1.41%
##    63:   1.98%  1.08%  2.71%  1.97%  3.13%  1.56%
##    64:   1.94%  1.11%  2.84%  1.87%  2.92%  1.41%
##    65:   1.95%  1.11%  2.84%  1.92%  2.97%  1.41%
##    66:   1.96%  1.08%  2.80%  1.92%  3.08%  1.46%
##    67:   1.94%  1.05%  2.71%  2.01%  2.97%  1.51%
##    68:   1.93%  1.02%  2.71%  1.97%  3.02%  1.46%
##    69:   1.96%  1.08%  2.71%  1.92%  2.97%  1.65%
##    70:   1.94%  1.08%  2.80%  1.92%  2.92%  1.51%
##    71:   1.92%  1.05%  2.62%  1.92%  2.97%  1.56%
##    72:   2.00%  1.11%  2.76%  1.97%  3.02%  1.65%
##    73:   1.98%  1.05%  2.76%  2.01%  3.08%  1.56%
##    74:   1.97%  1.02%  2.76%  2.01%  2.97%  1.65%
##    75:   1.94%  0.96%  2.76%  1.97%  3.02%  1.56%
##    76:   1.89%  0.96%  2.62%  1.92%  3.02%  1.51%
##    77:   1.92%  1.02%  2.62%  1.92%  3.08%  1.51%
##    78:   1.91%  1.02%  2.54%  1.92%  3.02%  1.60%
##    79:   1.89%  0.99%  2.67%  1.92%  2.97%  1.46%
##    80:   1.85%  1.02%  2.49%  1.82%  2.92%  1.51%
##    81:   1.92%  1.08%  2.49%  2.06%  2.92%  1.56%
##    82:   1.83%  1.02%  2.36%  1.87%  2.97%  1.46%
##    83:   1.86%  0.99%  2.49%  2.01%  2.92%  1.41%
##    84:   1.89%  1.02%  2.58%  1.92%  2.97%  1.46%
##    85:   1.92%  1.08%  2.67%  1.87%  2.97%  1.51%
##    86:   1.90%  1.08%  2.58%  1.87%  2.97%  1.51%
##    87:   1.87%  1.02%  2.58%  1.87%  2.87%  1.51%
##    88:   1.87%  1.05%  2.58%  1.87%  2.87%  1.46%
##    89:   1.88%  1.05%  2.58%  1.97%  2.82%  1.46%
##    90:   1.83%  1.05%  2.41%  1.92%  2.77%  1.51%
##    91:   1.89%  1.11%  2.54%  1.92%  2.82%  1.51%
##    92:   1.85%  1.05%  2.49%  1.82%  2.87%  1.51%
##    93:   1.83%  1.02%  2.49%  1.82%  2.82%  1.46%
##    94:   1.82%  1.02%  2.41%  1.82%  2.87%  1.46%
##    95:   1.83%  0.99%  2.49%  1.87%  2.87%  1.41%
##    96:   1.83%  0.96%  2.54%  1.82%  2.92%  1.46%
##    97:   1.88%  0.96%  2.62%  1.87%  3.02%  1.46%
##    98:   1.92%  1.02%  2.67%  1.87%  3.02%  1.56%
##    99:   1.89%  0.99%  2.54%  1.92%  3.02%  1.51%
##   100:   1.90%  0.99%  2.62%  1.92%  3.02%  1.51%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   6.63%  5.69%  8.52%  8.36%  5.27%  5.43%
##     2:   7.62%  6.18% 10.34%  9.27%  6.83%  6.06%
##     3:   7.21%  5.23%  9.48%  9.42%  7.20%  5.62%
##     4:   7.26%  5.52%  9.39%  9.13%  7.35%  5.70%
##     5:   6.85%  5.13%  8.95%  8.18%  7.67%  5.20%
##     6:   6.37%  4.56%  8.47%  8.21%  6.61%  4.82%
##     7:   5.87%  4.38%  7.94%  7.09%  6.26%  4.36%
##     8:   5.46%  3.89%  7.43%  6.49%  6.07%  4.20%
##     9:   4.77%  3.22%  6.65%  5.51%  5.69%  3.57%
##    10:   4.56%  3.20%  6.00%  5.30%  5.52%  3.49%
##    11:   4.40%  3.22%  5.85%  4.76%  5.37%  3.43%
##    12:   4.07%  3.24%  5.04%  4.28%  5.00%  3.22%
##    13:   3.86%  3.03%  4.65%  4.32%  4.58%  3.16%
##    14:   3.50%  2.64%  4.25%  3.84%  4.42%  2.87%
##    15:   3.39%  2.45%  4.42%  3.75%  4.16%  2.67%
##    16:   3.16%  2.12%  4.33%  3.65%  4.00%  2.24%
##    17:   3.07%  2.21%  3.85%  3.46%  3.79%  2.48%
##    18:   2.90%  2.00%  3.67%  3.04%  4.21%  2.14%
##    19:   2.85%  1.97%  3.89%  2.99%  3.69%  2.14%
##    20:   2.72%  2.06%  3.76%  2.62%  3.64%  1.85%
##    21:   2.63%  1.79%  3.81%  2.53%  3.49%  1.99%
##    22:   2.66%  1.79%  3.67%  2.76%  3.69%  1.85%
##    23:   2.67%  1.91%  3.67%  2.71%  3.43%  1.99%
##    24:   2.54%  1.82%  3.54%  2.39%  3.64%  1.70%
##    25:   2.54%  1.79%  3.46%  2.25%  3.69%  1.94%
##    26:   2.51%  1.85%  3.67%  2.25%  3.28%  1.85%
##    27:   2.36%  1.64%  3.37%  2.11%  3.38%  1.70%
##    28:   2.38%  1.55%  3.37%  2.15%  3.43%  1.85%
##    29:   2.38%  1.67%  3.24%  2.29%  3.43%  1.65%
##    30:   2.42%  1.67%  3.46%  2.06%  3.43%  1.90%
##    31:   2.25%  1.58%  3.24%  1.87%  3.08%  1.85%
##    32:   2.34%  1.64%  3.46%  2.06%  3.28%  1.60%
##    33:   2.23%  1.46%  3.28%  1.97%  3.28%  1.60%
##    34:   2.22%  1.38%  3.19%  1.92%  3.43%  1.70%
##    35:   2.18%  1.35%  3.28%  1.97%  3.23%  1.56%
##    36:   2.20%  1.29%  3.32%  2.01%  3.33%  1.56%
##    37:   2.17%  1.32%  3.28%  1.92%  3.18%  1.65%
##    38:   2.13%  1.43%  3.11%  1.87%  3.23%  1.41%
##    39:   2.09%  1.43%  3.02%  1.82%  3.18%  1.36%
##    40:   2.11%  1.38%  2.93%  1.87%  3.28%  1.51%
##    41:   2.09%  1.43%  3.06%  1.82%  3.02%  1.46%
##    42:   2.02%  1.41%  2.89%  1.82%  2.92%  1.41%
##    43:   2.08%  1.35%  2.93%  2.06%  3.02%  1.46%
##    44:   2.03%  1.41%  2.89%  1.82%  2.97%  1.41%
##    45:   2.00%  1.49%  2.80%  1.82%  2.82%  1.36%
##    46:   2.06%  1.52%  2.76%  1.87%  2.92%  1.51%
##    47:   2.00%  1.41%  2.58%  1.97%  2.92%  1.51%
##    48:   1.97%  1.38%  2.76%  1.97%  2.77%  1.31%
##    49:   1.94%  1.41%  2.49%  1.97%  2.77%  1.36%
##    50:   1.96%  1.41%  2.71%  1.82%  2.77%  1.41%
##    51:   1.92%  1.43%  2.49%  1.92%  2.72%  1.31%
##    52:   1.92%  1.41%  2.41%  2.06%  2.77%  1.26%
##    53:   1.93%  1.43%  2.54%  1.92%  2.72%  1.31%
##    54:   1.85%  1.38%  2.36%  1.87%  2.72%  1.22%
##    55:   1.87%  1.35%  2.45%  1.92%  2.72%  1.22%
##    56:   1.86%  1.32%  2.45%  1.92%  2.77%  1.17%
##    57:   1.88%  1.35%  2.49%  2.01%  2.61%  1.22%
##    58:   1.87%  1.41%  2.41%  1.97%  2.67%  1.17%
##    59:   1.88%  1.35%  2.45%  1.97%  2.61%  1.31%
##    60:   1.88%  1.32%  2.54%  1.97%  2.67%  1.22%
##    61:   1.90%  1.38%  2.41%  2.06%  2.72%  1.26%
##    62:   1.94%  1.38%  2.54%  2.06%  2.72%  1.31%
##    63:   1.87%  1.32%  2.45%  1.92%  2.67%  1.31%
##    64:   1.92%  1.38%  2.49%  2.06%  2.72%  1.26%
##    65:   1.97%  1.38%  2.58%  2.06%  2.82%  1.36%
##    66:   1.89%  1.32%  2.41%  1.87%  2.92%  1.31%
##    67:   1.88%  1.35%  2.36%  1.82%  2.82%  1.36%
##    68:   1.90%  1.38%  2.45%  1.92%  2.87%  1.22%
##    69:   1.85%  1.38%  2.41%  1.73%  2.87%  1.17%
##    70:   1.88%  1.35%  2.45%  1.87%  2.82%  1.22%
##    71:   1.84%  1.38%  2.27%  1.92%  2.77%  1.17%
##    72:   1.91%  1.35%  2.45%  1.97%  2.77%  1.36%
##    73:   1.86%  1.29%  2.41%  1.87%  2.72%  1.36%
##    74:   1.92%  1.32%  2.58%  2.01%  2.77%  1.26%
##    75:   1.89%  1.32%  2.41%  2.01%  2.77%  1.26%
##    76:   1.88%  1.32%  2.36%  2.01%  2.77%  1.26%
##    77:   1.89%  1.26%  2.45%  2.06%  2.82%  1.22%
##    78:   1.82%  1.29%  2.36%  1.97%  2.72%  1.07%
##    79:   1.85%  1.29%  2.45%  1.92%  2.77%  1.17%
##    80:   1.91%  1.38%  2.49%  1.97%  2.82%  1.22%
##    81:   1.84%  1.26%  2.36%  1.87%  2.77%  1.31%
##    82:   1.80%  1.17%  2.41%  1.82%  2.72%  1.26%
##    83:   1.75%  1.08%  2.32%  1.82%  2.67%  1.26%
##    84:   1.81%  1.20%  2.41%  1.87%  2.61%  1.31%
##    85:   1.78%  1.17%  2.32%  1.92%  2.61%  1.26%
##    86:   1.82%  1.23%  2.36%  1.92%  2.61%  1.31%
##    87:   1.77%  1.17%  2.27%  1.97%  2.61%  1.17%
##    88:   1.81%  1.17%  2.36%  1.87%  2.72%  1.31%
##    89:   1.82%  1.23%  2.32%  1.92%  2.67%  1.31%
##    90:   1.80%  1.11%  2.32%  1.92%  2.67%  1.41%
##    91:   1.79%  1.17%  2.19%  1.92%  2.67%  1.41%
##    92:   1.78%  1.11%  2.23%  1.97%  2.72%  1.31%
##    93:   1.79%  1.14%  2.32%  1.97%  2.67%  1.26%
##    94:   1.78%  1.14%  2.36%  1.78%  2.67%  1.36%
##    95:   1.82%  1.23%  2.36%  1.82%  2.67%  1.36%
##    96:   1.81%  1.26%  2.32%  1.92%  2.61%  1.26%
##    97:   1.80%  1.23%  2.23%  1.92%  2.67%  1.31%
##    98:   1.77%  1.20%  2.23%  1.87%  2.61%  1.26%
##    99:   1.76%  1.14%  2.14%  1.97%  2.61%  1.31%
##   100:   1.75%  1.14%  2.19%  1.87%  2.61%  1.31%
## ntree      OOB      1      2      3      4      5
##     1:   9.23%  5.08% 13.10% 12.79%  9.18%  8.04%
##     2:   9.79%  5.19% 13.29% 12.54% 11.18%  9.08%
##     3:   9.37%  5.27% 12.20% 12.54%  9.98%  8.95%
##     4:   8.66%  5.34% 10.89% 11.63%  8.85%  8.32%
##     5:   8.09%  5.07% 10.13% 10.70%  9.03%  7.29%
##     6:   7.48%  4.61%  9.57%  9.99%  8.29%  6.55%
##     7:   7.05%  4.26%  8.56%  9.72%  8.57%  5.84%
##     8:   6.37%  3.52%  8.31%  8.31%  7.49%  5.82%
##     9:   5.52%  3.55%  6.73%  7.28%  6.57%  4.64%
##    10:   5.33%  3.20%  6.77%  7.41%  5.84%  4.62%
##    11:   4.98%  3.21%  6.23%  6.59%  5.56%  4.28%
##    12:   4.53%  2.91%  6.09%  6.43%  4.96%  3.16%
##    13:   4.01%  2.69%  5.50%  4.99%  4.17%  3.34%
##    14:   3.58%  2.15%  5.03%  4.54%  4.11%  2.83%
##    15:   3.22%  2.30%  4.60%  4.49%  2.90%  2.23%
##    16:   3.20%  1.96%  4.30%  4.48%  3.47%  2.41%
##    17:   2.80%  1.69%  3.75%  3.99%  3.16%  2.04%
##    18:   2.85%  1.66%  4.05%  3.89%  3.00%  2.22%
##    19:   2.55%  1.54%  3.50%  3.50%  2.84%  1.90%
##    20:   2.57%  1.60%  3.50%  3.35%  2.84%  2.09%
##    21:   2.43%  1.57%  3.29%  3.15%  2.73%  1.85%
##    22:   2.45%  1.51%  3.42%  3.45%  2.63%  1.71%
##    23:   2.32%  1.39%  3.21%  3.30%  2.63%  1.58%
##    24:   2.34%  1.42%  3.42%  3.15%  2.63%  1.58%
##    25:   2.26%  1.57%  3.04%  2.86%  2.63%  1.58%
##    26:   2.12%  1.33%  2.78%  2.61%  2.68%  1.67%
##    27:   2.18%  1.39%  2.91%  2.96%  2.58%  1.53%
##    28:   1.97%  1.27%  2.74%  2.61%  2.31%  1.30%
##    29:   1.98%  1.36%  2.74%  2.36%  2.26%  1.48%
##    30:   1.94%  1.24%  2.70%  2.56%  2.21%  1.34%
##    31:   1.95%  1.36%  2.61%  2.32%  2.37%  1.44%
##    32:   1.91%  1.30%  2.57%  2.41%  2.37%  1.25%
##    33:   1.89%  1.18%  2.61%  2.56%  2.16%  1.30%
##    34:   1.83%  1.21%  2.32%  2.36%  2.26%  1.34%
##    35:   1.83%  1.21%  2.19%  2.61%  2.26%  1.25%
##    36:   1.72%  1.24%  2.02%  2.36%  2.16%  1.16%
##    37:   1.69%  1.18%  2.02%  2.27%  2.26%  1.07%
##    38:   1.66%  1.18%  1.98%  2.36%  2.05%  1.02%
##    39:   1.64%  1.06%  2.02%  2.22%  2.21%  1.07%
##    40:   1.64%  1.00%  2.11%  2.17%  2.31%  1.02%
##    41:   1.61%  1.09%  2.11%  1.92%  2.26%  1.02%
##    42:   1.54%  1.00%  2.15%  1.72%  2.16%  0.97%
##    43:   1.56%  1.15%  1.94%  1.82%  2.16%  1.02%
##    44:   1.58%  1.03%  2.19%  1.67%  2.31%  1.02%
##    45:   1.55%  1.12%  2.15%  1.63%  2.10%  0.97%
##    46:   1.55%  1.06%  2.02%  1.63%  2.31%  1.02%
##    47:   1.49%  1.06%  1.90%  1.72%  2.00%  1.02%
##    48:   1.52%  1.00%  1.86%  1.67%  2.31%  1.11%
##    49:   1.50%  1.03%  1.81%  1.67%  2.31%  1.02%
##    50:   1.41%  0.94%  1.77%  1.58%  2.16%  0.93%
##    51:   1.40%  0.97%  1.69%  1.63%  2.21%  0.83%
##    52:   1.42%  0.94%  1.73%  1.72%  2.16%  0.88%
##    53:   1.40%  0.97%  1.86%  1.58%  2.00%  0.88%
##    54:   1.38%  0.94%  1.77%  1.63%  2.00%  0.88%
##    55:   1.40%  0.87%  1.94%  1.43%  2.05%  1.02%
##    56:   1.41%  0.87%  1.94%  1.58%  2.05%  0.93%
##    57:   1.43%  0.90%  2.02%  1.58%  1.89%  1.02%
##    58:   1.39%  0.94%  1.86%  1.58%  1.89%  0.97%
##    59:   1.39%  0.90%  1.98%  1.43%  1.95%  0.97%
##    60:   1.35%  0.84%  1.98%  1.53%  1.79%  0.88%
##    61:   1.32%  0.75%  1.86%  1.63%  1.89%  0.79%
##    62:   1.39%  0.84%  1.94%  1.53%  1.95%  1.02%
##    63:   1.32%  0.69%  1.90%  1.63%  2.00%  0.79%
##    64:   1.30%  0.69%  1.86%  1.48%  1.95%  0.88%
##    65:   1.32%  0.75%  1.86%  1.48%  1.89%  0.93%
##    66:   1.31%  0.78%  1.90%  1.43%  1.89%  0.83%
##    67:   1.34%  0.81%  2.02%  1.43%  1.89%  0.83%
##    68:   1.26%  0.72%  1.77%  1.53%  1.84%  0.74%
##    69:   1.28%  0.75%  1.77%  1.48%  1.89%  0.83%
##    70:   1.29%  0.75%  1.77%  1.53%  1.89%  0.83%
##    71:   1.29%  0.72%  2.07%  1.33%  1.95%  0.70%
##    72:   1.27%  0.66%  1.90%  1.28%  2.05%  0.83%
##    73:   1.27%  0.63%  1.94%  1.38%  2.00%  0.74%
##    74:   1.24%  0.66%  1.81%  1.33%  1.95%  0.79%
##    75:   1.23%  0.51%  1.98%  1.33%  1.95%  0.79%
##    76:   1.24%  0.60%  1.90%  1.33%  1.95%  0.79%
##    77:   1.27%  0.63%  1.98%  1.33%  1.95%  0.83%
##    78:   1.27%  0.63%  1.86%  1.38%  2.00%  0.83%
##    79:   1.24%  0.63%  1.86%  1.33%  1.95%  0.79%
##    80:   1.25%  0.60%  1.81%  1.38%  2.00%  0.83%
##    81:   1.24%  0.57%  1.81%  1.38%  1.95%  0.88%
##    82:   1.25%  0.66%  1.69%  1.38%  2.00%  0.88%
##    83:   1.27%  0.63%  1.77%  1.43%  2.00%  0.88%
##    84:   1.29%  0.60%  1.86%  1.43%  2.00%  0.97%
##    85:   1.21%  0.51%  1.73%  1.28%  1.95%  0.97%
##    86:   1.22%  0.57%  1.81%  1.33%  1.79%  0.97%
##    87:   1.27%  0.57%  1.86%  1.43%  1.89%  0.97%
##    88:   1.23%  0.60%  1.81%  1.33%  1.89%  0.88%
##    89:   1.27%  0.57%  1.90%  1.38%  1.95%  0.97%
##    90:   1.26%  0.57%  1.90%  1.38%  1.84%  0.97%
##    91:   1.26%  0.54%  1.98%  1.28%  1.89%  0.97%
##    92:   1.24%  0.57%  1.94%  1.28%  1.84%  0.93%
##    93:   1.29%  0.57%  2.02%  1.43%  1.95%  0.88%
##    94:   1.27%  0.57%  1.98%  1.43%  1.95%  0.83%
##    95:   1.29%  0.57%  1.94%  1.53%  1.84%  0.97%
##    96:   1.25%  0.57%  1.90%  1.38%  1.89%  0.88%
##    97:   1.27%  0.57%  1.94%  1.43%  1.89%  0.93%
##    98:   1.27%  0.54%  2.07%  1.33%  1.89%  0.93%
##    99:   1.23%  0.51%  1.94%  1.28%  1.89%  0.93%
##   100:   1.19%  0.45%  1.81%  1.33%  1.84%  0.93%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.77%  4.46% 10.59% 11.84%  7.13%  6.38%
##     2:   7.24%  4.33%  9.57%  9.54%  8.10%  6.17%
##     3:   6.97%  4.37%  9.03% 10.08%  7.77%  5.11%
##     4:   6.91%  4.43%  9.06% 10.60%  7.18%  4.69%
##     5:   6.49%  4.16%  8.43%  8.89%  7.71%  4.62%
##     6:   6.32%  4.45%  8.02%  9.21%  7.22%  3.80%
##     7:   5.73%  3.87%  7.41%  8.27%  6.38%  3.77%
##     8:   5.57%  3.95%  7.39%  7.60%  6.28%  3.51%
##     9:   5.10%  3.29%  7.09%  6.77%  6.08%  3.24%
##    10:   4.76%  3.15%  6.33%  6.78%  5.62%  2.86%
##    11:   4.60%  3.08%  6.51%  5.96%  5.45%  2.80%
##    12:   4.30%  2.89%  5.30%  6.00%  5.32%  2.89%
##    13:   3.88%  2.54%  5.17%  5.19%  5.00%  2.28%
##    14:   3.79%  2.42%  5.20%  4.59%  5.10%  2.41%
##    15:   3.45%  2.36%  4.27%  4.34%  4.94%  2.09%
##    16:   3.38%  2.05%  4.44%  4.64%  4.52%  2.09%
##    17:   3.16%  1.72%  4.22%  4.24%  4.47%  2.04%
##    18:   3.07%  1.87%  4.01%  3.99%  4.10%  2.09%
##    19:   2.86%  1.75%  3.75%  3.65%  4.10%  1.76%
##    20:   2.73%  1.51%  3.63%  3.65%  3.94%  1.67%
##    21:   2.80%  1.51%  3.67%  3.79%  4.31%  1.58%
##    22:   2.56%  1.54%  3.12%  3.60%  3.73%  1.48%
##    23:   2.56%  1.51%  3.12%  3.55%  3.73%  1.58%
##    24:   2.54%  1.54%  3.04%  3.45%  3.63%  1.71%
##    25:   2.43%  1.39%  3.08%  3.25%  3.36%  1.71%
##    26:   2.45%  1.33%  3.08%  3.45%  3.58%  1.58%
##    27:   2.49%  1.39%  3.16%  3.30%  3.73%  1.58%
##    28:   2.45%  1.42%  3.29%  3.10%  3.63%  1.48%
##    29:   2.47%  1.36%  3.21%  3.30%  3.63%  1.58%
##    30:   2.43%  1.39%  3.12%  3.35%  3.47%  1.48%
##    31:   2.41%  1.45%  3.12%  3.15%  3.42%  1.53%
##    32:   2.34%  1.21%  3.12%  3.10%  3.42%  1.53%
##    33:   2.27%  1.15%  3.21%  3.10%  3.15%  1.39%
##    34:   2.29%  1.09%  3.12%  3.30%  3.36%  1.34%
##    35:   2.32%  1.15%  3.25%  3.20%  3.26%  1.44%
##    36:   2.25%  1.15%  3.16%  3.05%  3.21%  1.34%
##    37:   2.27%  1.12%  3.16%  3.20%  3.21%  1.34%
##    38:   2.26%  1.12%  3.16%  3.15%  3.21%  1.34%
##    39:   2.25%  1.12%  3.21%  3.15%  3.15%  1.30%
##    40:   2.19%  1.15%  3.08%  3.05%  3.00%  1.30%
##    41:   2.21%  1.21%  2.95%  3.15%  3.10%  1.25%
##    42:   2.26%  1.15%  3.12%  3.20%  3.21%  1.30%
##    43:   2.13%  1.06%  2.91%  3.00%  3.10%  1.25%
##    44:   2.12%  1.12%  2.91%  3.10%  2.84%  1.25%
##    45:   2.12%  1.09%  3.04%  2.96%  2.84%  1.30%
##    46:   2.15%  1.03%  2.95%  3.10%  3.00%  1.34%
##    47:   2.17%  1.06%  3.08%  3.00%  2.94%  1.39%
##    48:   2.07%  1.00%  2.99%  2.91%  2.94%  1.16%
##    49:   2.11%  1.03%  2.95%  2.91%  3.10%  1.25%
##    50:   2.10%  1.06%  2.95%  2.81%  3.10%  1.20%
##    51:   2.13%  1.12%  2.87%  2.91%  3.10%  1.30%
##    52:   2.07%  1.00%  2.74%  2.86%  3.21%  1.25%
##    53:   2.05%  0.81%  2.74%  3.00%  3.15%  1.30%
##    54:   2.12%  1.03%  2.78%  2.91%  3.26%  1.34%
##    55:   2.09%  0.90%  2.78%  2.91%  3.15%  1.44%
##    56:   2.06%  0.81%  2.83%  2.81%  3.21%  1.44%
##    57:   2.05%  0.90%  2.66%  2.91%  3.21%  1.30%
##    58:   2.09%  0.90%  2.83%  2.91%  3.15%  1.39%
##    59:   2.10%  1.03%  2.87%  2.81%  3.05%  1.39%
##    60:   2.08%  0.90%  2.91%  2.76%  3.15%  1.39%
##    61:   2.06%  0.90%  2.83%  2.66%  3.31%  1.30%
##    62:   2.11%  1.03%  2.83%  2.71%  3.31%  1.34%
##    63:   2.08%  0.94%  2.87%  2.71%  3.31%  1.30%
##    64:   2.06%  0.97%  2.87%  2.71%  3.21%  1.25%
##    65:   2.05%  0.97%  2.83%  2.66%  3.26%  1.20%
##    66:   2.09%  1.00%  2.87%  2.71%  3.21%  1.34%
##    67:   2.09%  1.03%  2.87%  2.76%  3.21%  1.25%
##    68:   2.07%  0.94%  2.91%  2.81%  3.15%  1.25%
##    69:   2.08%  1.00%  2.83%  2.71%  3.21%  1.34%
##    70:   2.07%  0.97%  2.87%  2.81%  3.15%  1.25%
##    71:   2.06%  0.97%  2.83%  2.71%  3.26%  1.25%
##    72:   2.11%  1.00%  2.91%  2.81%  3.31%  1.25%
##    73:   2.09%  1.03%  2.83%  2.66%  3.36%  1.25%
##    74:   2.11%  0.97%  2.87%  2.76%  3.36%  1.34%
##    75:   2.06%  1.03%  2.66%  2.61%  3.31%  1.39%
##    76:   2.10%  0.94%  2.87%  2.66%  3.36%  1.39%
##    77:   2.09%  0.90%  2.83%  2.76%  3.36%  1.34%
##    78:   2.07%  0.94%  2.78%  2.76%  3.31%  1.30%
##    79:   2.06%  0.84%  2.74%  2.71%  3.36%  1.44%
##    80:   2.10%  0.87%  2.83%  2.71%  3.42%  1.44%
##    81:   2.06%  0.84%  2.74%  2.71%  3.31%  1.44%
##    82:   2.06%  0.94%  2.74%  2.51%  3.36%  1.44%
##    83:   2.04%  0.97%  2.83%  2.56%  3.15%  1.34%
##    84:   1.97%  0.84%  2.70%  2.51%  3.10%  1.39%
##    85:   2.00%  0.90%  2.74%  2.56%  3.21%  1.30%
##    86:   1.99%  0.94%  2.74%  2.61%  3.05%  1.25%
##    87:   1.97%  0.87%  2.70%  2.56%  3.15%  1.25%
##    88:   1.96%  0.90%  2.66%  2.46%  3.15%  1.30%
##    89:   1.96%  0.94%  2.70%  2.46%  3.00%  1.34%
##    90:   1.93%  0.90%  2.61%  2.51%  2.94%  1.30%
##    91:   1.95%  1.00%  2.61%  2.41%  3.00%  1.34%
##    92:   1.91%  1.03%  2.53%  2.41%  2.84%  1.30%
##    93:   1.93%  1.03%  2.53%  2.46%  2.94%  1.25%
##    94:   1.92%  1.00%  2.61%  2.46%  2.89%  1.20%
##    95:   1.94%  1.06%  2.61%  2.46%  2.84%  1.30%
##    96:   1.93%  1.03%  2.57%  2.41%  3.00%  1.20%
##    97:   1.92%  1.00%  2.53%  2.41%  2.94%  1.30%
##    98:   1.86%  0.87%  2.57%  2.36%  2.84%  1.25%
##    99:   1.83%  0.90%  2.49%  2.46%  2.68%  1.20%
##   100:   1.82%  0.81%  2.49%  2.46%  2.73%  1.20%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   6.87%  5.46%  7.67%  9.57%  6.68%  5.80%
##     2:   6.87%  5.24%  8.43% 10.14%  6.62%  4.87%
##     3:   6.59%  4.68%  8.47%  8.82%  7.11%  4.89%
##     4:   6.44%  4.49%  7.91%  8.73%  6.96%  5.18%
##     5:   6.36%  4.44%  8.00%  8.92%  6.50%  4.96%
##     6:   5.93%  4.15%  8.03%  7.94%  5.93%  4.43%
##     7:   5.33%  3.04%  7.52%  7.07%  6.57%  3.73%
##     8:   5.12%  3.19%  6.80%  6.93%  5.99%  3.75%
##     9:   4.78%  2.86%  6.56%  6.85%  5.60%  3.10%
##    10:   4.44%  2.71%  6.22%  5.66%  5.40%  3.13%
##    11:   4.15%  2.86%  5.22%  5.70%  4.76%  2.98%
##    12:   3.79%  2.52%  5.09%  4.50%  4.70%  2.83%
##    13:   3.47%  2.42%  4.49%  4.64%  4.32%  2.09%
##    14:   3.28%  2.18%  4.44%  4.29%  3.79%  2.27%
##    15:   3.28%  2.23%  4.65%  4.14%  3.73%  2.18%
##    16:   3.03%  1.96%  4.01%  3.84%  3.84%  2.09%
##    17:   3.02%  1.99%  4.27%  3.70%  3.68%  2.04%
##    18:   3.02%  2.11%  4.26%  3.75%  3.68%  1.81%
##    19:   2.85%  1.78%  3.88%  3.70%  3.73%  1.81%
##    20:   2.71%  1.90%  3.54%  3.35%  3.58%  1.67%
##    21:   2.63%  1.72%  3.33%  3.25%  3.63%  1.81%
##    22:   2.61%  1.75%  3.12%  3.45%  3.36%  1.90%
##    23:   2.44%  1.63%  3.08%  3.05%  3.36%  1.58%
##    24:   2.37%  1.57%  3.08%  2.86%  3.21%  1.62%
##    25:   2.46%  1.54%  3.12%  3.25%  3.26%  1.71%
##    26:   2.45%  1.54%  3.08%  3.15%  3.26%  1.81%
##    27:   2.37%  1.51%  3.12%  3.05%  3.31%  1.39%
##    28:   2.37%  1.42%  3.21%  3.00%  3.47%  1.34%
##    29:   2.32%  1.42%  2.99%  2.91%  3.42%  1.44%
##    30:   2.23%  1.36%  3.04%  2.81%  3.15%  1.34%
##    31:   2.23%  1.24%  2.78%  3.05%  3.21%  1.53%
##    32:   2.22%  1.42%  2.78%  2.86%  3.26%  1.34%
##    33:   2.23%  1.45%  2.74%  2.81%  3.26%  1.44%
##    34:   2.28%  1.51%  2.83%  2.86%  3.31%  1.44%
##    35:   2.23%  1.48%  2.70%  2.91%  3.21%  1.39%
##    36:   2.26%  1.45%  2.78%  2.86%  3.47%  1.30%
##    37:   2.24%  1.33%  2.95%  2.76%  3.36%  1.39%
##    38:   2.22%  1.27%  2.87%  2.76%  3.42%  1.44%
##    39:   2.27%  1.33%  3.04%  2.71%  3.52%  1.34%
##    40:   2.19%  1.30%  2.91%  2.71%  3.31%  1.30%
##    41:   2.21%  1.39%  2.78%  2.71%  3.42%  1.30%
##    42:   2.22%  1.42%  2.78%  2.61%  3.52%  1.30%
##    43:   2.27%  1.30%  3.04%  2.71%  3.58%  1.34%
##    44:   2.18%  1.27%  2.99%  2.61%  3.36%  1.25%
##    45:   2.19%  1.33%  2.95%  2.71%  3.21%  1.30%
##    46:   2.12%  1.18%  2.91%  2.56%  3.21%  1.34%
##    47:   2.12%  1.12%  2.91%  2.56%  3.36%  1.30%
##    48:   2.18%  1.24%  2.91%  2.71%  3.36%  1.30%
##    49:   2.17%  1.27%  2.83%  2.71%  3.31%  1.30%
##    50:   2.17%  1.21%  3.12%  2.61%  3.21%  1.30%
##    51:   2.06%  1.15%  2.70%  2.56%  3.31%  1.20%
##    52:   2.02%  1.12%  2.74%  2.41%  3.10%  1.30%
##    53:   2.08%  1.15%  2.78%  2.56%  3.21%  1.30%
##    54:   2.06%  1.18%  2.74%  2.41%  3.15%  1.34%
##    55:   2.06%  1.21%  2.74%  2.41%  3.21%  1.30%
##    56:   2.10%  1.15%  2.78%  2.46%  3.36%  1.34%
##    57:   2.00%  1.09%  2.70%  2.32%  3.21%  1.30%
##    58:   2.01%  1.00%  2.66%  2.46%  3.26%  1.34%
##    59:   2.00%  1.09%  2.61%  2.51%  3.00%  1.34%
##    60:   2.01%  1.09%  2.70%  2.46%  3.10%  1.30%
##    61:   2.04%  1.06%  2.78%  2.51%  3.00%  1.44%
##    62:   2.08%  1.03%  2.74%  2.56%  3.36%  1.39%
##    63:   2.05%  1.03%  2.66%  2.51%  3.31%  1.39%
##    64:   2.04%  1.03%  2.70%  2.51%  3.26%  1.34%
##    65:   2.00%  1.03%  2.66%  2.36%  3.21%  1.34%
##    66:   2.02%  1.03%  2.66%  2.41%  3.26%  1.39%
##    67:   2.00%  1.06%  2.61%  2.46%  3.15%  1.30%
##    68:   1.95%  1.00%  2.53%  2.36%  3.21%  1.30%
##    69:   1.98%  1.06%  2.53%  2.36%  3.15%  1.39%
##    70:   1.93%  0.97%  2.53%  2.32%  3.00%  1.44%
##    71:   1.94%  1.03%  2.57%  2.36%  3.00%  1.34%
##    72:   1.96%  1.03%  2.57%  2.32%  3.15%  1.34%
##    73:   1.98%  1.09%  2.57%  2.32%  3.10%  1.39%
##    74:   1.96%  1.12%  2.53%  2.27%  3.00%  1.44%
##    75:   2.03%  1.06%  2.66%  2.27%  3.26%  1.53%
##    76:   2.03%  1.15%  2.57%  2.32%  3.10%  1.58%
##    77:   1.93%  1.06%  2.61%  2.17%  2.89%  1.44%
##    78:   1.97%  1.00%  2.61%  2.27%  3.10%  1.48%
##    79:   1.98%  1.12%  2.53%  2.27%  3.10%  1.44%
##    80:   2.00%  1.09%  2.57%  2.41%  3.05%  1.44%
##    81:   1.96%  1.12%  2.57%  2.27%  2.89%  1.48%
##    82:   2.00%  1.12%  2.57%  2.36%  3.05%  1.48%
##    83:   1.95%  1.00%  2.53%  2.41%  2.94%  1.48%
##    84:   1.96%  1.03%  2.57%  2.41%  3.00%  1.39%
##    85:   1.91%  1.00%  2.53%  2.27%  2.94%  1.39%
##    86:   1.94%  1.03%  2.61%  2.32%  3.05%  1.30%
##    87:   1.97%  1.06%  2.57%  2.32%  3.05%  1.44%
##    88:   1.93%  1.09%  2.57%  2.27%  2.89%  1.34%
##    89:   1.89%  0.94%  2.57%  2.27%  2.94%  1.30%
##    90:   1.90%  0.97%  2.61%  2.36%  2.89%  1.25%
##    91:   1.95%  1.00%  2.57%  2.41%  3.15%  1.25%
##    92:   1.94%  1.00%  2.57%  2.32%  3.10%  1.30%
##    93:   1.91%  0.94%  2.57%  2.36%  3.00%  1.30%
##    94:   1.89%  0.87%  2.57%  2.36%  3.05%  1.20%
##    95:   1.89%  0.90%  2.57%  2.36%  3.00%  1.25%
##    96:   1.88%  0.97%  2.57%  2.36%  2.89%  1.16%
##    97:   1.89%  0.97%  2.45%  2.41%  3.05%  1.20%
##    98:   1.90%  0.97%  2.49%  2.32%  3.26%  1.11%
##    99:   1.93%  0.97%  2.49%  2.36%  3.15%  1.30%
##   100:   1.92%  0.97%  2.57%  2.32%  3.05%  1.30%
## ntree      OOB      1      2      3      4      5
##     1:  10.66%  7.05% 13.83% 15.50%  9.49%  9.00%
##     2:  10.24%  6.64% 13.25% 13.66% 10.43%  8.81%
##     3:   9.55%  5.21% 12.59% 12.99% 10.55%  8.65%
##     4:   9.00%  5.19% 11.90% 12.94%  9.47%  7.41%
##     5:   8.85%  5.02% 11.94% 11.53%  9.96%  7.80%
##     6:   8.39%  4.94% 11.31% 10.58%  9.46%  7.50%
##     7:   7.63%  4.32% 10.82% 10.27%  8.65%  5.79%
##     8:   6.76%  3.97%  9.10%  9.16%  8.07%  5.00%
##     9:   6.32%  4.06%  9.12%  7.64%  7.16%  4.72%
##    10:   5.77%  3.75%  7.53%  8.07%  6.28%  4.29%
##    11:   5.14%  3.22%  7.17%  7.26%  5.81%  3.25%
##    12:   4.76%  2.97%  6.85%  6.46%  5.19%  3.20%
##    13:   4.43%  3.15%  5.81%  6.11%  4.69%  3.05%
##    14:   4.12%  2.70%  5.50%  5.95%  4.53%  2.71%
##    15:   3.81%  2.61%  5.36%  5.36%  4.03%  2.27%
##    16:   3.62%  2.49%  5.06%  5.26%  3.88%  1.98%
##    17:   3.48%  2.34%  4.88%  4.87%  3.88%  2.03%
##    18:   3.29%  2.16%  4.66%  4.67%  3.64%  1.88%
##    19:   3.11%  2.01%  4.40%  4.38%  3.44%  1.88%
##    20:   3.03%  1.83%  4.23%  4.52%  3.64%  1.59%
##    21:   2.99%  1.77%  4.36%  4.52%  3.19%  1.74%
##    22:   2.75%  1.74%  4.10%  3.93%  3.09%  1.40%
##    23:   2.75%  1.71%  4.10%  4.08%  3.04%  1.35%
##    24:   2.65%  1.59%  3.97%  3.93%  3.04%  1.25%
##    25:   2.46%  1.47%  3.84%  3.69%  2.59%  1.20%
##    26:   2.36%  1.53%  3.54%  3.34%  2.59%  1.20%
##    27:   2.34%  1.68%  3.45%  3.05%  2.69%  1.16%
##    28:   2.29%  1.62%  3.45%  3.29%  2.39%  1.01%
##    29:   2.26%  1.47%  3.50%  3.14%  2.49%  1.06%
##    30:   2.21%  1.41%  3.24%  3.00%  2.54%  1.25%
##    31:   2.21%  1.44%  3.37%  3.00%  2.44%  1.16%
##    32:   2.12%  1.38%  3.02%  2.95%  2.59%  1.06%
##    33:   2.11%  1.56%  3.02%  2.90%  2.24%  1.06%
##    34:   1.99%  1.38%  3.06%  2.65%  2.19%  0.92%
##    35:   2.01%  1.32%  3.06%  3.00%  2.09%  0.92%
##    36:   1.97%  1.41%  2.89%  2.60%  2.19%  1.01%
##    37:   1.99%  1.32%  3.06%  2.70%  2.04%  1.11%
##    38:   2.02%  1.41%  3.11%  2.51%  2.19%  1.16%
##    39:   1.94%  1.26%  3.06%  2.51%  2.14%  1.01%
##    40:   1.91%  1.20%  2.98%  2.70%  2.14%  0.87%
##    41:   1.82%  1.29%  2.76%  2.46%  1.94%  0.87%
##    42:   1.86%  1.14%  2.89%  2.51%  1.99%  1.11%
##    43:   1.75%  1.08%  2.68%  2.51%  1.99%  0.82%
##    44:   1.80%  1.14%  2.72%  2.51%  1.94%  1.01%
##    45:   1.77%  1.05%  2.76%  2.60%  1.94%  0.82%
##    46:   1.77%  1.14%  2.63%  2.51%  1.99%  0.92%
##    47:   1.74%  1.17%  2.68%  2.36%  1.84%  0.92%
##    48:   1.66%  1.11%  2.55%  2.31%  1.84%  0.77%
##    49:   1.75%  1.14%  2.63%  2.41%  2.04%  0.82%
##    50:   1.72%  1.05%  2.81%  2.31%  1.89%  0.87%
##    51:   1.71%  1.11%  2.59%  2.46%  1.74%  0.92%
##    52:   1.72%  0.96%  2.76%  2.56%  1.89%  0.77%
##    53:   1.74%  1.08%  2.85%  2.31%  1.84%  0.92%
##    54:   1.77%  1.08%  2.93%  2.51%  1.69%  0.92%
##    55:   1.73%  1.02%  2.85%  2.36%  1.84%  0.92%
##    56:   1.74%  0.96%  2.93%  2.41%  1.74%  1.01%
##    57:   1.74%  1.02%  2.89%  2.31%  1.84%  0.96%
##    58:   1.71%  0.96%  2.76%  2.46%  1.74%  0.96%
##    59:   1.73%  0.90%  2.89%  2.60%  1.74%  0.92%
##    60:   1.68%  0.90%  2.85%  2.31%  1.74%  0.96%
##    61:   1.70%  0.93%  2.72%  2.46%  1.89%  0.87%
##    62:   1.70%  0.96%  2.68%  2.31%  2.04%  0.87%
##    63:   1.65%  0.96%  2.50%  2.41%  1.89%  0.82%
##    64:   1.66%  0.96%  2.55%  2.46%  1.94%  0.77%
##    65:   1.62%  0.87%  2.68%  2.31%  1.84%  0.77%
##    66:   1.66%  0.99%  2.68%  2.26%  1.89%  0.82%
##    67:   1.64%  0.87%  2.85%  2.26%  1.74%  0.82%
##    68:   1.58%  0.84%  2.59%  2.41%  1.64%  0.77%
##    69:   1.64%  0.81%  2.72%  2.51%  1.79%  0.77%
##    70:   1.61%  0.78%  2.68%  2.51%  1.74%  0.77%
##    71:   1.61%  0.87%  2.55%  2.46%  1.79%  0.77%
##    72:   1.60%  0.81%  2.50%  2.56%  1.69%  0.82%
##    73:   1.60%  0.84%  2.50%  2.46%  1.74%  0.87%
##    74:   1.59%  0.81%  2.55%  2.31%  1.79%  0.87%
##    75:   1.55%  0.78%  2.50%  2.36%  1.69%  0.82%
##    76:   1.57%  0.81%  2.46%  2.46%  1.69%  0.82%
##    77:   1.57%  0.87%  2.46%  2.41%  1.69%  0.77%
##    78:   1.55%  0.81%  2.37%  2.41%  1.69%  0.82%
##    79:   1.53%  0.87%  2.37%  2.31%  1.69%  0.72%
##    80:   1.56%  0.87%  2.50%  2.41%  1.64%  0.72%
##    81:   1.55%  0.84%  2.42%  2.36%  1.74%  0.72%
##    82:   1.52%  0.84%  2.37%  2.26%  1.69%  0.77%
##    83:   1.51%  0.81%  2.46%  2.21%  1.59%  0.82%
##    84:   1.52%  0.87%  2.50%  2.16%  1.64%  0.72%
##    85:   1.54%  0.81%  2.55%  2.31%  1.64%  0.72%
##    86:   1.48%  0.72%  2.50%  2.16%  1.64%  0.72%
##    87:   1.44%  0.69%  2.46%  2.11%  1.59%  0.72%
##    88:   1.47%  0.75%  2.42%  2.16%  1.59%  0.77%
##    89:   1.45%  0.75%  2.37%  2.11%  1.59%  0.77%
##    90:   1.47%  0.78%  2.37%  2.21%  1.59%  0.72%
##    91:   1.49%  0.75%  2.42%  2.21%  1.64%  0.77%
##    92:   1.47%  0.78%  2.33%  2.11%  1.69%  0.77%
##    93:   1.51%  0.75%  2.37%  2.36%  1.69%  0.77%
##    94:   1.44%  0.75%  2.33%  2.01%  1.69%  0.77%
##    95:   1.49%  0.78%  2.33%  2.21%  1.69%  0.82%
##    96:   1.46%  0.75%  2.24%  2.16%  1.69%  0.82%
##    97:   1.48%  0.72%  2.33%  2.26%  1.69%  0.77%
##    98:   1.48%  0.75%  2.42%  2.21%  1.64%  0.72%
##    99:   1.48%  0.75%  2.42%  2.16%  1.64%  0.77%
##   100:   1.48%  0.78%  2.42%  2.16%  1.64%  0.72%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   8.05%  5.11% 10.57% 10.39%  8.12%  7.41%
##     2:   8.08%  4.75% 11.24%  9.69%  8.75%  7.54%
##     3:   7.63%  4.71% 11.06%  9.20%  8.35%  6.19%
##     4:   7.43%  4.63% 10.21%  9.87%  7.40%  6.40%
##     5:   6.84%  4.83%  8.95%  8.99%  6.84%  5.59%
##     6:   6.31%  4.39%  8.39%  8.35%  6.80%  4.56%
##     7:   5.64%  3.59%  7.70%  7.07%  6.88%  4.01%
##     8:   5.36%  3.56%  7.75%  6.33%  6.39%  3.66%
##     9:   4.87%  3.40%  6.93%  5.84%  5.72%  3.18%
##    10:   4.60%  3.02%  6.38%  5.96%  5.39%  3.07%
##    11:   4.26%  2.74%  6.05%  5.45%  4.87%  2.96%
##    12:   3.91%  2.55%  5.72%  4.84%  4.47%  2.62%
##    13:   3.65%  2.16%  5.37%  4.64%  4.46%  2.37%
##    14:   3.52%  2.19%  5.02%  4.68%  4.10%  2.32%
##    15:   3.37%  2.16%  5.14%  4.48%  3.69%  1.93%
##    16:   3.32%  2.16%  4.62%  4.14%  4.24%  2.03%
##    17:   3.09%  1.89%  4.58%  3.74%  3.99%  1.83%
##    18:   2.94%  1.80%  4.32%  3.64%  3.59%  1.93%
##    19:   2.91%  1.80%  4.10%  3.93%  3.59%  1.73%
##    20:   2.82%  1.77%  4.19%  3.74%  3.24%  1.69%
##    21:   2.84%  1.83%  4.19%  3.69%  3.24%  1.73%
##    22:   2.75%  1.62%  4.23%  3.83%  3.09%  1.54%
##    23:   2.74%  1.83%  3.97%  3.69%  3.19%  1.49%
##    24:   2.68%  1.74%  3.76%  3.54%  3.34%  1.54%
##    25:   2.57%  1.80%  3.54%  3.39%  3.09%  1.45%
##    26:   2.54%  1.68%  3.58%  3.29%  3.14%  1.45%
##    27:   2.50%  1.59%  3.45%  3.19%  3.19%  1.54%
##    28:   2.43%  1.41%  3.54%  3.14%  3.14%  1.45%
##    29:   2.41%  1.47%  3.32%  3.24%  3.14%  1.40%
##    30:   2.29%  1.44%  3.15%  3.14%  2.94%  1.25%
##    31:   2.41%  1.47%  3.50%  3.24%  2.99%  1.35%
##    32:   2.35%  1.35%  3.45%  3.10%  3.14%  1.25%
##    33:   2.39%  1.41%  3.71%  3.10%  2.99%  1.25%
##    34:   2.27%  1.26%  3.28%  3.14%  2.99%  1.20%
##    35:   2.30%  1.41%  3.24%  3.19%  2.94%  1.20%
##    36:   2.30%  1.41%  3.37%  3.10%  2.89%  1.20%
##    37:   2.34%  1.56%  3.37%  3.05%  2.84%  1.25%
##    38:   2.30%  1.44%  3.28%  3.10%  2.99%  1.16%
##    39:   2.25%  1.38%  3.24%  3.00%  2.89%  1.20%
##    40:   2.25%  1.44%  3.24%  2.95%  2.84%  1.20%
##    41:   2.18%  1.32%  3.11%  3.00%  2.74%  1.20%
##    42:   2.22%  1.47%  3.11%  3.10%  2.64%  1.20%
##    43:   2.22%  1.44%  3.02%  3.19%  2.74%  1.16%
##    44:   2.17%  1.35%  3.02%  2.95%  2.84%  1.16%
##    45:   2.11%  1.20%  3.06%  3.00%  2.64%  1.11%
##    46:   2.12%  1.26%  2.89%  3.19%  2.69%  1.06%
##    47:   2.02%  1.20%  2.89%  2.75%  2.59%  1.11%
##    48:   2.03%  1.26%  2.81%  2.70%  2.79%  1.01%
##    49:   2.09%  1.29%  3.02%  2.80%  2.69%  1.06%
##    50:   2.05%  1.20%  2.76%  2.90%  2.79%  1.06%
##    51:   2.04%  1.26%  2.72%  2.90%  2.69%  1.06%
##    52:   2.06%  1.26%  2.81%  2.85%  2.64%  1.16%
##    53:   1.96%  1.20%  2.63%  2.85%  2.44%  1.11%
##    54:   1.94%  1.08%  2.68%  2.80%  2.54%  1.06%
##    55:   1.93%  1.14%  2.68%  2.80%  2.39%  1.06%
##    56:   1.96%  1.11%  2.68%  2.80%  2.69%  1.01%
##    57:   1.94%  1.20%  2.59%  2.75%  2.49%  1.11%
##    58:   1.89%  1.08%  2.59%  2.70%  2.44%  1.11%
##    59:   1.96%  1.20%  2.63%  2.75%  2.59%  1.06%
##    60:   1.94%  1.08%  2.68%  2.75%  2.69%  1.01%
##    61:   1.92%  1.11%  2.68%  2.70%  2.54%  1.01%
##    62:   1.89%  1.17%  2.81%  2.46%  2.44%  0.92%
##    63:   1.87%  1.05%  2.76%  2.56%  2.49%  0.92%
##    64:   1.88%  1.08%  2.68%  2.51%  2.54%  1.01%
##    65:   1.86%  1.05%  2.72%  2.36%  2.69%  0.92%
##    66:   1.91%  1.14%  2.72%  2.41%  2.74%  0.96%
##    67:   1.91%  1.14%  2.81%  2.46%  2.49%  1.06%
##    68:   1.85%  1.08%  2.68%  2.41%  2.54%  0.96%
##    69:   1.84%  1.08%  2.59%  2.51%  2.44%  1.01%
##    70:   1.80%  0.96%  2.50%  2.46%  2.49%  1.06%
##    71:   1.81%  0.96%  2.55%  2.41%  2.59%  1.01%
##    72:   1.80%  0.99%  2.63%  2.31%  2.49%  1.01%
##    73:   1.80%  1.02%  2.50%  2.36%  2.49%  1.06%
##    74:   1.78%  0.93%  2.63%  2.31%  2.49%  1.01%
##    75:   1.78%  0.96%  2.55%  2.36%  2.54%  0.96%
##    76:   1.77%  0.99%  2.50%  2.46%  2.34%  1.01%
##    77:   1.79%  0.96%  2.68%  2.41%  2.44%  0.92%
##    78:   1.77%  1.02%  2.55%  2.36%  2.34%  0.96%
##    79:   1.73%  0.84%  2.63%  2.26%  2.44%  0.96%
##    80:   1.72%  0.96%  2.42%  2.31%  2.34%  0.96%
##    81:   1.75%  0.99%  2.59%  2.21%  2.29%  1.06%
##    82:   1.72%  1.05%  2.42%  2.21%  2.29%  0.96%
##    83:   1.71%  0.99%  2.46%  2.21%  2.24%  1.01%
##    84:   1.72%  0.99%  2.55%  2.26%  2.24%  0.96%
##    85:   1.72%  0.99%  2.50%  2.16%  2.34%  0.96%
##    86:   1.68%  0.96%  2.46%  2.16%  2.24%  0.96%
##    87:   1.69%  0.93%  2.50%  2.26%  2.24%  0.92%
##    88:   1.68%  0.93%  2.50%  2.21%  2.19%  0.96%
##    89:   1.70%  0.96%  2.46%  2.21%  2.39%  0.87%
##    90:   1.65%  0.84%  2.37%  2.16%  2.39%  0.92%
##    91:   1.68%  0.93%  2.46%  2.16%  2.39%  0.87%
##    92:   1.68%  0.90%  2.42%  2.26%  2.34%  0.92%
##    93:   1.69%  0.93%  2.46%  2.21%  2.39%  0.87%
##    94:   1.74%  0.96%  2.50%  2.31%  2.44%  0.92%
##    95:   1.69%  0.96%  2.42%  2.21%  2.39%  0.87%
##    96:   1.72%  0.96%  2.50%  2.26%  2.39%  0.92%
##    97:   1.73%  0.96%  2.50%  2.31%  2.39%  0.92%
##    98:   1.68%  0.93%  2.42%  2.16%  2.39%  0.92%
##    99:   1.72%  0.96%  2.46%  2.21%  2.49%  0.92%
##   100:   1.70%  0.93%  2.29%  2.31%  2.44%  0.96%
```

```
## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid
## mtry: reset to within valid range
```

```
## ntree      OOB      1      2      3      4      5
##     1:   7.97%  5.25% 11.16% 12.17%  7.64%  5.15%
##     2:   7.72%  4.75% 11.44% 11.84%  7.06%  5.12%
##     3:   8.01%  4.76% 11.70% 11.29%  8.27%  5.56%
##     4:   7.12%  4.78% 10.18%  9.95%  7.02%  4.79%
##     5:   6.61%  3.90%  9.26%  9.97%  6.58%  4.72%
##     6:   6.09%  3.77%  8.55%  8.84%  6.38%  4.09%
##     7:   5.91%  3.70%  8.75%  8.54%  5.69%  3.95%
##     8:   5.51%  3.52%  7.82%  8.41%  5.18%  3.56%
##     9:   4.92%  3.06%  6.86%  7.42%  4.97%  3.24%
##    10:   4.71%  3.03%  6.56%  7.01%  4.97%  2.82%
##    11:   4.20%  2.69%  5.77%  6.09%  4.96%  2.28%
##    12:   3.98%  2.62%  5.19%  5.98%  4.20%  2.62%
##    13:   3.64%  2.26%  4.93%  5.77%  3.80%  2.18%
##    14:   3.50%  2.13%  4.89%  5.37%  3.54%  2.27%
##    15:   3.33%  1.95%  4.28%  5.56%  3.84%  1.83%
##    16:   3.25%  1.89%  4.63%  5.21%  3.34%  1.88%
##    17:   2.96%  1.68%  4.41%  4.57%  3.14%  1.64%
##    18:   2.94%  1.65%  4.19%  4.86%  3.14%  1.54%
##    19:   2.89%  1.62%  4.32%  4.62%  3.14%  1.40%
##    20:   2.71%  1.50%  4.23%  4.13%  3.04%  1.25%
##    21:   2.67%  1.56%  4.06%  4.03%  2.79%  1.45%
##    22:   2.56%  1.59%  4.14%  3.73%  2.54%  1.21%
##    23:   2.56%  1.68%  3.84%  3.98%  2.49%  1.25%
##    24:   2.45%  1.65%  3.63%  3.69%  2.44%  1.25%
##    25:   2.42%  1.53%  3.54%  3.78%  2.39%  1.30%
##    26:   2.43%  1.47%  3.67%  3.73%  2.39%  1.35%
##    27:   2.36%  1.41%  3.32%  3.69%  2.54%  1.35%
##    28:   2.37%  1.47%  3.41%  3.19%  2.74%  1.49%
##    29:   2.35%  1.35%  3.28%  3.73%  2.59%  1.35%
##    30:   2.34%  1.38%  3.37%  3.44%  2.54%  1.49%
##    31:   2.28%  1.35%  3.50%  3.14%  2.54%  1.35%
##    32:   2.28%  1.32%  3.41%  3.44%  2.44%  1.30%
##    33:   2.39%  1.41%  3.45%  3.34%  2.84%  1.40%
##    34:   2.35%  1.35%  3.58%  3.24%  2.64%  1.45%
##    35:   2.28%  1.38%  3.63%  2.80%  2.64%  1.40%
##    36:   2.17%  1.32%  3.37%  2.75%  2.54%  1.25%
##    37:   2.28%  1.38%  3.50%  3.05%  2.49%  1.40%
##    38:   2.24%  1.35%  3.37%  3.00%  2.64%  1.30%
##    39:   2.23%  1.26%  3.45%  2.95%  2.54%  1.45%
##    40:   2.14%  1.29%  3.37%  2.85%  2.44%  1.16%
##    41:   2.18%  1.35%  3.24%  2.80%  2.69%  1.25%
##    42:   2.24%  1.32%  3.41%  3.14%  2.59%  1.20%
##    43:   2.20%  1.29%  3.37%  2.95%  2.54%  1.30%
##    44:   2.07%  1.23%  3.15%  2.80%  2.39%  1.20%
##    45:   2.11%  1.14%  3.24%  2.90%  2.49%  1.25%
##    46:   2.05%  1.11%  3.06%  2.75%  2.64%  1.16%
##    47:   2.02%  1.11%  3.02%  2.60%  2.59%  1.25%
##    48:   2.01%  1.17%  3.06%  2.70%  2.39%  1.16%
##    49:   1.97%  1.14%  3.02%  2.60%  2.39%  1.11%
##    50:   1.93%  1.05%  2.98%  2.51%  2.29%  1.25%
##    51:   1.98%  1.14%  2.98%  2.65%  2.39%  1.16%
##    52:   1.95%  1.17%  3.02%  2.60%  2.29%  1.06%
##    53:   1.93%  1.08%  2.98%  2.60%  2.29%  1.11%
##    54:   1.95%  1.08%  2.93%  2.60%  2.44%  1.16%
##    55:   1.94%  1.11%  2.85%  2.60%  2.39%  1.20%
##    56:   1.94%  1.08%  2.81%  2.65%  2.44%  1.16%
##    57:   1.92%  0.99%  2.85%  2.65%  2.54%  1.06%
##    58:   1.90%  0.99%  2.85%  2.51%  2.54%  1.11%
##    59:   1.94%  1.02%  3.02%  2.56%  2.49%  1.11%
##    60:   1.97%  1.05%  3.11%  2.70%  2.39%  1.06%
##    61:   1.96%  1.05%  2.85%  2.70%  2.49%  1.20%
##    62:   1.94%  1.05%  2.76%  2.65%  2.54%  1.16%
##    63:   1.87%  0.96%  2.81%  2.51%  2.39%  1.16%
##    64:   1.90%  0.96%  2.81%  2.56%  2.49%  1.20%
##    65:   1.85%  0.99%  2.68%  2.56%  2.44%  1.06%
##    66:   1.84%  0.96%  2.81%  2.46%  2.39%  1.06%
##    67:   1.89%  0.99%  2.89%  2.56%  2.34%  1.11%
##    68:   1.89%  0.99%  2.89%  2.60%  2.44%  1.01%
##    69:   1.86%  0.93%  2.81%  2.70%  2.39%  0.96%
##    70:   1.89%  1.02%  2.89%  2.65%  2.29%  1.01%
##    71:   1.85%  0.96%  2.76%  2.56%  2.39%  1.06%
##    72:   1.89%  1.02%  2.81%  2.70%  2.39%  1.01%
##    73:   1.88%  0.96%  2.81%  2.65%  2.39%  1.06%
##    74:   1.85%  0.96%  2.76%  2.65%  2.34%  1.01%
##    75:   1.81%  0.93%  2.72%  2.56%  2.24%  1.06%
##    76:   1.82%  0.93%  2.76%  2.51%  2.29%  1.06%
##    77:   1.81%  0.90%  2.81%  2.56%  2.24%  1.01%
##    78:   1.78%  0.90%  2.72%  2.51%  2.19%  1.06%
##    79:   1.80%  0.90%  2.76%  2.51%  2.29%  1.01%
##    80:   1.77%  0.90%  2.63%  2.46%  2.39%  0.92%
##    81:   1.76%  0.90%  2.76%  2.31%  2.34%  0.92%
##    82:   1.78%  0.93%  2.72%  2.36%  2.39%  0.96%
##    83:   1.77%  0.93%  2.63%  2.51%  2.29%  0.96%
##    84:   1.78%  0.93%  2.72%  2.51%  2.19%  1.01%
##    85:   1.77%  0.93%  2.72%  2.51%  2.14%  0.96%
##    86:   1.75%  0.87%  2.68%  2.51%  2.14%  1.01%
##    87:   1.74%  0.90%  2.59%  2.51%  2.14%  1.01%
##    88:   1.75%  0.90%  2.59%  2.41%  2.24%  1.06%
##    89:   1.77%  0.93%  2.68%  2.41%  2.19%  1.06%
##    90:   1.79%  0.93%  2.72%  2.46%  2.24%  1.06%
##    91:   1.76%  0.90%  2.72%  2.41%  2.19%  1.01%
##    92:   1.72%  0.90%  2.72%  2.41%  1.99%  1.01%
##    93:   1.77%  0.87%  2.68%  2.60%  2.14%  1.01%
##    94:   1.77%  0.90%  2.76%  2.65%  2.04%  0.96%
##    95:   1.76%  0.93%  2.76%  2.56%  1.99%  0.96%
##    96:   1.72%  0.93%  2.59%  2.60%  1.99%  0.92%
##    97:   1.73%  0.93%  2.68%  2.46%  2.04%  0.96%
##    98:   1.73%  0.87%  2.63%  2.56%  2.14%  0.92%
##    99:   1.72%  0.90%  2.63%  2.51%  2.04%  0.92%
##   100:   1.72%  0.90%  2.68%  2.51%  2.04%  0.92%
## ntree      OOB      1      2      3      4      5
##     1:  16.82% 13.13% 22.46% 20.31% 17.99% 12.42%
##     2:  16.46% 11.80% 22.76% 19.60% 15.71% 14.70%
##     3:  15.73% 11.08% 21.97% 19.78% 14.99% 13.21%
##     4:  15.85% 10.94% 21.63% 20.25% 15.39% 13.54%
##     5:  14.48%  9.39% 20.29% 18.06% 14.88% 12.50%
##     6:  13.64%  9.13% 18.66% 17.42% 14.71% 10.81%
##     7:  12.62%  7.95% 17.38% 16.32% 13.99% 10.09%
##     8:  11.32%  7.25% 15.16% 14.63% 12.15%  9.63%
##     9:  10.41%  6.61% 14.39% 13.74% 10.81%  8.58%
##    10:   9.60%  6.55% 12.47% 12.71% 10.58%  7.46%
##    11:   9.10%  6.03% 12.21% 11.37% 10.73%  6.96%
##    12:   8.35%  5.27% 11.97% 11.48%  9.15%  5.61%
##    13:   7.92%  5.33% 11.07% 10.21%  8.47%  5.97%
##    14:   7.09%  4.58%  9.89%  9.41%  8.20%  4.81%
##    15:   6.55%  4.09%  8.83%  8.86%  7.88%  4.58%
##    16:   6.32%  3.83%  9.18%  9.06%  7.56%  3.47%
##    17:   5.85%  3.65%  8.69%  8.03%  6.68%  3.47%
##    18:   5.76%  3.73%  8.78%  7.06%  6.58%  3.74%
##    19:   5.32%  3.38%  7.90%  7.50%  5.80%  3.14%
##    20:   5.16%  3.17%  7.85%  6.86%  6.27%  2.82%
##    21:   4.82%  2.84%  7.46%  6.33%  5.85%  2.77%
##    22:   4.89%  2.81%  7.77%  6.52%  5.91%  2.63%
##    23:   4.55%  2.75%  6.89%  6.04%  5.39%  2.73%
##    24:   4.31%  2.54%  6.76%  5.79%  5.23%  2.22%
##    25:   4.40%  2.51%  6.89%  5.84%  5.60%  2.26%
##    26:   4.17%  2.33%  6.63%  5.79%  5.08%  2.08%
##    27:   4.07%  2.39%  6.27%  5.45%  5.23%  1.99%
##    28:   3.79%  2.00%  5.79%  5.16%  5.34%  1.76%
##    29:   3.89%  2.03%  6.10%  5.31%  5.34%  1.80%
##    30:   3.71%  2.15%  5.75%  4.67%  5.39%  1.57%
##    31:   3.61%  1.97%  5.48%  4.92%  5.13%  1.57%
##    32:   3.59%  1.85%  5.66%  4.82%  4.92%  1.76%
##    33:   3.51%  2.00%  5.44%  4.38%  4.82%  1.80%
##    34:   3.37%  1.94%  4.91%  4.43%  4.82%  1.66%
##    35:   3.39%  2.03%  5.13%  4.19%  4.72%  1.71%
##    36:   3.35%  1.94%  5.05%  4.09%  4.87%  1.66%
##    37:   3.33%  1.91%  4.91%  4.33%  4.72%  1.66%
##    38:   3.24%  1.91%  4.39%  4.43%  4.77%  1.57%
##    39:   3.23%  1.88%  4.56%  4.28%  4.72%  1.57%
##    40:   3.23%  1.88%  4.74%  4.38%  4.56%  1.43%
##    41:   3.17%  1.85%  4.87%  3.89%  4.56%  1.48%
##    42:   3.12%  1.85%  4.43%  4.04%  4.56%  1.52%
##    43:   3.18%  1.79%  4.91%  3.99%  4.51%  1.52%
##    44:   3.16%  1.91%  4.74%  4.04%  4.46%  1.43%
##    45:   3.15%  1.85%  4.70%  4.09%  4.51%  1.43%
##    46:   3.13%  1.76%  4.65%  4.09%  4.77%  1.29%
##    47:   3.01%  1.64%  4.39%  4.04%  4.56%  1.34%
##    48:   2.99%  1.52%  4.48%  3.85%  4.82%  1.25%
##    49:   2.90%  1.64%  4.12%  3.85%  4.51%  1.25%
##    50:   2.90%  1.61%  4.12%  3.99%  4.51%  1.11%
##    51:   2.75%  1.43%  3.91%  3.75%  4.40%  1.15%
##    52:   2.87%  1.49%  3.99%  3.94%  4.72%  1.15%
##    53:   2.89%  1.52%  4.04%  4.04%  4.51%  1.25%
##    54:   2.90%  1.49%  4.08%  3.94%  4.66%  1.25%
##    55:   2.84%  1.43%  4.04%  3.85%  4.72%  1.15%
##    56:   2.77%  1.40%  3.95%  3.80%  4.40%  1.20%
##    57:   2.83%  1.43%  3.99%  3.85%  4.51%  1.29%
##    58:   2.89%  1.46%  4.26%  3.75%  4.61%  1.29%
##    59:   2.82%  1.40%  3.86%  4.09%  4.40%  1.29%
##    60:   2.86%  1.55%  3.91%  3.85%  4.72%  1.20%
##    61:   2.84%  1.46%  4.12%  3.75%  4.66%  1.11%
##    62:   2.79%  1.43%  4.17%  3.65%  4.40%  1.20%
##    63:   2.73%  1.37%  3.95%  3.65%  4.51%  1.06%
##    64:   2.73%  1.34%  3.95%  3.70%  4.40%  1.20%
##    65:   2.73%  1.31%  3.91%  3.75%  4.35%  1.29%
##    66:   2.83%  1.40%  3.91%  3.70%  4.77%  1.34%
##    67:   2.79%  1.40%  4.04%  3.55%  4.66%  1.25%
##    68:   2.78%  1.28%  4.04%  3.70%  4.61%  1.25%
##    69:   2.77%  1.34%  3.95%  3.55%  4.56%  1.39%
##    70:   2.79%  1.34%  3.91%  3.65%  4.82%  1.25%
##    71:   2.75%  1.28%  3.86%  3.60%  4.82%  1.20%
##    72:   2.79%  1.34%  4.12%  3.60%  4.72%  1.15%
##    73:   2.73%  1.19%  4.04%  3.55%  4.66%  1.25%
##    74:   2.73%  1.34%  3.82%  3.36%  4.82%  1.29%
##    75:   2.79%  1.34%  4.21%  3.36%  4.72%  1.25%
##    76:   2.71%  1.28%  4.04%  3.36%  4.56%  1.25%
##    77:   2.77%  1.37%  4.26%  3.36%  4.46%  1.29%
##    78:   2.76%  1.19%  4.12%  3.51%  4.72%  1.29%
##    79:   2.75%  1.25%  4.21%  3.46%  4.56%  1.25%
##    80:   2.73%  1.14%  4.17%  3.46%  4.66%  1.29%
##    81:   2.73%  1.19%  4.04%  3.51%  4.66%  1.29%
##    82:   2.75%  1.16%  4.12%  3.65%  4.61%  1.25%
##    83:   2.76%  1.11%  4.17%  3.55%  4.66%  1.39%
##    84:   2.69%  1.14%  4.08%  3.60%  4.46%  1.20%
##    85:   2.67%  1.08%  3.99%  3.55%  4.56%  1.25%
##    86:   2.67%  1.16%  4.08%  3.41%  4.51%  1.15%
##    87:   2.65%  1.14%  4.04%  3.36%  4.51%  1.20%
##    88:   2.65%  1.14%  3.95%  3.41%  4.56%  1.20%
##    89:   2.65%  1.22%  4.04%  3.46%  4.35%  1.11%
##    90:   2.68%  1.16%  4.21%  3.36%  4.46%  1.20%
##    91:   2.67%  1.19%  4.26%  3.26%  4.51%  1.11%
##    92:   2.67%  1.16%  4.30%  3.16%  4.51%  1.15%
##    93:   2.62%  1.11%  4.17%  3.16%  4.51%  1.11%
##    94:   2.51%  0.99%  4.17%  3.07%  4.25%  1.06%
##    95:   2.56%  1.05%  4.26%  3.12%  4.25%  1.11%
##    96:   2.58%  1.05%  4.26%  3.07%  4.35%  1.15%
##    97:   2.60%  1.02%  4.12%  3.07%  4.56%  1.25%
##    98:   2.53%  1.02%  4.08%  3.07%  4.40%  1.06%
##    99:   2.61%  1.08%  4.12%  3.16%  4.51%  1.15%
##   100:   2.56%  1.08%  3.86%  3.16%  4.46%  1.25%
```

```r
ModelRF
```

```
## Random Forest 
## 
## 11776 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction (79), scaled
##  (79), centered (79) 
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9682364  0.9598276
##   40    0.9557849  0.9440779
##   79    0.9560543  0.9444219
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

Cross-validation & Confusion Matrix:

```r
PredictRF <- predict(ModelRF, newdata=SubTest, type="raw")
confusionMatrix(PredictRF, SubTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2203   25    0    0    0
##          B   10 1477   25    0    0
##          C   18   16 1341   56    1
##          D    0    0    2 1229   10
##          E    1    0    0    1 1431
## 
## Overall Statistics
##                                          
##                Accuracy : 0.979          
##                  95% CI : (0.9755, 0.982)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9734         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9870   0.9730   0.9803   0.9557   0.9924
## Specificity            0.9955   0.9945   0.9860   0.9982   0.9997
## Pos Pred Value         0.9888   0.9769   0.9365   0.9903   0.9986
## Neg Pred Value         0.9948   0.9935   0.9958   0.9914   0.9983
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2808   0.1882   0.1709   0.1566   0.1824
## Detection Prevalence   0.2840   0.1927   0.1825   0.1582   0.1826
## Balanced Accuracy      0.9913   0.9837   0.9831   0.9769   0.9960
```



### Overall Results

The Random Forest algorithm has the highest accuracy 0.9797 and the out of sample error is 0.0203. I will use Random Forest algorithm for final predictions.



## Prediction of Test Data


```r
Prediction <- predict(ModelRF, newdata=TestData)
Prediction
```

```
##  [1] B A C A A E D B A A A C B A E E A B B B
## Levels: A B C D E
```
