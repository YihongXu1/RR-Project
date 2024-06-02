---
title: "Machine Learning Method Application to Covid-19 Dataset"
author: ""
date: today
format:
  html:
    toc: true
    toc-title: Contents
    toc-depth: 2
    toc-expand: 1
    smooth-scroll: true
    theme:
      light: lumen
      dark: superhero
number-sections: true
number-depth: 2
editor: visual
execute:
  warning: false
  cache: true 
keep-md: true
title-block-banner: true
---



# Preparation

Load all related packages, and define hyper-parameters from the very beginning. To ensure reproductivity, all seed are set to the fixed value `seed`.

```{.r .cell-code}
library(tidyverse) # data cleaning
library(knitr) # display tables in good format
library(caret) # confusion matrics
library(bnclassify) # BN
library(kernlab) # SVM
library(randomForest) # random forest
library(C50) # C5.0 decision tree
```
```{.r .cell-code}
# set hyperparameters
path_csv <- "../Data/patient.csv"
seed <- 9
```

# Data

## Loading

First of all, load the raw data. There are 95839 samples and 20 features (including the response variable) here.

```{.r .cell-code}
df <- read.csv(path_csv)
dim(df)
```

```
[1] 95839    20
```

```{.r .cell-code}
kable(head(df), format = "html")
```


## Cleaning

Aiming to solve the binary classification problem, binary `y` should be elicited first. As for features, not all of them are available. For example, `pneumonia = 99` indicate the feature `pneumonia` is not available for this sample. In this case, we should label `99` as `NA` to avoid future mistakes. Some machine learning methods cannot handle cases with NA value, we actually use sample without NA values. however, NA for feature `pregnant` is not really not available. All males are labelled `NA`, but it does not make sense to eliminate all males. In this case, `NA` for feature `pregnant` should be changed to `2` which indicate not pregnant. Besides, `age` is the only continuous variable in this dataset. NaiveBayes and BayesNet could only handle factor features, while others work with numeric features. We prepare two version of data frames, one with factor age column and the other with numeric age column.



```{.r .cell-code}
df_ageori <- df %>% mutate(y = factor(ifelse(death_date == "9999-99-99", 0, 1),
                                      labels = c("live", "die")),
                           pregnant = factor(ifelse(pregnant == 1, 1, 2)),
              across(c(sex, patient_type, intubated, pneumonia,
                       diabetes, copd, asthma, immunosuppression, hypertension,
                       other_diseases, cardiovascular, obesity, chronic_kidney_failure,
                       smoker, another_case, icu, outcome), ~ factor(ifelse(.>2, NA, .)))) %>%
  select(-c(death_date))
df_agefac <- df_ageori %>% mutate(age = factor(age))
kable(head(df_agefac), format = "html")
```


# Model

In this part, we managed to reproduce the result in the papar [Classification of Covid-19 Dataset with Some Machine Learning Methods](https://dergipark.org.tr/en/pub/jauist/issue/55760/748667). machine learning classifier methods are considered:

-   NaiveBayes

-   BayesNet

-   SVM

-   Random Forest

-   Decision Tree

-   KNN

For each model, we calculate the confusion matrix to get accuracy, precision, recall and F1 score as summarized in the original paper. 10-fold cross validation is applied for each model. Related model file are save to the file `model/*`.

## NaiveBayes

NaiveBayes assumes all variables are independent from each other. By experience, it works relatively well even if the assumptions are not met.


::: {.cell}

```{.r .cell-code}
df <- na.omit(df_agefac) # use factorized age version 
set.seed(seed)
model_nb <- nb('y', df)
model_nb <- lp(model_nb, df, smooth = 1) # learn parameter
cv(model_nb, df, k = 10) # cross validation
pred_nb <- predict(model_nb, df)
confusionMatrix(pred_nb, df$y)
```
```
Confusion Matrix and Statistics

          Reference
Prediction live  die
      live 8363  861
      die   499  468
                                          
               Accuracy : 0.8665          
                 95% CI : (0.8598, 0.8731)
    No Information Rate : 0.8696          
    P-Value [Acc > NIR] : 0.8231          
                                          
                  Kappa : 0.3346          
                                          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9437          
            Specificity : 0.3521          
         Pos Pred Value : 0.9067          
         Neg Pred Value : 0.4840          
             Prevalence : 0.8696          
         Detection Rate : 0.8206          
   Detection Prevalence : 0.9051          
      Balanced Accuracy : 0.6479          
                                          
       'Positive' Class : live            
                                          
```





## BayesNet

Different from Naive Bayes, Bayes net define a complicated network structure which indicate relationships among a set of features. The assumption makes sense but is more time-consuming than NaiveBayes Model.

```{.r .cell-code}
set.seed(seed)
model_bn <- tan_cl('y', df, score = 'aic')
model_bn <- lp(model_bn, df, smooth = 1)
cv(model_bn, df, k = 10)
pred_bn <- predict(model_bn, df)
confusionMatrix(pred_bn, df$y)
```
```
Confusion Matrix and Statistics

          Reference
Prediction live  die
      live 8604  961
      die   258  368
                                          
               Accuracy : 0.8804          
                 95% CI : (0.8739, 0.8866)
    No Information Rate : 0.8696          
    P-Value [Acc > NIR] : 0.0005642       
                                          
                  Kappa : 0.3197          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.9709          
            Specificity : 0.2769          
         Pos Pred Value : 0.8995          
         Neg Pred Value : 0.5879          
             Prevalence : 0.8696          
         Detection Rate : 0.8443          
   Detection Prevalence : 0.9386          
      Balanced Accuracy : 0.6239          
                                          
       'Positive' Class : live            
                                          
```

## SVM

Support Vector Machine (SVM) aim to find the hyperplane with the largest margin to classify data points. Kernel trick is applied here to improve accuracy. We use  radial basis kernel. 

```{.r .cell-code}
df <- na.omit(df_ageori)
set.seed(seed)
train_control <- trainControl(method = "cv", number = 10)
model_svm <- train(y ~ ., data = df, method = "svmRadial", trControl = train_control)
pred_svm <- predict(model_svm, df)
confusionMatrix(pred_svm, df$y)
```
```
Confusion Matrix and Statistics

          Reference
Prediction live  die
      live 8862 1329
      die     0    0
                                          
               Accuracy : 0.8696          
                 95% CI : (0.8629, 0.8761)
    No Information Rate : 0.8696          
    P-Value [Acc > NIR] : 0.5073          
                                          
                  Kappa : 0               
                                          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 1.0000          
            Specificity : 0.0000          
         Pos Pred Value : 0.8696          
         Neg Pred Value :    NaN          
             Prevalence : 0.8696          
         Detection Rate : 0.8696          
   Detection Prevalence : 1.0000          
      Balanced Accuracy : 0.5000          
                                          
       'Positive' Class : live            
                                          
```


## Random Forest

Random Forest (RF) is consisted of a set of decision trees. Each tree is a weak classifier trained with only a subset of data and features. RF is actually a ensemble learning method.


::: {.cell}

```{.r .cell-code}
set.seed(seed)
df <- na.omit(df_ageori)
train_control <- trainControl(method = "cv", number = 10)
model_rf <- train(y ~ ., data = df, method = "rf", trControl = train_control)
pred_rf <- predict(model_rf, df)
confusionMatrix(pred_rf, df$y)
```
```
Confusion Matrix and Statistics

          Reference
Prediction live  die
      live 8862 1300
      die     0   29
                                          
               Accuracy : 0.8724          
                 95% CI : (0.8658, 0.8789)
    No Information Rate : 0.8696          
    P-Value [Acc > NIR] : 0.2012          
                                          
                  Kappa : 0.0373          
                                          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 1.00000         
            Specificity : 0.02182         
         Pos Pred Value : 0.87207         
         Neg Pred Value : 1.00000         
             Prevalence : 0.86959         
         Detection Rate : 0.86959         
   Detection Prevalence : 0.99715         
      Balanced Accuracy : 0.51091         
                                          
       'Positive' Class : live            
                                          
```


## Decision Tree (C4.5)

Different decision tree algorithms have different feature selection methods, ID3 uses information gain, CART uses gini coefficient and C4.5 uses information gain rate. C5.0, which we use here, is a modified version of C4.5 to be more efficient and accurate.


::: {.cell}

```{.r .cell-code}
set.seed(seed)
train_control <- trainControl(method = "cv", number = 10)
model_tree <- train(y ~ ., data = df, method = "C5.0", trControl = train_control)
pred_tree <- predict(model_tree, df)
confusionMatrix(pred_tree, df$y)
```
```
Confusion Matrix and Statistics

          Reference
Prediction live  die
      live 8862 1329
      die     0    0
                                          
               Accuracy : 0.8696          
                 95% CI : (0.8629, 0.8761)
    No Information Rate : 0.8696          
    P-Value [Acc > NIR] : 0.5073          
                                          
                  Kappa : 0               
                                          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 1.0000          
            Specificity : 0.0000          
         Pos Pred Value : 0.8696          
         Neg Pred Value :    NaN          
             Prevalence : 0.8696          
         Detection Rate : 0.8696          
   Detection Prevalence : 1.0000          
      Balanced Accuracy : 0.5000          
                                          
       'Positive' Class : live            
                                          
```


## kNN

kNN method find k neighbors near the datapoints first. Class of new points is determined by classes of its neighbors. kNN is non-linear.


::: {.cell}

```{.r .cell-code}
set.seed(seed)
train_control <- trainControl(method = "cv", number = 10)
model_knn <- train(y ~ ., data = df, method = "knn", trControl = train_control)
pred_knn <- predict(model_knn, df)
confusionMatrix(pred_knn, df$y)
```
```
Confusion Matrix and Statistics

          Reference
Prediction live  die
      live 8798 1200
      die    64  129
                                          
               Accuracy : 0.876           
                 95% CI : (0.8694, 0.8823)
    No Information Rate : 0.8696          
    P-Value [Acc > NIR] : 0.02826         
                                          
                  Kappa : 0.1411          
                                          
 Mcnemar's Test P-Value : < 2e-16         
                                          
            Sensitivity : 0.99278         
            Specificity : 0.09707         
         Pos Pred Value : 0.87998         
         Neg Pred Value : 0.66839         
             Prevalence : 0.86959         
         Detection Rate : 0.86331         
   Detection Prevalence : 0.98106         
      Balanced Accuracy : 0.54492         
                                          
       'Positive' Class : live            
                                          
```


# Summary

In this part, we summarize all the results above to reproduce the main result table in the original paper. 

```{.r .cell-code}
# get all scores of a model in a function
get_all_scores <- function(pred, y = df$y){
  acc <- accuracy(pred, y)
  prec <- precision(pred, y)
  rec <- recall(pred, y)
  F1 <- 2*(prec*rec)/(prec+rec)
  return(c(acc, prec, F1, rec))
}
```
```{.r .cell-code}
pred_list <- list(pred_nb, pred_bn, pred_svm, 
                  pred_rf, pred_tree, pred_knn)
summary_table <- do.call(rbind, lapply(pred_list, get_all_scores))
summary_table <- round(summary_table, 3)
colnames(summary_table) <- c("Accuracy", "Precision", "F1", "Recall")
summary_table <- cbind(data.frame(Model = c("NaiveBayes", "BayesNet", "SVM", "RandomForest", "DecisionTree", "kNN")), summary_table)
kable(summary_table, format = "html")
```

$$
\begin{array}{|c|c|c|c|c|}
\hline \text { Model } & \text { Accuracy } & \text { Precision } & \text { F1 } & \text { Recall } \\
\hline \text { NaiveBayes } & 0.867 & 0.907 & 0.925 & 0.944 \\
\hline \text { BayesNet } & 0.880 & 0.900 & 0.934 & 0.971 \\
\hline \text { SVM } & 0.870 & 0.870 & 0.930 & 1.000 \\
\hline \text { RandomForest } & 0.872 & 0.872 & 0.932 & 1.000 \\
\hline \text { DecisionTree } & 0.870 & 0.870 & 0.930 & 1.000 \\
\hline \mathrm{kNN} & 0.876 & 0.880 & 0.933 & 0.993 \\
\hline
\end{array}
$$

Results are not totally the same as shown in the paper. The paper states that SVM renders the best accuracy score, all scores are rather close to 1, while ours is not the same. kNN is the best accordin to accuracy score. recall of SVM, random forest and decision tree is 1 (the same as the paper), while accuracy and precision could not reach 1. Possible reasons of the difference are listed below:

- There are different data preprocessing methods. As some models cannot handle NA values, what to do with NA recording matters. Details of preprocessing is not available in the paper, so we may get different data to fit the model.

- Seeds are also different. In our study, we fix the seed to ensure reproducibility. But the seed is not the same as the paper. In this cased, for each fold of the cross validation method, we got different subsets of data from the ones used in the paper.

- Different program languages also matter. The paper used WEKA, while we use R. Underlying implementation varies.

- Subtypes of the methods are not totally the same. For example, SMO version of SVM is used in the paper. However, we could not do exactly the same, instead we apply radial basis kernel. There are some ambiguity is the paper which prevents us from reproducing as well. As we known, NaiveBayes and BayesNet adopt different assumptions, but the original paper refer to both method as NaiveBayes.

In conclusion, we cannot reproduce every details of the original paper without specification. However, results of our own can be reproduced since random seed is fixed and every detail is displayed in the chunks above. To reproduce all results at once, run `reproduce.r`.









































