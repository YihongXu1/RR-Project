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


::: {.cell}

```{.r .cell-code}
library(tidyverse) # data cleaning
library(knitr) # display tables in good format
library(caret) # confusion matrics
library(bnclassify) # BN
library(kernlab) # SVM
library(randomForest) # random forest
library(C50) # C5.0 decision tree
```
:::

::: {.cell}

```{.r .cell-code}
# set hyperparameters
path_csv <- "../Data/patient.csv"
seed <- 9
```
:::


# Data

## Loading

First of all, load the raw data. There are 95839 samples and 20 features (including the response variable) here.


::: {.cell}

```{.r .cell-code}
df <- read.csv(path_csv)
dim(df)
```

::: {.cell-output .cell-output-stdout}

```
[1] 95839    20
```


:::

```{.r .cell-code}
kable(head(df), format = "html")
```

::: {.cell-output-display}

`````{=html}
<table>
 <thead>
  <tr>
   <th style="text-align:right;"> sex </th>
   <th style="text-align:right;"> patient_type </th>
   <th style="text-align:right;"> intubated </th>
   <th style="text-align:right;"> pneumonia </th>
   <th style="text-align:right;"> age </th>
   <th style="text-align:right;"> pregnant </th>
   <th style="text-align:right;"> diabetes </th>
   <th style="text-align:right;"> copd </th>
   <th style="text-align:right;"> asthma </th>
   <th style="text-align:right;"> immunosuppression </th>
   <th style="text-align:right;"> hypertension </th>
   <th style="text-align:right;"> other_diseases </th>
   <th style="text-align:right;"> cardiovascular </th>
   <th style="text-align:right;"> obesity </th>
   <th style="text-align:right;"> chronic_kidney_failure </th>
   <th style="text-align:right;"> smoker </th>
   <th style="text-align:right;"> another_case </th>
   <th style="text-align:right;"> outcome </th>
   <th style="text-align:right;"> icu </th>
   <th style="text-align:left;"> death_date </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 42 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 99 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:left;"> 9999-99-99 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 51 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 99 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:left;"> 9999-99-99 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 51 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 99 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:left;"> 9999-99-99 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 57 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 99 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:left;"> 2020-04-01 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 44 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:left;"> 9999-99-99 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 40 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 98 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 99 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:left;"> 9999-99-99 </td>
  </tr>
</tbody>
</table>

`````

:::
:::


## Cleaning

Aiming to solve the binary classification problem, binary `y` should be elicited first. As for features, not all of them are available. For example, `pneumonia = 99` indicate the feature `pneumonia` is not available for this sample. In this case, we should label `99` as `NA` to avoid future mistakes.


::: {.cell}

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

::: {.cell-output-display}

`````{=html}
<table>
 <thead>
  <tr>
   <th style="text-align:left;"> sex </th>
   <th style="text-align:left;"> patient_type </th>
   <th style="text-align:left;"> intubated </th>
   <th style="text-align:left;"> pneumonia </th>
   <th style="text-align:left;"> age </th>
   <th style="text-align:left;"> pregnant </th>
   <th style="text-align:left;"> diabetes </th>
   <th style="text-align:left;"> copd </th>
   <th style="text-align:left;"> asthma </th>
   <th style="text-align:left;"> immunosuppression </th>
   <th style="text-align:left;"> hypertension </th>
   <th style="text-align:left;"> other_diseases </th>
   <th style="text-align:left;"> cardiovascular </th>
   <th style="text-align:left;"> obesity </th>
   <th style="text-align:left;"> chronic_kidney_failure </th>
   <th style="text-align:left;"> smoker </th>
   <th style="text-align:left;"> another_case </th>
   <th style="text-align:left;"> outcome </th>
   <th style="text-align:left;"> icu </th>
   <th style="text-align:left;"> y </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 42 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> live </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 51 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> live </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 51 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> live </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 57 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> die </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 44 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> live </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 40 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> live </td>
  </tr>
</tbody>
</table>

`````

:::
:::


# Model

In this part, we managed to reproduce the result in the papar [Classification of Covid-19 Dataset with Some Machine Learning Methods](https://dergipark.org.tr/en/pub/jauist/issue/55760/748667). machine learning classifier methods are considered:

-   NaiveBayes

-   BayesNet

-   SVM

-   Random Forest

-   Decision Tree

-   KNN

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
:::

::: {.cell}

:::

::: {.cell}
::: {.cell-output .cell-output-stdout}

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


:::
:::


## BayesNet

Different from Naive Bayes, Bayes net


::: {.cell}

```{.r .cell-code}
set.seed(seed)
model_bn <- tan_cl('y', df, score = 'aic')
model_bn <- lp(model_bn, df, smooth = 1)
cv(model_bn, df, k = 10)
pred_bn <- predict(model_bn, df)
confusionMatrix(pred_bn, df$y)
```
:::

::: {.cell}

:::

::: {.cell}
::: {.cell-output .cell-output-stdout}

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


:::
:::

## SVM


::: {.cell}

```{.r .cell-code}
df <- na.omit(df_ageori)
set.seed(seed)
train_control <- trainControl(method = "cv", number = 10)
model_svm <- train(y ~ ., data = df, method = "svmRadial", trControl = train_control)
pred_svm <- predict(model_svm, df)
confusionMatrix(pred_svm, df$y)
```
:::

::: {.cell}

:::

::: {.cell}
::: {.cell-output .cell-output-stdout}

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


:::
:::


## Random Forest


::: {.cell}

```{.r .cell-code}
set.seed(seed)
df <- na.omit(df_ageori)
train_control <- trainControl(method = "cv", number = 10)
model_rf <- train(y ~ ., data = df, method = "rf", trControl = train_control)
pred_rf <- predict(model_rf, df)
confusionMatrix(pred_rf, df$y)
```
:::

::: {.cell}

:::

::: {.cell}
::: {.cell-output .cell-output-stdout}

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


:::
:::


## Decision Tree (C4.5)


::: {.cell}

```{.r .cell-code}
set.seed(seed)
train_control <- trainControl(method = "cv", number = 10)
model_tree <- train(y ~ ., data = df, method = "C5.0", trControl = train_control)
pred_tree <- predict(model_tree, df)
confusionMatrix(pred_tree, df$y)
```
:::

::: {.cell}

:::

::: {.cell}
::: {.cell-output .cell-output-stdout}

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


:::
:::


## kNN




::: {.cell}

```{.r .cell-code}
set.seed(seed)
train_control <- trainControl(method = "cv", number = 10)
model_knn <- train(y ~ ., data = df, method = "knn", trControl = train_control)
pred_knn <- predict(model_knn, df)
confusionMatrix(pred_knn, df$y)
```
:::

::: {.cell}

:::

::: {.cell}
::: {.cell-output .cell-output-stdout}

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


:::
:::


# Summary

In this part, we summarize all the results above to reproduce the main result table in the original paper

accuracy
precision
F-measure
recall


::: {.cell}

```{.r .cell-code}
get_all_scores <- function(pred, y = df$y){
  acc <- accuracy(pred, y)
  prec <- precision(pred, y)
  rec <- recall(pred, y)
  F1 <- 2*(prec*rec)/(prec+rec)
  return(c(acc, prec, F1, rec))
}
```
:::

::: {.cell}

:::

::: {.cell}

```{.r .cell-code}
pred_list <- list(pred_nb, pred_bn, pred_svm, 
                  pred_rf, pred_tree, pred_knn)
summary_table <- do.call(rbind, lapply(pred_list, get_all_scores))
summary_table <- round(summary_table, 3)
colnames(summary_table) <- c("Accuracy", "Precision", "F1", "Recall")
summary_table <- cbind(data.frame(Model = c("NaiveBayes", "BayesNet", "SVM", "RandomForest", "DecisionTree", "kNN")), summary_table)
kable(summary_table, format = "html")
```

::: {.cell-output-display}

`````{=html}
<table>
 <thead>
  <tr>
   <th style="text-align:left;"> Model </th>
   <th style="text-align:right;"> Accuracy </th>
   <th style="text-align:right;"> Precision </th>
   <th style="text-align:right;"> F1 </th>
   <th style="text-align:right;"> Recall </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> NaiveBayes </td>
   <td style="text-align:right;"> 0.867 </td>
   <td style="text-align:right;"> 0.907 </td>
   <td style="text-align:right;"> 0.925 </td>
   <td style="text-align:right;"> 0.944 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> BayesNet </td>
   <td style="text-align:right;"> 0.880 </td>
   <td style="text-align:right;"> 0.900 </td>
   <td style="text-align:right;"> 0.934 </td>
   <td style="text-align:right;"> 0.971 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> SVM </td>
   <td style="text-align:right;"> 0.870 </td>
   <td style="text-align:right;"> 0.870 </td>
   <td style="text-align:right;"> 0.930 </td>
   <td style="text-align:right;"> 1.000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> RandomForest </td>
   <td style="text-align:right;"> 0.872 </td>
   <td style="text-align:right;"> 0.872 </td>
   <td style="text-align:right;"> 0.932 </td>
   <td style="text-align:right;"> 1.000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> DecisionTree </td>
   <td style="text-align:right;"> 0.870 </td>
   <td style="text-align:right;"> 0.870 </td>
   <td style="text-align:right;"> 0.930 </td>
   <td style="text-align:right;"> 1.000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kNN </td>
   <td style="text-align:right;"> 0.876 </td>
   <td style="text-align:right;"> 0.880 </td>
   <td style="text-align:right;"> 0.933 </td>
   <td style="text-align:right;"> 0.993 </td>
  </tr>
</tbody>
</table>

`````

:::
:::
