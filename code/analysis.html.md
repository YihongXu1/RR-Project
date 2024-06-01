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

-   SMO (SVM)

-   Random Forest

-   Decision Tree (C4.5)

-   KNN

## NaiveBayes


::: {.cell}

```{.r .cell-code}
df <- df_agefac
set.seed(seed)
model_nb <- nb('y', df)
model_nb <- lp(model_nb, df, smooth = 1)
cv(model_nb, df, k = 10)
pred <- predict(model_nb, df)
confusionMatrix(pred, df$y)
```
:::

::: {.cell}

:::

::: {.cell}
::: {.cell-output .cell-output-stdout}

```
Confusion Matrix and Statistics

          Reference
Prediction  live   die
      live 85674  1477
      die   6730  1958
                                          
               Accuracy : 0.9144          
                 95% CI : (0.9126, 0.9161)
    No Information Rate : 0.9642          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.2864          
                                          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9272          
            Specificity : 0.5700          
         Pos Pred Value : 0.9831          
         Neg Pred Value : 0.2254          
             Prevalence : 0.9642          
         Detection Rate : 0.8939          
   Detection Prevalence : 0.9093          
      Balanced Accuracy : 0.7486          
                                          
       'Positive' Class : live            
                                          
```


:::
:::



## BayesNet


::: {.cell}

```{.r .cell-code}
set.seed(seed)
model_bn <- tan_cl('y', df, score = 'aic')
model_bn <- lp(model_bn, df, smooth = 1)
cv(model_bn, df, k = 10)
pred <- predict(model_bn, df)
confusionMatrix(pred, df$y)
```
:::

::: {.cell}

:::

::: {.cell}
::: {.cell-output .cell-output-stdout}

```
Confusion Matrix and Statistics

          Reference
Prediction  live   die
      live 91376  2746
      die   1028   689
                                          
               Accuracy : 0.9606          
                 95% CI : (0.9594, 0.9618)
    No Information Rate : 0.9642          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.2495          
                                          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9889          
            Specificity : 0.2006          
         Pos Pred Value : 0.9708          
         Neg Pred Value : 0.4013          
             Prevalence : 0.9642          
         Detection Rate : 0.9534          
   Detection Prevalence : 0.9821          
      Balanced Accuracy : 0.5947          
                                          
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
pred <- predict(model_svm, df)
confusionMatrix(pred, df$y)
```
:::

::: {.cell}

:::

::: {.cell}
::: {.cell-output .cell-output-stdout}

```
Confusion Matrix and Statistics

          Reference
Prediction  live   die
      live 91376  2746
      die   1028   689
                                          
               Accuracy : 0.9606          
                 95% CI : (0.9594, 0.9618)
    No Information Rate : 0.9642          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.2495          
                                          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9889          
            Specificity : 0.2006          
         Pos Pred Value : 0.9708          
         Neg Pred Value : 0.4013          
             Prevalence : 0.9642          
         Detection Rate : 0.9534          
   Detection Prevalence : 0.9821          
      Balanced Accuracy : 0.5947          
                                          
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
pred <- predict(model_rf, df)
confusionMatrix(pred, df$y)
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
pred <- predict(model_tree, df)
confusionMatrix(pred, df$y)
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


## IBK (KNN)




::: {.cell}

```{.r .cell-code}
set.seed(seed)
train_control <- trainControl(method = "cv", number = 10)
model_knn <- train(y ~ ., data = df, method = "knn", trControl = train_control)
pred <- predict(model_knn, df)
confusionMatrix(pred, df$y)
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
















