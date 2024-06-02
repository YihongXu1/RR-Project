# This file is used for repruducing all results in `analysis.Qmd` at once
# Packages needed are listed as below

library(tidyverse) # data cleaning
library(knitr) # display tables in good format
library(caret) # confusion matrics
library(bnclassify) # BN
library(kernlab) # SVM
library(randomForest) # random forest
library(C50) # C5.0 decision tree

# set hyperparameters
path_csv <- "../Data/patient.csv"
seed <- 9

# load data
df <- read.csv(path_csv)
dim(df)
head(df)

# data cleaning
df_ageori <- df %>% mutate(y = factor(ifelse(death_date == "9999-99-99", 0, 1),
                                      labels = c("live", "die")),
                           pregnant = factor(ifelse(pregnant == 1, 1, 2)),
                           across(c(sex, patient_type, intubated, pneumonia,
                                    diabetes, copd, asthma, immunosuppression, hypertension,
                                    other_diseases, cardiovascular, obesity, chronic_kidney_failure,
                                    smoker, another_case, icu, outcome), ~ factor(ifelse(.>2, NA, .)))) %>%
  select(-c(death_date))
df_agefac <- df_ageori %>% mutate(age = factor(age))
head(df_agefac)

# Model
## NaiveBayes
df <- na.omit(df_agefac) # use factorized age version 
set.seed(seed)
model_nb <- nb('y', df)
model_nb <- lp(model_nb, df, smooth = 1) # learn parameter
cv(model_nb, df, k = 10) # cross validation
pred_nb <- predict(model_nb, df)
confusionMatrix(pred_nb, df$y)

## BayesNet
set.seed(seed)
model_bn <- tan_cl('y', df, score = 'aic')
model_bn <- lp(model_bn, df, smooth = 1)
cv(model_bn, df, k = 10)
pred_bn <- predict(model_bn, df)
confusionMatrix(pred_bn, df$y)

## SVM
df <- na.omit(df_ageori)
set.seed(seed)
train_control <- trainControl(method = "cv", number = 10)
model_svm <- train(y ~ ., data = df, method = "svmRadial", trControl = train_control)
pred_svm <- predict(model_svm, df)
confusionMatrix(pred_svm, df$y)

## RandomForest
set.seed(seed)
df <- na.omit(df_ageori)
train_control <- trainControl(method = "cv", number = 10)
model_rf <- train(y ~ ., data = df, method = "rf", trControl = train_control)
pred_rf <- predict(model_rf, df)
confusionMatrix(pred_rf, df$y)

## Decision Tree
set.seed(seed)
train_control <- trainControl(method = "cv", number = 10)
model_tree <- train(y ~ ., data = df, method = "C5.0", trControl = train_control)
pred_tree <- predict(model_tree, df)
confusionMatrix(pred_tree, df$y)

## kNN
set.seed(seed)
train_control <- trainControl(method = "cv", number = 10)
model_knn <- train(y ~ ., data = df, method = "knn", trControl = train_control)
pred_knn <- predict(model_knn, df)
confusionMatrix(pred_knn, df$y)

# Summary of all results
# get all scores of a model in a function
get_all_scores <- function(pred, y = df$y){
  acc <- accuracy(pred, y)
  prec <- precision(pred, y)
  rec <- recall(pred, y)
  F1 <- 2*(prec*rec)/(prec+rec)
  return(c(acc, prec, F1, rec))
}

pred_list <- list(pred_nb, pred_bn, pred_svm, 
                  pred_rf, pred_tree, pred_knn)
summary_table <- do.call(rbind, lapply(pred_list, get_all_scores))
summary_table <- round(summary_table, 3)
colnames(summary_table) <- c("Accuracy", "Precision", "F1", "Recall")
summary_table <- cbind(data.frame(Model = c("NaiveBayes", "BayesNet", "SVM", "RandomForest", "DecisionTree", "kNN")), summary_table)
summary_table












