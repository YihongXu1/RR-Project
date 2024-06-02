# ML methods in Covid-19 Classification

## Purpose

This repo is to reproduce the main findings of a published paper [Classification of Covid-19 Dataset with Some Machine Learning Methods](https://dergipark.org.tr/en/pub/jauist/issue/55760/748667). Data [COVID-19 Mexico Patient Health Dataset](https://www.kaggle.com/datasets/riteshahlawat/covid19-mexico-patient-health-dataset/data) is downloaded from Kaggle. We aim to do a binary classification problem, where 1 indicates the patient is alive while 0 indicates the patient die.

## Version Control

- This package is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

- All packages related to this repo is listed in [session_info.txt](code/session_info.txt)

## Main Result

The summary table we calculate and the one in the paper is as below. They are not actually totally the same. (The second table is copied from the paper, while the first is generated in this repo)


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

$$
\begin{array}{clllll}
\hline \begin{array}{c}
\text { Classification } \\
\text { Tecnique }
\end{array} & \begin{array}{l}
\text { Accuracy } \\
\mathbf{( \% )}
\end{array} & \text { Precision } & \text { F-Measure } & \text { Recall } & \begin{array}{l}
\text { Classfier } \\
\text { Name }
\end{array} \\
\hline & 99.7704 & 0,999 & 0,999 & 0,999 & \text { BayesNet } \\
\text { Naive Bayes } & 98.9931 & 0,997 & 0,995 & 0,993 & \text { NaiveBayes } \\
\hline \begin{array}{c}
\text { Support } \\
\text { Vector } \\
\text { Machine }
\end{array} & 100 & 1,000 & 1,000 & 1,000 & \text { SMO } \\
\hline & 99.9812 & 1,000 & 1,000 & 1,000 & \begin{array}{l}
\text { Random } \\
\text { Forest }
\end{array} \\
\hline \text { Trees } & & & & 1,000 & \text { J48 } \\
\hline \begin{array}{l}
\text { K Nearest } \\
\text { Neighbor }
\end{array} & 99.79 & 0,999 & 0,999 & 0,999 & \text { IBK } \\
\hline
\end{array}
$$



Results are not totally the same as shown in the paper. The paper states that SVM renders the best accuracy score, all scores are rather close to 1, while ours is not the same. kNN is the best accordin to accuracy score. recall of SVM, random forest and decision tree is 1 (the same as the paper), while accuracy and precision could not reach 1. Possible reasons of the difference are listed below:

- There are different data preprocessing methods. As some models cannot handle NA values, what to do with NA recording matters. Details of preprocessing is not available in the paper, so we may get different data to fit the model.

- Seeds are also different. In our study, we fix the seed to ensure reproducibility. But the seed is not the same as the paper. In this cased, for each fold of the cross validation method, we got different subsets of data from the ones used in the paper.

- Different program languages also matter. The paper used WEKA, while we use R. Underlying implementation varies.

- Subtypes of the methods are not totally the same. For example, SMO version of SVM is used in the paper. However, we could not do exactly the same, instead we apply radial basis kernel. There are some ambiguity is the paper which prevents us from reproducing as well. As we known, NaiveBayes and BayesNet adopt different assumptions, but the original paper refer to both method as NaiveBayes.

In conclusion, we cannot reproduce every details of the original paper without specification. However, results of our own can be reproduced since random seed is fixed and every detail is displayed in the chunks above. To reproduce all results at once, run `reproduce.r`.

## Instructions

The structure of this repo is listed as below:

- code
  - analysis.*: files of detailed analysis in this repo. You can read `analysis.html` at first.
  - reproduce.r: all main steps of the analysis is summarized in this sinle executable file.
  - renv: environment to reproduce the same result as the analysis
  - session_info.txt: all session info of the environment
- Data/patients.csv: data downloaded from Kaggle.
- ref：original paper file
- .gitignore: files which should be ignored by git. You can reproduce all result without these files
- LICENSE: copyright
- README.md: the general introduction of this repo and analysis, which you're reading now.

## Reference

- [.gitignore template](https://github.com/github/gitignore/blob/main/R.gitignore)


