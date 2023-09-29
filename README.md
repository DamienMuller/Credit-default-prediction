# Credit default prediction with class imbalance using linear and non-linear models
<br>

## TLDR
---
- Logistic regression, Linear discriminant analysis and decision tree are tested with different class imbalance remedies. 
- Logistic regression (with data transformation) and Linear discriminant analysis (with raw data and up-down sampling) are overall best predictors for loan defaults (average precision). 
- Decision tree outperforms linear models when threshold levels are used for predictions (F-score).  

## Project Objective 
---
The objective of this project is to find the right combination of machine learning algorithm and class imbalance remedy that leads to the most accurate cedit default predictions. 


## Rationale 
---
Among other things, financial institutions use forecasting models to assess the probability of default of a loan applicant. These prediction models are created using data records that contain information about previous borrowers and whether or not the loans have been repaid. Specifically, the models learn the relationship between characteristics of borrowers (e.g. demographic and financial) and their respective loan status (default or non-default).  
One common issue with such datasets is that the number of non-performing loans is largely outnumbered by the number of repaid loans. This class imbalance results in forecasting models having difficulties to learn the specific characteristics of non-performing loans, as these patterns are overlaid by the characteristics of the dominant class (non-default loans). This in turn leads to an underestimate of loan defaults, and thus of the overall risk of default, which can have significant financial consequences for credit institutions. 


## Project overview
---
  
| | |
|-|-|
| **Predictive problem** | Binary classification problem with class imbalance |
| **Data** | Dataset with 32581 observations of bank client characteristics and loan details with associated loan status (default or non-default). Dataset was retrieved from Datacamp. |
| **Output variable** | *loan_status*: Binary variable with 1 = loan default and 0 = non-default. <br><br> Number of defaults: 7108 <br> Number of non-defaults: 25473 <br><br> Class imbalance 1:4 |
| **X-matrix** |Client characteristics <br><br> *person_age*: Age of client <br> *person_income*: Annual income <br> *person_home_ownership*: Type of home ownership (rent, own property, mortgage repayment, other) <br> *person_emp_length*: Number of years working <br>  *cb_person_default_on_file*: Whether client has defaulted before <br> *cb_person_cred_hist_length*: Client credit history (in years) <br><br> Loan characteristics <br><br> *loan_intent*: Loan spending (personal, education, medical, venture, home improvement, debt consolidation) <br> *loan_grade*: Loan quality score (A-G) <br> *loan_amnt*: Total principal amount <br> *loan_int_rate*: Fixed interest rate on loan <br> *loan_percent_income*: $\frac{loan\_amnt}{person\_income}$ <br> |
| **Learning process** | Stratified sampling is used to split the data into three parts while ensuring that each part has the same class partition: <ul> <li> Training set (70%): Fit data transforms and find optimal model parameters <li> Evaluation set (10%): Test trained models and extract cutoff rate that maximizes f-score (except for 1st and 2nd model variation, see below) <li> Test set (20%): Evaluate fitted models with optimal cutoff rates from evaluation set|
| **Algorithms** | Logistic regression <br> Linear Discriminant Analysis <br> Decision Tree|
| **Model variations** | For each algorithm, the following variations are tested: <ol> <li> Baseline model: Fit basic data transform (training set) &rarr; Fit model parameters (training set) &rarr; Predict evaluation set &rarr; Predict test set (default cutoff).  <li> Models with data transforms: Fit advanced data transforms (training set) &rarr; Fit model parameters (training set) &rarr; Predict evaluation set &rarr; Predict test set (default cutoff). <li> Alternate cutoffs: Fit advanced data transforms (training set) &rarr; Fit model parameters (training set) &rarr; Predict evaluation set + extract cutoff that maximizes F-score &rarr; Predict test set with optimal cutoff. <li> Oversampling: Oversample minority class to get 1:1 class partition (training set, random sampling with replacement) &rarr; Fit advanced data transforms (training set) &rarr; Fit model parameters (training set) &rarr; Predict evaluation set + extract cutoff that maximizes F-score &rarr; Predict test set with optimal cutoff. <li> Downsampling: Randomly delete non-default observations to get 1:1 class partition (training set) &rarr; Fit advanced data transforms (training set) &rarr; Fit model parameters (training set) &rarr; Predict evaluation set + extract cutoff that maximizes F-score &rarr; Predict test set with optimal cutoff. <li> Over- and Downsampling: Randomly oversample minority class to get 1:2 class partition, then randomly downsample majority class to get approx. 1:1.67 partition (training set) &rarr; Fit advanced data transforms (training set) &rarr; Fit model parameters (training set) &rarr; Predict evaluation set + extract cutoff that maximizes F-score &rarr; Predict test set with optimal cutoff. <li> SMOTE: Oversample minortiy class by generating new synthetic minority observations (interpolation of 5 nearest neighbors) to get 1:1 class partition (training set) &rarr; Fit advanced data transforms (training set) &rarr; Fit model parameters (training set) &rarr; Predict evaluation set + extract cutoff that maximizes F-score &rarr; Predict test set with optimal cutoff. 
| **Evaluation method** | The algorithm and model variation that has a high overall minority class accuracy (as measured by average precision) while keeping a high majority class accuracy (specificity) will be considered a good model. |
<br>
    
## Results
---

### Evaluation set performance
<br>



|   |   |  ROC AUC  | PR AP |  F - score  |  Sensitivity  |  Precision   |  Specificity  |
|----|-----|:----:|:----: |:----------:|:-------:|:------------:|:-------------:|
|**Logistic Regression**| Baseline | 0.68 | <font color="red"> 0.36 | <font color="red">0.50 | 0.51 | 0.50 | 0.86 |
|    | Transform | 0.86 |<font color="green"> 0.70 | 0.61 | 0.52 | 0.74 | <font color="green"> 0.95 | 
|    | Cutoff | 0.86 |<font color="green"> 0.70 |<font color="green"> 0.66 | 0.67 | 0.64 |<font color="green"> 0.90 | 
|    | Oversampling | 0.86 | 0.70 | 0.64 | 0.75 | 0.55 | 0.83 |
|    | Downsampling | 0.86 | 0.70 | 0.65 | 0.71 | 0.60 | 0.87 |
|    | Over- + Downsampling | 0.86 | 0.70 | 0.64 | 0.63 | 0.65 | 0.90 |
|    | SMOTE | 0.86 | 0.69 | 0.63 | 0.72 | 0.56 | 0.84 | 
|**LDA**| Baseline | 0.86 |<font color="green"> 0.70 |<font color="green"> 0.65 | 0.58 | 0.73 | <font color="green">0.94 |
|    | Transform | 0.86 | 0.68 | 0.60 | 0.52 | 0.72 | 0.94 |
|    | Cutoff | 0.86 | 0.68 | 0.67 | 0.70 | 0.64 | 0.89 |
|    | Oversampling | 0.86 | 0.70 | 0.63 | 0.77 | 0.54 | 0.82 |
|    | Downsampling | 0.86 | 0.70 | 0.63 | 0.77 | 0.53 | 0.81 | 
|    | Over- + Downsampling | 0.86 | <font color="green"> 0.70 | <font color="green"> 0.66 | 0.69 | 0.64 | <font color="green"> 0.89 |
|    | SMOTE | 0.86 | 0.70 | 0.63 | 0.76 | 0.53 | 0.81 | 
|**Decision Tree**| Baseline | 0.84 | 0.60 | 0.74 | 0.76 | 0.73 | 0.92 |
|    | Transform | 0.84 | 0.61 | 0.74 | 0.76 | 0.73 | 0.92 |
|    | Cutoff | 0.84 | 0.61 | 0.74 | 0.76 | 0.73 | 0.92 |
|    | Oversampling | 0.85 | <font color="green">0.63 | <font color="green">0.76 | 0.76 | 0.75 | <font color="green">0.93 |
|    | Downsampling | 0.82 | <font color="red"> 0.49 | <font color="red">0.66 | 0.81 | 0.55 | <font color="red">0.82 | 
|    | Over- + Downsampling | 0.84 | 0.58 | 0.73 | 0.78 | 0.69 | 0.90 |
|    | SMOTE | 0.83 | 0.56 | 0.71 | 0.75 | 0.68 | 0.90 | 

<br>

### Test set performance 
<br>
    
|   |   |  ROC AUC  | PR AP |  F - score  |  Sensitivity  |  Precision   |  Specificity  |
|----|-----|:----:|:----: |:----------:|:-------:|:------------:|:-------------:|
|**Logistic Regression**| Baseline | 0.67 | <font color="red"> 0.35 | <font color="red">0.49 | 0.50 | 0.48 | <font color="red">0.85 |
|    | Transform | 0.86 | <font color="green"> 0.70 | <font color="green">0.62 | 0.53 | 0.75 | <font color="green">0.95 | 
|    | Cutoff | 0.86 |<font color="green"> 0.70 |<font color="green"> 0.66 | 0.68 | 0.65 | 0.90 | 
|    | Oversampling | 0.87 | 0.70 | 0.67 | 0.69 | 0.65 | 0.90 |
|    | Downsampling | 0.86 | 0.70 | 0.67 | 0.68 | 0.65 | 0.90 |
|    | Over- + Downsampling | 0.86 | 0.70 | 0.66 | 0.63 | 0.68 | 0.92 |
|    | SMOTE | 0.86 | 0.70 | 0.65 | 0.65 | 0.65 | 0.90 | 
|**LDA**| Baseline | 0.86 | <font color="green"> 0.71 | <font color="green"> 0.65 | 0.57 | 0.74 | 0.94 |
|    | Transform | 0.86 | 0.69 | 0.61 | 0.52 | 0.73 | 0.95 | 
|    | Cutoff | 0.86 | 0.69 | 0.66 | 0.70 | 0.63 | 0.89 |
|    | Oversampling | 0.87 | 0.71 | 0.67 | 0.68 | 0.66 | 0.90 |
|    | Downsampling | 0.87 | 0.70 | 0.67 | 0.69 | 0.64 | 0.89 | 
|    | Over- + Downsampling | 0.87 | <font color="green"> 0.70 | <font color="green"> 0.67 | 0.67 | 0.67 | <font color="green"> 0.91 |
|    | SMOTE | 0.86 | 0.70 | 0.66 | 0.67 | 0.65 | 0.90 | 
|**Decision Tree**| Baseline | 0.85 | <font color="green">0.62 | <font color="green">0.76 | 0.78 | 0.74 | 0.92 |
|    | Transform | 0.85 | <font color="green">0.62 | <font color="green">0.76 | 0.78 | 0.73 | 0.92 | 
|    | Cutoff | 0.85 | <font color="green">0.62 | <font color="green">0.76 | 0.78 | 0.73 | 0.92 |
|    | Oversampling | 0.84 | 0.61 | 0.75 | 0.76 | 0.74 | 0.93 |
|    | Downsampling | 0.81 | <font color="red"> 0.48 | <font color="red"> 0.65 | 0.81 | 0.54 | <font color="red"> 0.81 | 
|    | Over- + Downsampling | 0.84 | 0.59 | 0.73 | 0.77 | 0.70 | 0.91 |
|    | SMOTE | 0.84 | 0.59 | 0.74 | 0.77 | 0.70 | 0.91 | 


## Key Findings 
---

<ul>
    <li> Logistic regression and Linear discriminant analysis show the overall best accuracy for predicting loan defaults irrespective of threshold levels. This is true for both the evaluation and test sets. Both models also showed a high majority class accuracy (specificity $\geq$ 0.9).
    <li> When default thresholds are considered (0.5 for linear models), decision tree outperforms linear models in terms of minority (default) class predictions with a higher F-score in the evaluation set. This remains true for the test set even after cutoff rates are optimized for linear models (as optimal cutoff rates from evaluation set are used to predict test set classes). 
    <li> Data transformation alone was highly effective in increasing default predictions for logistic regression. The average precision score increased by 100% in the evaluation and test sets after transforming raw data. Optimizing thresholds also results in consistent default and non-default predictions. 
    <li> Although raw input variables are skewed and have different variance, LDA performs surprisingly well without any data transformations. The mixture of over- and downsampling also showed consistent performance for minority and majority classes. 
    <li> Decision tree performance similar for baseline, data transformation, alternate cutoff and oversampling variations. However, downsampling significantly reduced majority and minority class accuracies. 

<!-- -->

## References
---
