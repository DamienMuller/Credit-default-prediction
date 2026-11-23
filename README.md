# Credit default prediction with class imbalance

## TLDR
- Logistic regression, linear discriminant analysis (LDA), a decision tree and gradient boosting are compared with several class imbalance remedies, using stratified cross-validation and a held-out test set.
- Tuned gradient boosting is the best model: Gini 0.88, KS 0.74 and average precision 0.89 on the test set, vs. 0.83, 0.69 and 0.85 for a tuned decision tree and approx. 0.74, 0.60 and 0.71-0.72 for logistic regression and LDA.
- A WoE scorecard, the transparent model banks typically deploy, reaches Gini 0.76 and average precision 0.76: better ranked and calibrated than logistic regression, at the cost of 0.12 Gini vs. gradient boosting.
- Class imbalance remedies (class weights, over-/downsampling, SMOTE) do not improve how loans are ranked by risk for any algorithm, and they inflate the predicted probabilities of default (PD): up to 38% mean PD for an actual default rate of 21.8%.

## Objective
Find the combination of algorithm and class imbalance remedy that predicts loan defaults most accurately, both in ranking clients by risk and in estimating their probability of default.

Lenders use such models to estimate the probability that an applicant defaults. Defaults are usually far outnumbered by repaid loans, which is commonly addressed with resampling. In credit risk, the PD value itself feeds provisions, capital and pricing, so a remedy must not distort it.

## Project overview

| | |
|-|-|
| **Problem** | Binary classification with class imbalance (1:4) |
| **Data** | 32,581 loans from Datacamp: 7,108 defaults, 25,473 non-defaults. 7 loans with implausible age or employment length (> 100 years) are dropped. |
| **Target** | `loan_status`: 1 = default, 0 = non-default |
| **Features** | Client: age, income, home ownership, employment length, previous default on file, credit history length. Loan: intent, grade (A-G), amount, interest rate, loan-to-income ratio. |
| **Pre-processing** | Mean imputation of missing values, standardization or Yeo-Johnson transform of numerical features, one-hot encoding. All steps are fitted inside the pipeline on training data only. |
| **Learning process** | Stratified 80/20 train/test split. Models are compared, hyperparameters and thresholds tuned with 5-fold stratified cross-validation on the training set. The test set is used once at the end. |
| **Algorithms** | Logistic regression, LDA (default hyperparameters, confirmed optimal by grid search), decision tree (max depth 12, min. 25 loans per leaf) and gradient boosting (`HistGradientBoostingClassifier`: learning rate 0.03, max. 63 leaves, min. 20 loans per leaf, L2 1.0, early stopping), tuned by grid search |
| **Scorecard** | `optbinning`: optimal monotonic binning, weight of evidence (WoE) logistic regression, 600 points at odds 50:1 and 20 points to double the odds. Features with information value below 0.02 are dropped, and the previous-default flag is excluded for its wrong-sign coefficient. |
| **Variations** | Baseline (standardization), Transform (Yeo-Johnson), and five remedies on standardized features: Class weights, Oversampling, Downsampling, Over- + Downsampling, SMOTE (SMOTENC variant, which keeps categories valid). Resampling runs inside an imblearn pipeline, so it only affects training folds. |
| **Thresholds** | For every model, the threshold maximizing the F-score is selected by cross-validation on the training set (`TunedThresholdClassifierCV`). |
| **Evaluation** | Ranking: Gini, KS, average precision (PR AP). Calibration: Brier score, mean PD vs. actual default rate, calibration curves. Classification: F-score, sensitivity (defaults), precision, specificity (non-defaults). |

## Results

CV PR AP: mean ± std over 5 folds on the training set. All other columns: test set (actual default rate 21.8%). Classification metrics use the threshold tuned on the training set.

### Ranking and calibration

| Model | Variation | CV PR AP | Gini | KS | PR AP | Brier | Mean PD |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **Logistic Regression** | Baseline | 0.725 ± 0.005 | 0.738 | 0.600 | 0.723 | 0.103 | 21.6% |
|  | Transform | 0.715 ± 0.004 | 0.741 | 0.588 | 0.717 | 0.104 | 21.7% |
|  | Class weights | 0.716 ± 0.006 | 0.741 | 0.597 | 0.715 | 0.138 | 37.7% |
|  | Oversampling | 0.717 ± 0.006 | 0.741 | 0.598 | 0.715 | 0.138 | 37.7% |
|  | Downsampling | 0.717 ± 0.005 | 0.740 | 0.594 | 0.715 | 0.138 | 37.8% |
|  | Over- + Downsampling | 0.719 ± 0.006 | 0.740 | 0.598 | 0.719 | 0.115 | 30.6% |
|  | SMOTE | 0.719 ± 0.006 | 0.738 | 0.592 | 0.716 | 0.137 | 37.2% |
| **LDA** | Baseline | 0.708 ± 0.008 | 0.731 | 0.599 | 0.706 | 0.106 | 20.9% |
|  | Transform | 0.693 ± 0.008 | 0.732 | 0.599 | 0.699 | 0.108 | 21.2% |
|  | Class weights | 0.708 ± 0.008 | 0.731 | 0.599 | 0.706 | 0.128 | 33.5% |
|  | Oversampling | 0.716 ± 0.006 | 0.740 | 0.598 | 0.715 | 0.131 | 35.9% |
|  | Downsampling | 0.716 ± 0.005 | 0.740 | 0.598 | 0.716 | 0.132 | 36.0% |
|  | Over- + Downsampling | 0.713 ± 0.007 | 0.737 | 0.600 | 0.713 | 0.112 | 28.6% |
|  | SMOTE | 0.719 ± 0.006 | 0.739 | 0.595 | 0.717 | 0.132 | 35.7% |
| **Decision Tree** | Baseline | 0.853 ± 0.007 | 0.825 | 0.692 | 0.852 | 0.067 | 21.6% |
|  | Transform | 0.853 ± 0.007 | 0.825 | 0.692 | 0.852 | 0.067 | 21.6% |
|  | Class weights | 0.855 ± 0.007 | 0.822 | 0.680 | 0.851 | 0.089 | 31.9% |
|  | Oversampling | 0.847 ± 0.008 | 0.815 | 0.683 | 0.844 | 0.088 | 31.4% |
|  | Downsampling | 0.844 ± 0.006 | 0.811 | 0.669 | 0.830 | 0.097 | 33.4% |
|  | Over- + Downsampling | 0.845 ± 0.007 | 0.814 | 0.689 | 0.842 | 0.075 | 26.9% |
|  | SMOTE | 0.841 ± 0.005 | 0.812 | 0.685 | 0.844 | 0.084 | 30.4% |
| **Gradient Boosting** | Baseline | 0.900 ± 0.005 | 0.882 | 0.738 | 0.892 | 0.057 | 21.2% |
|  | Transform | 0.900 ± 0.005 | 0.882 | 0.738 | 0.892 | 0.057 | 21.2% |
|  | Class weights | 0.897 ± 0.006 | 0.882 | 0.739 | 0.892 | 0.069 | 29.6% |
|  | Oversampling | 0.899 ± 0.006 | 0.887 | 0.748 | 0.898 | 0.059 | 24.9% |
|  | Downsampling | 0.891 ± 0.005 | 0.880 | 0.741 | 0.889 | 0.079 | 31.9% |
|  | Over- + Downsampling | 0.898 ± 0.006 | 0.884 | 0.744 | 0.895 | 0.058 | 23.2% |
|  | SMOTE | 0.899 ± 0.006 | 0.879 | 0.745 | 0.893 | 0.056 | 22.3% |
| **Scorecard** | Baseline | 0.770 ± 0.004 | 0.764 | 0.629 | 0.755 | 0.096 | 21.6% |

### Classification

| Model | Variation | Threshold | F-score (0.5) | F-score | Sensitivity | Precision | Specificity |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **Logistic Regression** | Baseline | 0.32 | 0.646 | 0.675 | 0.695 | 0.656 | 0.898 |
|  | Transform | 0.36 | 0.627 | 0.666 | 0.659 | 0.672 | 0.910 |
|  | Class weights | 0.68 | 0.634 | 0.671 | 0.653 | 0.691 | 0.919 |
|  | Oversampling | 0.65 | 0.634 | 0.671 | 0.678 | 0.664 | 0.904 |
|  | Downsampling | 0.65 | 0.640 | 0.669 | 0.670 | 0.668 | 0.907 |
|  | Over- + Downsampling | 0.53 | 0.672 | 0.671 | 0.678 | 0.664 | 0.904 |
|  | SMOTE | 0.66 | 0.635 | 0.671 | 0.669 | 0.673 | 0.909 |
| **LDA** | Baseline | 0.35 | 0.652 | 0.667 | 0.662 | 0.672 | 0.910 |
|  | Transform | 0.27 | 0.625 | 0.664 | 0.717 | 0.618 | 0.877 |
|  | Class weights | 0.66 | 0.656 | 0.667 | 0.665 | 0.669 | 0.908 |
|  | Oversampling | 0.65 | 0.647 | 0.673 | 0.663 | 0.683 | 0.914 |
|  | Downsampling | 0.64 | 0.648 | 0.670 | 0.667 | 0.674 | 0.910 |
|  | Over- + Downsampling | 0.53 | 0.672 | 0.671 | 0.664 | 0.678 | 0.912 |
|  | SMOTE | 0.65 | 0.645 | 0.674 | 0.669 | 0.679 | 0.912 |
| **Decision Tree** | Baseline | 0.51 | 0.781 | 0.781 | 0.676 | 0.925 | 0.985 |
|  | Transform | 0.51 | 0.781 | 0.781 | 0.676 | 0.925 | 0.985 |
|  | Class weights | 0.79 | 0.735 | 0.788 | 0.673 | 0.950 | 0.990 |
|  | Oversampling | 0.81 | 0.748 | 0.787 | 0.675 | 0.944 | 0.989 |
|  | Downsampling | 0.80 | 0.722 | 0.769 | 0.652 | 0.936 | 0.988 |
|  | Over- + Downsampling | 0.73 | 0.774 | 0.782 | 0.671 | 0.938 | 0.988 |
|  | SMOTE | 0.85 | 0.773 | 0.782 | 0.654 | 0.971 | 0.995 |
| **Gradient Boosting** | Baseline | 0.49 | 0.811 | 0.811 | 0.704 | 0.956 | 0.991 |
|  | Transform | 0.49 | 0.811 | 0.811 | 0.704 | 0.956 | 0.991 |
|  | Class weights | 0.68 | 0.798 | 0.815 | 0.719 | 0.941 | 0.987 |
|  | Oversampling | 0.72 | 0.814 | 0.824 | 0.718 | 0.967 | 0.993 |
|  | Downsampling | 0.81 | 0.783 | 0.801 | 0.694 | 0.947 | 0.989 |
|  | Over- + Downsampling | 0.62 | 0.816 | 0.816 | 0.722 | 0.938 | 0.987 |
|  | SMOTE | 0.60 | 0.821 | 0.822 | 0.717 | 0.962 | 0.992 |
| **Scorecard** | Baseline | 0.38 | 0.680 | 0.698 | 0.685 | 0.711 | 0.922 |

### Calibration curves (test set)

Without remedy (thick line), predicted PDs match observed default rates. Remedies lie below the diagonal, i.e. overestimate PD.

![Calibration curves](images/calibration.png)

## Key findings
- **Gradient boosting ranks loans best.** It reaches a CV average precision of 0.900, vs. 0.853 for the tuned decision tree and 0.71-0.73 for the linear models. Tuning its hyperparameters matters little (0.894-0.900 over the whole grid).
- **Limiting tree complexity matters for a single tree.** It raises the tree's CV average precision from 0.61 (unlimited tree) to 0.85.
- **A scorecard trades some performance for transparency.** Binning captures non-linear effects within each feature, so the scorecard beats logistic regression on standardized features (Gini 0.764 vs. 0.738, Brier 0.096 vs. 0.103) with only 8 features and a readable points table. Gradient boosting, which also models interactions between features, remains clearly better (Gini 0.882).
- **Remedies do not improve ranking.** For every algorithm, the CV average precision of the remedies is within approx. 0.01 of the baseline.
- **Remedies distort PD.** They inflate the mean predicted PD strongly for the linear models (29-38% vs. 21.8% actual), moderately for the decision tree (27-33%) and, depending on the remedy, for gradient boosting (22-32%). They never meaningfully improve the Brier score. Models trained on the original class partition are well calibrated (mean PD 20.9-21.7%).
- **Threshold tuning is the better remedy.** To classify loans, tuning the threshold reaches F-scores similar to resampling without distorting PD, e.g. baseline logistic regression goes from 0.65 at the 0.5 cutoff to 0.68.
- **SMOTE needs care with categorical features.** Plain SMOTE on one-hot encoded features creates impossible fractional categories (38.5% of synthetic loans), which flexible models learn to recognize. SMOTENC avoids this.
- **Standardization is enough.** Logistic regression needs standardized features to converge (the solver did not converge on raw features even after more than 12,000 iterations). Removing skewness with a Yeo-Johnson transform does not improve the linear models further.

## Run it
```bash
uv sync
uv run python -m credit_risk.train    # full comparison (approx. 3 min), saves results, curves and PD models to artifacts/
```
`Loan_prediction.ipynb` contains the exploratory analysis and the same comparison step by step, including hyperparameter tuning (select the `.venv` kernel).

## References
- Kuhn, M. & Johnson, K., 2013. Remedies for Severe Class Imbalance. In: Applied Predictive Modeling. New York: Springer, pp. 419-442.
- van den Goorbergh, R., van Smeden, M., Timmerman, D. & Van Calster, B., 2022. The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. Journal of the American Medical Informatics Association, 29(9), pp. 1525-1534.
