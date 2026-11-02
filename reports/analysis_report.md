# Kantar Data Science Take-Home: Fraud Detection

## Executive summary

This analysis investigates registration activity for concentrated fraud events and builds binary classifiers to predict the supplied `is_fraud` target. The dataset contains 50,000 registrations and the overall fraud prevalence is 5.00%. The analysis is intended as a transparent baseline; model scores should not be interpreted as proof of fraud.

## 1. Data and quality checks

The supplied workbook contains 13 columns and 50,000 rows. `user_id` is an identifier and is excluded from model predictors. `registration_timestamp` is parsed as a datetime and transformed into time-derived features. `education_level` has missing values, so the modeling pipeline imputes missing categorical values using the most frequent training-set category. All preprocessing is fitted within each model pipeline to avoid learning imputation or encoding rules from the test set.

The target is imbalanced: 95.0% legitimate and 5.0% fraudulent registrations. Consequently, accuracy alone is not an adequate measure. Precision, recall, F1, ROC-AUC and average precision (PR-AUC summary) are reported.

## 2. Task 1 – Anomaly detection

Daily aggregation identified **2025-01-11** as the date with the highest observed fraud concentration: 574 fraudulent registrations among 2,128 registrations, a fraud rate of 26.97%. Hourly investigation narrowed the main event to **2025-01-11 01:00:00 through 03:59:59**. During this window, 514 of 728 registrations were labelled fraudulent (70.60%).

Outside the candidate window, the mean hourly registration volume was 68.72 (standard deviation 8.29); the candidate-window mean was 242.67. This large increase supports the interpretation of a concentrated event. The volume Z-score is descriptive and assumes a suitable baseline; hourly patterns, seasonality, overdispersion and multiple testing should be addressed in a production detector.

### Detecting events without labels

When fraud labels are not yet available, the system cannot calculate a fraud rate. It can instead monitor observable activity: registrations per time interval, registration duration distributions, domain mix, device/OS mix and other feature distributions. A practical layered design is: (1) time-aware rolling baselines or robust control limits for volume spikes; (2) Isolation Forest on aggregated behavioural features to flag unusual combinations; (3) distribution-shift measures such as PSI, KS statistics or Jensen–Shannon divergence; and (4) human investigation and feedback from later-confirmed outcomes. Isolation Forest flags unusual observations, not confirmed fraud. Its contamination setting is an operational assumption and should be calibrated.

The included Isolation Forest example uses hourly registration volume and time context without passing `is_fraud` to the algorithm. Its output is an anomaly-review queue, not a fraud classifier.

## 3. Task 2 – Fraud classification

### Preprocessing and feature engineering

The modeling dataset excludes `user_id` because it is a unique identifier and has no defensible generalizable meaning. The raw timestamp is transformed into hour of day, day of week, weekend indicator and sine/cosine hour encodings. Registration duration is log-transformed to reduce skew. Numeric missing values are median-imputed; categorical missing values are mode-imputed; categorical features are one-hot encoded with unknown categories handled safely. Logistic Regression additionally standardizes numeric features.

The data is divided into a stratified 80% training set and 20% test set using random state 42. Stratification preserves the target prevalence in both partitions. The test set is held out from fitting and preprocessing. Logistic Regression uses class weights as an interpretable baseline; Random Forest uses balanced subsampling to account for class imbalance. No synthetic oversampling is applied.

### Model results

| Model | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.366 | 0.886 | 0.518 | 0.968 | 0.774 |
| random_forest | 0.727 | 0.662 | 0.693 | 0.958 | 0.758 |

The model selected for the summary is **logistic_regression**, chosen by average precision on the held-out test set. Its confusion matrix (rows = actual legitimate/fraud; columns = predicted legitimate/fraud) is `[[8731, 769], [57, 443]]`. Precision describes the proportion of alerts that are fraud; recall describes the proportion of labelled fraud cases detected. The operating threshold should be selected with the business cost of false positives and missed fraud in mind, rather than assuming 0.5 is optimal.

### Evaluation limitations

The random stratified split estimates performance when train and test examples are drawn from a similar population. The dataset contains a pronounced, time-localized event; a random split can place examples from the same event in both sets and may therefore overstate performance on a genuinely future period. A next step is a chronological holdout and, ideally, validation on later months or a separate campaign/market. The single supplied month is insufficient to establish robust seasonality or long-term drift.

## 4. Further Steps

1. Obtain additional months and independently labelled outcomes to test temporal and market generalization.
2. Use a chronological validation design and report performance by week, device, domain and other relevant cohorts.
3. Tune the alert threshold against the operational cost of false positives, missed fraud and investigation capacity.
4. Review feature importance and use SHAP or permutation importance with care; association is not causation and sensitive/proxy attributes require governance review.
5. Monitor feature drift, alert volume, precision/recall after labels mature, and model calibration.
6. Investigate the 11 January event using available network, IP, session, device-fingerprint and referral data; these fields are not present in the supplied dataset and are not assumed in this analysis.
7. Add human review, audit trails, privacy controls and a retraining process before any automated blocking decision.

## Reproducibility

Run the scripts in the order listed in `README.md`. The random seed, split proportion, preprocessing pipeline and model parameters are recorded in `outputs/classification_metrics.json`. Generated tables and figures are stored under `outputs/`.
