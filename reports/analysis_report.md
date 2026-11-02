# Kantar Data Science Take-Home: Fraud Detection

## Executive Summary

This analysis investigates registration activity for concentrated anomalous events and builds binary classification models to predict the supplied `is_fraud` target.

The dataset contains 50,000 registrations, with an overall fraud prevalence of 5.00%. The analysis uses two complementary approaches:

- **Anomaly detection:** Identifies unusual registration-volume patterns without using fraud labels to select the candidate event.
- **Fraud classification:** Uses the supplied historical fraud labels to train and evaluate predictive models.

The analysis is intended as a transparent baseline. An anomaly flag or model prediction should not be interpreted as proof of fraud. Any operational deployment would require further validation, monitoring and human review.

## 1. Data and Quality Checks

The supplied workbook contains 13 columns and 50,000 rows. `user_id` is an identifier and is excluded from model predictors. `registration_timestamp` is parsed as a datetime and transformed into time-derived features.

`education_level` contains missing values. The modeling pipeline imputes missing categorical values using the most frequent category learned from the training data. Numeric missing values are handled through median imputation.

All preprocessing steps are fitted within each model pipeline to avoid learning imputation or encoding rules from the test set.

The target is imbalanced:

- Legitimate registrations: 95.0%
- Fraudulent registrations: 5.0%

Consequently, accuracy alone is not an adequate evaluation measure. Precision, recall, F1-score, ROC-AUC and average precision are reported.

## 2. Task 1 – Anomaly Detection

### 2.1 Daily and hourly registration analysis

Daily aggregation identified **11 January 2025** as the day with the highest registration volume:

- Total registrations: 2,128
- Fraud-labelled registrations: 574
- Retrospective fraud rate: 26.97%

To identify the most concentrated registration spike without relying on fraud labels, the analysis calculated a rolling three-hour registration total across the hourly time series. The three-hour window with the highest combined registration volume was selected as the candidate event window.

The selected window was:

**11 January 2025, 01:00–03:59 (exclusive end: 04:00).**

Its retrospective results were:

- Total registrations: 728
- Fraud-labelled registrations: 514
- Retrospective fraud rate: 70.60%

Importantly, the candidate window was selected using registration volume, not the fraud labels. The labels were used afterwards to assess how strongly the selected event was associated with known fraud in this historical dataset.

### 2.2 Registration-volume anomaly scoring

The mean hourly registration volume outside the selected candidate window was 68.72, with a standard deviation of 8.29.

The candidate window had an average hourly volume of 242.67 registrations, approximately 3.53 times the baseline mean.

The three hourly Z-scores within the candidate window were:

| Hour | Registration-volume Z-score |
|---|---:|
| 01:00 | 21.37 |
| 02:00 | 21.86 |
| 03:00 | 19.69 |

All three values exceed the illustrative threshold of 3 standard deviations, indicating an unusually high registration volume relative to the calculated baseline.

The complete hourly analysis flagged four hours using the Z-score threshold. The additional flagged hour falls outside the selected three-hour candidate window.

This Z-score approach is a simple descriptive baseline. It assumes that the comparison distribution is sufficiently stable. A production detector should account for time-of-day patterns, seasonality, overdispersion, changes in traffic and the effects of repeated testing.

### 2.3 Isolation Forest

Isolation Forest was also implemented as an unsupervised anomaly detection method.

The model uses hourly registration volume and time-context features, including:

- Total registrations per hour
- Hour of day
- Day of week
- Sine and cosine transformations of hour of day

The `is_fraud` target is not passed to Isolation Forest.

With 300 trees and a contamination assumption of 2%, the model flagged 15 hourly observations as anomalous.

The contamination parameter is a modelling assumption that influences the number of observations flagged. It is not an estimate of the actual fraud prevalence.

The Z-score and Isolation Forest results are not expected to match exactly. Z-score identifies unusually high registration volume relative to a global baseline, whereas Isolation Forest identifies observations that are unusual in the feature space. The latter can therefore flag unusual combinations of volume and time context, including observations that are not necessarily volume spikes.

Both methods produce observations for investigation rather than confirmed fraud classifications.

### 2.4 Detecting events without fraud labels

When fraud labels are unavailable, a system cannot calculate a fraud rate or directly measure the number of fraudulent registrations. It can instead monitor observable activity and identify deviations from expected behaviour.

A practical layered approach would include:

1. **Time-aware volume monitoring:** Use rolling baselines, robust control limits or time-series models to detect unexpected registration spikes.
2. **Behavioural anomaly detection:** Apply Isolation Forest or similar unsupervised methods to aggregated behavioural features and unusual combinations of attributes.
3. **Distribution-shift monitoring:** Track changes in feature distributions using Population Stability Index (PSI), Kolmogorov–Smirnov statistics or Jensen–Shannon divergence.
4. **Investigation and feedback:** Send suspicious patterns for human review and use subsequently confirmed outcomes to evaluate and improve detection.

The current Isolation Forest implementation is an illustrative example using hourly volume and time context. Its output is an anomaly-review queue, not a fraud classifier.

## 3. Task 2 – Fraud Classification

### 3.1 Preprocessing and feature engineering

The modeling dataset excludes `user_id` because it is a unique identifier and has no defensible generalizable meaning.

The raw timestamp is transformed into:

- Hour of day
- Day of week
- Weekend indicator
- Sine and cosine encodings of hour of day

Registration duration is log-transformed to reduce skewness.

The preprocessing pipeline includes:

- Median imputation for numeric variables
- Most-frequent-category imputation for categorical variables
- One-hot encoding for categorical features
- Safe handling of previously unseen categorical values
- Numeric standardization for Logistic Regression

The data is divided into stratified 80% training and 20% test partitions using random state 42. Stratification preserves the target prevalence in both partitions.

The test set is held out from model fitting and preprocessing. Logistic Regression uses class weights as an interpretable baseline, while Random Forest uses balanced subsampling to account for class imbalance. No synthetic oversampling is applied.

### 3.2 Model results

The following results were obtained at the default classification threshold of 0.5:

| Model | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.366 | 0.886 | 0.518 | 0.968 | 0.774 |
| Random Forest | 0.727 | 0.662 | 0.693 | 0.958 | 0.758 |

The selected baseline model is **Logistic Regression**, based on its higher average precision (0.774 compared with 0.758 for Random Forest).

However, the models offer different operating characteristics:

- Logistic Regression detects a larger proportion of labelled fraud cases, with recall of 0.886, but produces more false-positive alerts.
- Random Forest has higher precision of 0.727, meaning a larger proportion of its alerts are correctly labelled fraud, but its recall is lower at 0.662.

The Logistic Regression confusion matrix is:

`[[8731, 769], [57, 443]]`

Rows represent actual classes (legitimate and fraud), and columns represent predicted classes (legitimate and fraud).

In this test set, Logistic Regression identifies 443 of the 500 fraud-labelled registrations, while generating 769 false-positive alerts.

Precision describes the proportion of predicted fraud alerts that are actually labelled fraud. Recall describes the proportion of all labelled fraud cases detected by the model.

The operating threshold should be selected according to the business cost of false positives, missed fraud and available investigation capacity rather than assuming that 0.5 is optimal.

### 3.3 Evaluation limitations

The random stratified split estimates performance when training and test examples are drawn from a similar population.

However, the dataset contains a pronounced, time-localized registration event. A random split can distribute observations from the same event across both training and test sets. This may overstate performance when the model is applied to a genuinely future period.

A chronological holdout would provide a more realistic assessment of future performance. Ideally, the model should also be validated on later months or a separate market or campaign.

The single supplied month is insufficient to establish robust seasonality, long-term stability or generalization across markets.

## 4. Further Steps

1. Obtain additional months of data and independently labelled outcomes to test temporal and market generalization.
2. Introduce chronological validation and report performance by week, device, domain and other relevant cohorts.
3. Tune the alert threshold against the operational cost of false positives, missed fraud and investigation capacity.
4. Review feature importance using permutation importance or SHAP, while recognizing that feature association does not establish causation. Sensitive attributes and potential proxies should undergo governance review.
5. Monitor feature drift, alert volumes, model calibration and precision/recall once labels mature.
6. Investigate the 11 January event using network, IP, session, device-fingerprint and referral information where available. These fields are not present in the supplied dataset and are not assumed in this analysis.
7. Introduce human review, audit trails, privacy controls and a retraining process before considering automated blocking decisions.

## 5. Reproducibility

Run the scripts in the order listed in `README.md`.

The random seed, train-test split proportion, preprocessing pipeline and model parameters are recorded in `outputs/classification_metrics.json`.

Generated tables and figures are stored under `outputs/`. The anomaly detection script also generates `outputs/anomaly_summary.json`, containing the selected candidate window, registration-volume statistics, Z-score results and Isolation Forest flags.

The analysis can be reproduced by running the project scripts in the documented sequence with the dependencies listed in `requirements.txt`.
