# Kantar Data Science Take-Home: Fraud Detection

## Executive Summary

This analysis investigates registration activity to identify concentrated anomalous events and builds binary classification models to predict the supplied `is_fraud` target.

The dataset contains 50,000 registrations, with an overall fraud prevalence of 5.00%. The analysis is divided into two tasks:

- **Anomaly Detection:** Investigate registration patterns to identify an unusual period and evaluate its historical fraud concentration.
- **Fraud Classification:** Train and compare supervised models to predict fraudulent registrations.

The objective is to establish transparent and explainable baseline methods, understand their performance and limitations, and identify potential improvements. An anomaly flag or model prediction should not be treated as proof of fraud.

## Approach at a Glance

I followed an investigative approach before moving to predictive modelling.

1. Examined data quality, missing values, feature distributions and target imbalance.
2. Aggregated registration activity by day and hour to identify unusual patterns.
3. Used a rolling three-hour registration-volume window to identify a candidate event without relying on fraud labels.
4. Used historical fraud labels retrospectively to investigate the selected event.
5. Built Logistic Regression and Random Forest classifiers using a preprocessing pipeline.
6. Compared model performance using metrics suitable for imbalanced classification and considered the business implications of false positives and false negatives.

## 1. Data and Quality Checks

The supplied workbook contains 13 columns and 50,000 rows. `user_id` is an identifier and is excluded from model predictors. `registration_timestamp` is parsed as a datetime and transformed into time-derived features.

`education_level` contains missing values. The modeling pipeline imputes missing categorical values using the most frequent category learned from the training data. Numeric missing values are handled using median imputation.

All preprocessing steps are fitted within each model pipeline to avoid learning imputation or encoding rules from the test set.

The target is imbalanced:

- Legitimate registrations: 95.0%
- Fraudulent registrations: 5.0%

Consequently, accuracy alone is not an adequate evaluation measure. Precision, recall, F1-score, ROC-AUC and average precision are reported.

## 2. Task 1 – Anomaly Detection

### 2.1 Identifying the candidate event

I first aggregated registrations at daily and hourly levels to understand changes in registration volume and historical fraud concentration.

Daily aggregation identified **11 January 2025** as the day with the highest registration volume:

- Total registrations: 2,128
- Fraud-labelled registrations: 574
- Retrospective fraud rate: 26.97%

To identify the most concentrated registration spike without using fraud labels, I calculated a rolling three-hour registration total across the hourly time series. The three-hour window with the highest combined registration volume was selected as the candidate event.

The selected window was:

**11 January 2025, 01:00–03:59 (end-exclusive timestamp: 04:00).**

The retrospective results for this window were:

- Total registrations: 728
- Fraud-labelled registrations: 514
- Retrospective fraud rate: 70.60%

The candidate window was selected using registration volume only. Fraud labels were examined afterwards to assess whether the unusual activity coincided with a high concentration of known fraudulent registrations.

This distinction is important: the detection method identifies unusual activity, while the historical labels help investigate the nature of that activity.

### 2.2 Registration-volume anomaly scoring

The mean hourly registration volume outside the selected candidate window was 68.72, with a standard deviation of 8.29.

The candidate window had an average hourly volume of 242.67 registrations, approximately 3.53 times the baseline mean.

The hourly Z-scores within the candidate window were:

| Hour | Registration-volume Z-score |
|---|---:|
| 01:00 | 21.37 |
| 02:00 | 21.86 |
| 03:00 | 19.69 |

All three values exceed the illustrative threshold of 3 standard deviations, indicating unusually high registration volume relative to the calculated baseline.

The complete hourly analysis flagged four hours using the Z-score threshold. The additional flagged hour falls outside the selected three-hour candidate window.

This is a simple statistical baseline rather than a production-ready detector. It uses a global hourly mean and standard deviation and does not fully account for time-of-day patterns, seasonality, overdispersion or repeated testing.

### 2.3 Isolation Forest

I also implemented Isolation Forest as an unsupervised anomaly detection method.

The model uses hourly registration volume and time-context features:

- Total registrations per hour
- Hour of day
- Day of week
- Sine and cosine transformations of hour of day

The `is_fraud` target is not passed to the algorithm.

Using 300 trees and a contamination assumption of 2%, Isolation Forest flagged 15 hourly observations as anomalous.

The contamination parameter is a modelling assumption that influences the proportion of observations flagged. It is not an estimate of actual fraud prevalence.

The Z-score and Isolation Forest results are not expected to match exactly. Z-score identifies unusually high registration volumes relative to a global baseline, whereas Isolation Forest identifies observations that are unusual within the supplied feature space.

Both methods generate observations for further investigation; neither confirms fraud independently.

### 2.4 Detecting anomalies without fraud labels

In a real-world scenario where fraud labels are unavailable, the system cannot directly calculate fraud rates or determine whether a registration is fraudulent. It can instead monitor observable activity and identify deviations from expected behaviour.

Some relevant approaches include:

- **Rolling statistical baselines:** Monitor registration volumes using historical windows and time-aware thresholds.
- **Robust statistical methods:** Use Median Absolute Deviation (MAD) or control charts to reduce sensitivity to extreme observations.
- **Unsupervised learning:** Apply Isolation Forest, Local Outlier Factor or One-Class SVM to identify unusual observations.
- **Distribution monitoring:** Use Population Stability Index (PSI), Kolmogorov–Smirnov statistics or Jensen–Shannon divergence to identify changes in feature distributions.
- **Change-point detection:** Identify significant changes in registration behaviour over time.

A practical system could combine these methods with human investigation and feedback from subsequently confirmed outcomes.

The current implementation demonstrates this approach using registration volume and time context. Its output should be treated as an anomaly-review queue rather than a confirmed fraud classification.

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
- Handling of previously unseen categorical values
- Numeric standardization for Logistic Regression

The dataset is divided into stratified 80% training and 20% test partitions using random state 42. Stratification preserves the target prevalence in both partitions.

The test set is held out from model fitting and preprocessing. Logistic Regression uses class weights as an interpretable baseline, while Random Forest uses balanced subsampling to account for class imbalance. No synthetic oversampling is applied.

### 3.2 Model training and results

I trained two supervised classification models: Logistic Regression and Random Forest.

The following results were obtained at the default classification threshold of 0.5:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.917 | 0.366 | 0.886 | 0.518 | 0.968 | 0.774 |
| Random Forest | 0.965 | 0.727 | 0.662 | 0.693 | 0.958 | 0.758 |

Accuracy is included for completeness, but it is not used as the primary selection metric because the dataset is imbalanced. A model predicting every registration as legitimate would achieve 95% accuracy while failing to identify any fraud.

Average precision provides a more informative summary of precision-recall performance for this imbalanced classification problem.

### 3.3 Model selection and business interpretation

Based on average precision, **Logistic Regression** is selected as the baseline model, achieving 0.774 compared with 0.758 for Random Forest.

However, the models offer different operating characteristics:

- **Logistic Regression:** Higher recall (0.886), identifying a larger proportion of fraud-labelled registrations, but lower precision (0.366), resulting in more false-positive alerts.
- **Random Forest:** Higher precision (0.727), meaning a larger proportion of its alerts are correctly labelled fraud, but lower recall (0.662), meaning it misses more fraud-labelled registrations.

The Logistic Regression confusion matrix is:

`[[8731, 769], [57, 443]]`

Rows represent actual classes (legitimate and fraud), and columns represent predicted classes (legitimate and fraud).

At the default threshold, Logistic Regression identifies 443 of the 500 fraud-labelled registrations, while generating 769 false-positive alerts.

The appropriate operating threshold depends on the business cost of false positives, missed fraud and available investigation capacity. A threshold of 0.5 should not automatically be considered optimal.

### 3.4 Alternative classification approaches

The selected models provide interpretable baselines. Other approaches could be considered if additional data, time and validation resources were available.

- **Gradient Boosting:** XGBoost and LightGBM can capture complex nonlinear relationships and feature interactions.
- **Imbalance handling:** Cost-sensitive learning, threshold optimisation and techniques such as SMOTE could be evaluated depending on the data and validation design.
- **Model interpretation:** Permutation importance and SHAP can help investigate which features influence model predictions.

These methods have not been implemented in this submission. Their suitability would need to be established through comparative evaluation rather than assuming that more complex models will perform better.

### 3.5 Evaluation limitations

The random stratified split estimates performance when training and test examples are drawn from a similar population.

However, the dataset contains a pronounced, time-localized registration event. A random split can distribute observations from the same event across both training and test sets. This may overstate performance when the model is applied to a genuinely future period.

A chronological holdout would provide a more realistic assessment of future performance. Ideally, the model should also be validated on later months or a separate market or campaign.

The single supplied month is insufficient to establish robust seasonality, long-term stability or generalization across markets.

## 4. Further Steps

1. Obtain additional months of data and independently labelled outcomes to evaluate temporal and market generalization.
2. Introduce chronological validation and report performance across weeks, devices, domains and other relevant cohorts.
3. Tune the classification threshold against the operational cost of false positives, missed fraud and investigation capacity.
4. Review feature importance using permutation importance or SHAP, while recognising that association does not establish causation. Sensitive attributes and potential proxies should undergo governance review.
5. Monitor feature drift, alert volumes, model calibration and precision/recall once labels mature.
6. Investigate the 11 January event using network, IP, session, device-fingerprint and referral information where available. These fields are not present in the supplied dataset and are not assumed in this analysis.
7. Introduce human review, audit trails, privacy controls and a retraining process before considering automated blocking decisions.

## 5. Reproducibility

Run the scripts in the order listed in `README.md`.

The random seed, train-test split proportion, preprocessing pipeline and model parameters are recorded in `outputs/classification_metrics.json`.

Generated tables and figures are stored under `outputs/`. The anomaly detection script also generates `outputs/anomaly_summary.json`, containing the selected candidate window, registration-volume statistics, Z-score results and Isolation Forest flags.

The analysis can be reproduced by running the project scripts in the documented sequence with the dependencies listed in `requirements.txt`.
