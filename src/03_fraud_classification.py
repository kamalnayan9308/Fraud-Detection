import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score,
    RocCurveDisplay, PrecisionRecallDisplay)
from common import load_data, FIGURES, TABLES, MODELS, OUTPUT


def make_features(df):
    x = df.drop(columns=['is_fraud', 'user_id', 'registration_timestamp']).copy()
    ts = pd.to_datetime(df['registration_timestamp'])
    x['registration_hour'] = ts.dt.hour
    x['registration_day_of_week'] = ts.dt.dayofweek
    x['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)
    # Cyclical encoding respects the circular nature of time-of-day.
    x['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
    x['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
    x['log_registration_duration'] = np.log1p(x['registration_duration_ms'].clip(lower=0))
    x = x.drop(columns=['registration_duration_ms'])
    return x


def build_pipeline(model, numeric, categorical, scale_numeric=False):
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        num_steps.append(('scaler', StandardScaler()))
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    prep = ColumnTransformer([
        ('num', Pipeline(num_steps), numeric),
        ('cat', cat_pipe, categorical)
    ])
    return Pipeline([('preprocess', prep), ('model', model)])


def main():
    df = load_data()
    X = make_features(df)
    y = df['is_fraud']
    numeric = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        'logistic_regression': build_pipeline(
            LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
            numeric, categorical, scale_numeric=True
        ),
        'random_forest': build_pipeline(
            RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                class_weight='balanced_subsample', n_jobs=-1, random_state=42),
            numeric, categorical, scale_numeric=False
        )
    }

    results = []
    fitted = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        row = {
            'model': name,
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, prob),
            'pr_auc_average_precision': average_precision_score(y_test, prob),
            'balanced_accuracy': balanced_accuracy_score(y_test, pred)
        }
        results.append(row)
        fitted[name] = pipe
        joblib.dump(pipe, MODELS / f'{name}.joblib')
        report = classification_report(y_test, pred, target_names=['legitimate', 'fraud'], zero_division=0, output_dict=True)
        pd.DataFrame(report).T.to_csv(TABLES / f'{name}_classification_report.csv')
        cm = confusion_matrix(y_test, pred)
        pd.DataFrame(cm, index=['actual_legitimate', 'actual_fraud'],
                     columns=['predicted_legitimate', 'predicted_fraud']).to_csv(TABLES / f'{name}_confusion_matrix.csv')

    result_df = pd.DataFrame(results).sort_values('pr_auc_average_precision', ascending=False)
    result_df.to_csv(TABLES / 'model_comparison.csv', index=False)
    best_name = result_df.iloc[0]['model']
    best_model = fitted[best_name]
    best_prob = best_model.predict_proba(X_test)[:, 1]
    best_pred = best_model.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, best_prob, ax=ax)
    ax.set_title(f'ROC curve – {best_name}'); fig.tight_layout()
    fig.savefig(FIGURES / 'best_model_roc_curve.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_test, best_prob, ax=ax)
    ax.set_title(f'Precision–Recall curve – {best_name}'); fig.tight_layout()
    fig.savefig(FIGURES / 'best_model_precision_recall_curve.png', dpi=160); plt.close(fig)

    # Save test metrics and split details for transparent reporting.
    metrics = {
        'n_rows': int(len(df)), 'n_features_before_encoding': int(X.shape[1]),
        'fraud_rate': float(y.mean()), 'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)), 'random_state': 42,
        'test_size': 0.20, 'stratified': True,
        'features_used': list(X.columns), 'features_excluded': ['user_id', 'registration_timestamp (raw)'],
        'best_model_by_pr_auc': best_name,
        'best_model_test_metrics': result_df.iloc[0].to_dict(),
        'best_model_confusion_matrix': confusion_matrix(y_test, best_pred).tolist()
    }
    (OUTPUT / 'classification_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(result_df.to_string(index=False))
    print(f'Best model by average precision: {best_name}')

if __name__ == '__main__':
    main()
