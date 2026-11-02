
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from common import load_data, FIGURES, TABLES, OUTPUT


def main():
    # ---------------------------------------------------------
    # 1. Load and prepare data
    # ---------------------------------------------------------
    df = load_data()
    df = df.set_index('registration_timestamp').sort_index()

    # ---------------------------------------------------------
    # 2. Daily registration and fraud analysis
    # ---------------------------------------------------------
    daily = df.resample('D').agg(
        total_registrations=('user_id', 'size'),
        fraudulent_registrations=('is_fraud', 'sum')
    )

    daily['fraud_rate'] = (
        daily['fraudulent_registrations'] /
        daily['total_registrations'].replace(0, np.nan)
    )

    daily.reset_index().to_csv(
        TABLES / 'daily_registration_summary.csv', index=False
    )

    # Dynamically identify the day with the highest registration volume.
    peak_day_timestamp = daily['total_registrations'].idxmax()
    peak_day = daily.loc[peak_day_timestamp]

    # ---------------------------------------------------------
    # 3. Hourly registration analysis
    # ---------------------------------------------------------
    hourly = df.resample('h').agg(
        total_registrations=('user_id', 'size'),
        fraudulent_registrations=('is_fraud', 'sum')
    )

    hourly['fraud_rate'] = (
        hourly['fraudulent_registrations'] /
        hourly['total_registrations'].replace(0, np.nan)
    )

    hourly['fraud_rate'] = hourly['fraud_rate'].fillna(0)

    # ---------------------------------------------------------
    # 4. Dynamically identify the busiest 3-hour window
    # ---------------------------------------------------------
    window_size = 3

    rolling_volume = (
        hourly['total_registrations']
        .rolling(window=window_size, min_periods=window_size)
        .sum()
    )

    if rolling_volume.notna().sum() == 0:
        raise ValueError(
            "Insufficient hourly observations to identify "
            "a three-hour candidate window."
        )

    # idxmax returns the ending hour of the busiest window.
    candidate_end_hour = rolling_volume.idxmax()

    candidate_start = (
        candidate_end_hour -
        pd.Timedelta(hours=window_size - 1)
    )

    candidate_end = candidate_end_hour + pd.Timedelta(hours=1)

    candidate_mask = (
        (hourly.index >= candidate_start) &
        (hourly.index < candidate_end)
    )

    # ---------------------------------------------------------
    # 5. Statistical anomaly detection using Z-scores
    # ---------------------------------------------------------
    # Exclude the candidate window from the baseline to reduce
    # the influence of the event being investigated.
    baseline_mask = (
        (~candidate_mask) &
        (hourly['total_registrations'] > 0)
    )

    baseline_values = hourly.loc[
        baseline_mask, 'total_registrations'
    ]

    if len(baseline_values) < 2:
        raise ValueError(
            "Not enough non-event hourly observations "
            "to calculate the baseline."
        )

    baseline_mean = float(baseline_values.mean())
    baseline_std = float(baseline_values.std(ddof=1))

    if np.isfinite(baseline_std) and baseline_std > 0:
        hourly['volume_z_score'] = (
            hourly['total_registrations'] - baseline_mean
        ) / baseline_std
    else:
        hourly['volume_z_score'] = np.nan

    hourly['volume_z_anomaly'] = (
        hourly['volume_z_score'] > 3
    )

    hourly.reset_index().to_csv(
        TABLES / 'hourly_volume_anomaly_scores.csv',
        index=False
    )

    # ---------------------------------------------------------
    # 6. Unsupervised anomaly detection using Isolation Forest
    # ---------------------------------------------------------
    # Fraud labels are deliberately excluded from model features.
    hourly_features = hourly[['total_registrations']].copy()

    hourly_features['hour_of_day'] = hourly_features.index.hour
    hourly_features['day_of_week'] = hourly_features.index.dayofweek

    hourly_features['hour_sin'] = np.sin(
        2 * np.pi * hourly_features['hour_of_day'] / 24
    )

    hourly_features['hour_cos'] = np.cos(
        2 * np.pi * hourly_features['hour_of_day'] / 24
    )

    iso = IsolationForest(
        n_estimators=300,
        contamination=0.02,
        random_state=42
    )

    hourly['isolation_forest_flag'] = (
        iso.fit_predict(hourly_features) == -1
    )

    hourly.reset_index().to_csv(
        TABLES / 'hourly_unsupervised_flags.csv',
        index=False
    )

    # ---------------------------------------------------------
    # 7. Daily fraud-rate visualization
    # ---------------------------------------------------------
    d = daily.reset_index()

    plt.figure(figsize=(13, 5))

    plt.plot(
        d['registration_timestamp'],
        d['fraud_rate'] * 100,
        marker='o',
        linewidth=1.5
    )

    plt.ylabel('Fraud rate (%)')
    plt.xlabel('Registration date')
    plt.title('Daily fraud rate (retrospective labelled analysis)')

    plt.xticks(rotation=45)
    plt.grid(alpha=.25)
    plt.tight_layout()

    plt.savefig(
        FIGURES / 'daily_fraud_rate.png',
        dpi=160
    )

    plt.close()

    # ---------------------------------------------------------
    # 8. Hourly registration volume visualization
    # ---------------------------------------------------------
    h = hourly.reset_index()

    plt.figure(figsize=(14, 5))

    plt.plot(
        h['registration_timestamp'],
        h['total_registrations'],
        linewidth=1
    )

    plt.axvspan(
        candidate_start,
        candidate_end,
        color='red',
        alpha=.2,
        label='Dynamically identified busiest 3-hour window'
    )

    plt.ylabel('Registrations per hour')
    plt.xlabel('Timestamp')
    plt.title('Hourly registration volume and candidate event')

    plt.legend()
    plt.grid(alpha=.25)
    plt.tight_layout()

    plt.savefig(
        FIGURES / 'hourly_registration_volume.png',
        dpi=160
    )

    plt.close()

    # ---------------------------------------------------------
    # 9. Candidate event analysis
    # ---------------------------------------------------------
    event = hourly.loc[candidate_mask]

    event_n = int(event['total_registrations'].sum())
    event_f = int(event['fraudulent_registrations'].sum())

    event_fraud_rate = (
        float(event_f / event_n) if event_n > 0 else 0.0
    )

    # ---------------------------------------------------------
    # 10. Generate anomaly summary
    # ---------------------------------------------------------
    summary = {
        'date_range': [
            str(df.index.min()),
            str(df.index.max())
        ],

        'overall_users': int(len(df)),
        'overall_fraud': int(df['is_fraud'].sum()),
        'overall_fraud_rate': float(df['is_fraud'].mean()),

        'peak_day': str(peak_day_timestamp.date()),
        'peak_day_registrations': int(
            peak_day['total_registrations']
        ),
        'peak_day_fraud': int(
            peak_day['fraudulent_registrations']
        ),
        'peak_day_fraud_rate': float(
            peak_day['fraud_rate']
        ),

        'candidate_window_start': str(candidate_start),
        'candidate_window_end_exclusive': str(candidate_end),

        'candidate_window_registrations': event_n,
        'candidate_window_fraud': event_f,
        'candidate_window_fraud_rate': event_fraud_rate,

        'baseline_hourly_volume_mean_excluding_candidate':
            baseline_mean,

        'baseline_hourly_volume_std_excluding_candidate':
            baseline_std,

        'candidate_hourly_mean_volume': float(
            event['total_registrations'].mean()
        ),

        'candidate_hourly_z_scores': [
            float(x) for x in
            hourly.loc[candidate_mask, 'volume_z_score']
            if pd.notna(x)
        ],

        'z_score_anomaly_hours': int(
            hourly['volume_z_anomaly'].sum()
        ),

        'isolation_forest_anomaly_hours': int(
            hourly['isolation_forest_flag'].sum()
        ),

        'isolation_forest_contamination_assumption': 0.02,

        'candidate_selection_method':
            'Highest rolling three-hour registration volume',

        'candidate_window_is_confirmed_fraud':
            False
    }

    # ---------------------------------------------------------
    # 11. Save summary
    # ---------------------------------------------------------
    (OUTPUT / 'anomaly_summary.json').write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
