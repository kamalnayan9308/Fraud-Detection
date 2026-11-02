import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from common import load_data, FIGURES, TABLES, OUTPUT


def main():
    df = load_data()
    df = df.set_index('registration_timestamp').sort_index()

    daily = df.resample('D').agg(
        total_registrations=('user_id', 'size'),
        fraudulent_registrations=('is_fraud', 'sum')
    )
    daily['fraud_rate'] = daily['fraudulent_registrations'] / daily['total_registrations']
    daily.reset_index().to_csv(TABLES / 'daily_registration_summary.csv', index=False)

    hourly = df.resample('h').agg(
        total_registrations=('user_id', 'size'),
        fraudulent_registrations=('is_fraud', 'sum')
    )
    hourly['fraud_rate'] = hourly['fraudulent_registrations'] / hourly['total_registrations'].replace(0, np.nan)
    hourly = hourly.fillna({'fraud_rate': 0})
    hourly.reset_index().to_csv(TABLES / 'hourly_registration_summary.csv', index=False)

    # The label is used here only to investigate the historical event.
    baseline = hourly.loc[hourly['total_registrations'] > 0, 'total_registrations']
    # Exclude the visually identified three-hour candidate from the baseline estimate.
    candidate_mask = ((hourly.index >= pd.Timestamp('2025-01-11 01:00:00')) &
                      (hourly.index < pd.Timestamp('2025-01-11 04:00:00')))
    baseline_values = hourly.loc[(~candidate_mask) & (hourly.total_registrations > 0), 'total_registrations']
    baseline_mean = float(baseline_values.mean())
    baseline_std = float(baseline_values.std(ddof=1))
    hourly['volume_z_score'] = (hourly.total_registrations - baseline_mean) / baseline_std
    hourly['volume_z_anomaly'] = hourly['volume_z_score'] > 3
    hourly.reset_index().to_csv(TABLES / 'hourly_volume_anomaly_scores.csv', index=False)

    # Unsupervised illustration: hourly features only; fraud labels are not passed to Isolation Forest.
    hourly_features = hourly[['total_registrations']].copy()
    hourly_features['hour_of_day'] = hourly_features.index.hour
    hourly_features['day_of_week'] = hourly_features.index.dayofweek
    hourly_features['hour_sin'] = np.sin(2 * np.pi * hourly_features['hour_of_day'] / 24)
    hourly_features['hour_cos'] = np.cos(2 * np.pi * hourly_features['hour_of_day'] / 24)
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42)
    hourly['isolation_forest_flag'] = iso.fit_predict(hourly_features) == -1
    hourly.reset_index().to_csv(TABLES / 'hourly_unsupervised_flags.csv', index=False)

    # Charts
    d = daily.reset_index()
    plt.figure(figsize=(13, 5))
    plt.plot(d['registration_timestamp'], d['fraud_rate'] * 100, marker='o', linewidth=1.5)
    plt.ylabel('Fraud rate (%)'); plt.xlabel('Registration date')
    plt.title('Daily fraud rate (retrospective labelled analysis)')
    plt.xticks(rotation=45); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(FIGURES / 'daily_fraud_rate.png', dpi=160); plt.close()

    h = hourly.reset_index()
    plt.figure(figsize=(14, 5))
    plt.plot(h['registration_timestamp'], h['total_registrations'], linewidth=1)
    plt.axvspan(pd.Timestamp('2025-01-11 01:00'), pd.Timestamp('2025-01-11 04:00'),
                color='red', alpha=.2, label='Candidate event: 01:00–03:59')
    plt.ylabel('Registrations per hour'); plt.xlabel('Timestamp')
    plt.title('Hourly registration volume and candidate event')
    plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(FIGURES / 'hourly_registration_volume.png', dpi=160); plt.close()

    event = hourly.loc[candidate_mask]
    event_n = int(event.total_registrations.sum())
    event_f = int(event.fraudulent_registrations.sum())
    day = daily.loc[pd.Timestamp('2025-01-11')]
    summary = {
        'date_range': [str(df.index.min()), str(df.index.max())],
        'overall_users': int(len(df)),
        'overall_fraud': int(df.is_fraud.sum()),
        'overall_fraud_rate': float(df.is_fraud.mean()),
        'peak_day': '2025-01-11',
        'peak_day_registrations': int(day.total_registrations),
        'peak_day_fraud': int(day.fraudulent_registrations),
        'peak_day_fraud_rate': float(day.fraud_rate),
        'candidate_window_start': '2025-01-11 01:00:00',
        'candidate_window_end_exclusive': '2025-01-11 04:00:00',
        'candidate_window_registrations': event_n,
        'candidate_window_fraud': event_f,
        'candidate_window_fraud_rate': event_f / event_n,
        'baseline_hourly_volume_mean_excluding_candidate': baseline_mean,
        'baseline_hourly_volume_std_excluding_candidate': baseline_std,
        'candidate_hourly_mean_volume': float(event.total_registrations.mean()),
        'candidate_hourly_z_scores': [float(x) for x in hourly.loc[candidate_mask, 'volume_z_score']],
        'isolation_forest_anomaly_hours': int(hourly.isolation_forest_flag.sum()),
        'isolation_forest_contamination_assumption': 0.02
    }
    (OUTPUT / 'anomaly_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
