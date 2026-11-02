import json
import pandas as pd
import matplotlib.pyplot as plt
from common import load_data, OUTPUT, FIGURES, TABLES


def main():
    df = load_data()
    quality = pd.DataFrame({
        'column': df.columns,
        'dtype': [str(x) for x in df.dtypes],
        'missing_count': [int(x) for x in df.isna().sum()],
        'missing_pct': [round(float(x) * 100, 3) for x in df.isna().mean()],
        'unique_count': [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    quality.to_csv(TABLES / 'data_quality.csv', index=False)

    target = df['is_fraud'].value_counts().rename_axis('is_fraud').reset_index(name='users')
    target['share_pct'] = target['users'] / len(df) * 100
    target.to_csv(TABLES / 'target_distribution.csv', index=False)

    numeric = df[['age', 'registration_duration_ms']].describe(percentiles=[.01, .05, .5, .95, .99]).T
    numeric.to_csv(TABLES / 'numeric_summary.csv')

    categorical_cols = ['email_domain', 'gender', 'job_title', 'education_level', 'is_smoker', 'is_drinker', 'os', 'device_type']
    for col in categorical_cols:
        summary = (df.groupby(col, dropna=False)['is_fraud']
                   .agg(users='size', frauds='sum', fraud_rate='mean')
                   .reset_index())
        summary['fraud_rate_pct'] = summary['fraud_rate'] * 100
        summary.sort_values(['fraud_rate', 'users'], ascending=[False, False]).to_csv(
            TABLES / f'fraud_by_{col}.csv', index=False
        )

    fraud_counts = df['is_fraud'].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    plt.bar(['Legitimate', 'Fraud'], [fraud_counts.get(0, 0), fraud_counts.get(1, 0)])
    plt.title('Target class distribution')
    plt.ylabel('Number of users')
    plt.tight_layout()
    plt.savefig(FIGURES / 'target_distribution.png', dpi=160)
    plt.close()

    print(f'Rows: {len(df):,}; columns: {len(df.columns)}')
    print(f'Date range: {df.registration_timestamp.min()} to {df.registration_timestamp.max()}')
    print(f'Fraud rate: {df.is_fraud.mean():.2%}')
    print('Missing values:')
    print(quality.loc[quality.missing_count > 0, ['column', 'missing_count', 'missing_pct']].to_string(index=False))
    print(f'EDA tables saved in {TABLES}')

if __name__ == '__main__':
    main()
