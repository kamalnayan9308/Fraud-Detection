from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'dataset_20260430.csv.xlsx'
OUTPUT = ROOT / 'outputs'
FIGURES = OUTPUT / 'figures'
TABLES = OUTPUT / 'tables'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'

for folder in (OUTPUT, FIGURES, TABLES, MODELS, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)


def load_data(path=DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f'Dataset not found: {path}. Place the supplied Excel file in data/.')
    df = pd.read_excel(path)
    required = {
        'user_id', 'registration_timestamp', 'email_domain', 'age', 'gender',
        'job_title', 'education_level', 'is_smoker', 'is_drinker', 'os',
        'device_type', 'registration_duration_ms', 'is_fraud'
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df['registration_timestamp'] = pd.to_datetime(df['registration_timestamp'], errors='coerce')
    if df['registration_timestamp'].isna().any():
        raise ValueError('Some registration_timestamp values could not be parsed.')
    if df['is_fraud'].isna().any():
        raise ValueError('Target is_fraud contains missing values.')
    df['is_fraud'] = df['is_fraud'].astype(int)
    return df
