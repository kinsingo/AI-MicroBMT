import pandas as pd
import glob
import os

cols_to_drop = [
    'epoch_sample_latency_average_min', 'epoch_sample_latency_average_max',
    'epoch_samples_per_second_min', 'epoch_samples_per_second_max',
    'cpu_type', 'submitter',
]
base = r'C:\Users\Jonghyun Shin\OneDrive\바탕 화면\AI-MicroBMT'

csv_files = glob.glob(os.path.join(base, '**', '*.csv'), recursive=True)

for f in csv_files:
    df = pd.read_csv(f)
    existing = [c for c in cols_to_drop if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        df.to_csv(f, index=False)
        print(f'Dropped {existing} from: {os.path.basename(f)}')

print('Done.')
