"""Integrity checks for included CSVs (not an independent retraining)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results'
SEEDS = {101, 202, 303, 404, 505}
RATES = {0.0, 0.2, 0.4}

for file in ['edge_ablation_original_five_seeds.csv', 'edge_ablation_unique_five_seeds.csv']:
    df = pd.read_csv(DATA / file)
    assert len(df) == 15, f'{file}: expected 15 rows'
    assert set(df['seed']) == SEEDS, f'{file}: unexpected seeds'
    assert set(df['training_edge_drop_requested'].round(3)) == RATES
    assert (df.groupby(['seed', 'training_edge_drop_requested']).size() == 1).all()
    assert (df['split_seed'] == 4444).all()
    assert (df['epochs_available'] == 40).all()
    assert df['test_mcc'].notna().all()
    print(f'{file}: 15 complete seed × rate results')

comparison = pd.read_csv(DATA / 'european_validation_selected_all.csv')
assert len(comparison) == 15, 'expected 3 × 5 baseline evaluations'
assert set(comparison['configuration']) == {10, 40, 75}
assert comparison.groupby('configuration').size().eq(5).all()
assert comparison['test_mcc'].notna().all()
print('european_validation_selected_all.csv: 15 baseline rows')

for stem in ['original', 'unique']:
    runs = pd.read_csv(DATA / f'edge_ablation_{stem}_five_seeds.csv')
    summary = pd.read_csv(DATA / f'edge_ablation_{stem}_five_seeds_summary.csv')
    expected = runs.groupby('training_edge_drop_requested')['test_mcc'].agg(['mean', 'std'])
    for _, row in summary.iterrows():
        group = expected.loc[row['training_edge_drop_requested']]
        assert abs(row['test_mcc_mean'] - group['mean']) < 1e-9
        assert abs(row['test_mcc_std'] - group['std']) < 1e-9
    print(f'{stem} summary matches individual results')
print('All CSV checks passed.')
