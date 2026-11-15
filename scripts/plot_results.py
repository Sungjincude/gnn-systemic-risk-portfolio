"""Recreate portfolio figures from recorded CSVs, without model retraining."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

original = pd.read_csv(RESULTS / 'edge_ablation_original_five_seeds.csv')
unique = pd.read_csv(RESULTS / 'edge_ablation_unique_five_seeds.csv')
rates = [0.0, 0.2, 0.4]
a = original.groupby('training_edge_drop_requested')['test_mcc'].agg(['mean','std']).reindex(rates)
b = unique.groupby('training_edge_drop_requested')['test_mcc'].agg(['mean','std']).reindex(rates)
x = np.arange(3)
fig, ax = plt.subplots(figsize=(9,5.3))
ax.bar(x-0.18, a['mean'], 0.36, yerr=a['std'], capsize=4, label='Original (with replacement)')
ax.bar(x+0.18, b['mean'], 0.36, yerr=b['std'], capsize=4, label='Exact unique-edge removal')
ax.set_xticks(x, ['0%', '20%', '40%'])
ax.set_xlabel('Requested training-edge removal')
ax.set_ylabel('Test MCC (mean ± sample SD; five seeds)')
ax.set_title('GraphSAGE edge-ablation pilot — European network, config 75')
ax.legend()
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'edge_ablation_comparison.png', dpi=180)
plt.close(fig)

baseline = pd.read_csv(RESULTS / 'european_validation_selected_all.csv')
pivot = baseline.pivot(index='model', columns='configuration', values='test_mcc')
order = ['GraphSAGE','FNN','Random Forest','GAT','GCN']
pivot = pivot.reindex([m for m in order if m in pivot.index])
ax = pivot.plot(kind='bar', rot=0, figsize=(9.5,5.0), ylabel='Test MCC')
ax.set_xlabel('Model')
ax.set_title('European network: validation-selected classifiers')
ax.legend(title='Configuration')
ax.spines[['top','right']].set_visible(False)
ax.figure.tight_layout()
ax.figure.savefig(OUT / 'european_model_comparison.png', dpi=180)
plt.close(ax.figure)
print('Wrote figures to', OUT)
