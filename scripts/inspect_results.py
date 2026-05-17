import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

OUT = 'results'
F = OUT + '/features.csv'
df = pd.read_csv(F)
FEATURE_COLS = [
    'nuclear_fragmentation_index','cell_shrinkage_ratio','membrane_blebbing_score','chromatin_condensation_proxy','cell_swelling_index','membrane_permeability_proxy','mean_intensity','total_intensity','intensity_variance','area_covered_ratio','cell_count','density_cells_per_10k_px','cell_area_mean','cell_area_std','cell_area_median','small_cell_fraction','medium_cell_fraction','large_cell_fraction'
]
X = df[FEATURE_COLS].values
y = df['class_id'].values
X = np.nan_to_num(X,0)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
y_test = y[test_idx]
print('Test size:', len(test_idx))
unique, counts = np.unique(y_test, return_counts=True)
print('Test class counts:', dict(zip(unique, counts)))

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(Xs[train_idx], y[train_idx])
rf_pred = rf.predict(Xs[test_idx])
cm = confusion_matrix(y_test, rf_pred)
print('Confusion matrix:\n', cm)
print('Accuracy:', (rf_pred==y_test).mean())
