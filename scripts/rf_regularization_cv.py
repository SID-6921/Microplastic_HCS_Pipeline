"""Evaluate Random Forest robustness under different max_depth settings using stratified 5-fold CV.
Saves per-fold metrics and summary table to results/tables.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score

OUT = Path('results')
TABLES = OUT / 'tables'
TABLES.mkdir(parents=True, exist_ok=True)
FEATURE_COLS = [
    'nuclear_fragmentation_index','cell_shrinkage_ratio','membrane_blebbing_score','chromatin_condensation_proxy','cell_swelling_index','membrane_permeability_proxy','mean_intensity','total_intensity','intensity_variance','area_covered_ratio','cell_count','density_cells_per_10k_px','cell_area_mean','cell_area_std','cell_area_median','small_cell_fraction','medium_cell_fraction','large_cell_fraction'
]

def evaluate(max_depth):
    df = pd.read_csv(OUT / 'features.csv')
    X = df[FEATURE_COLS].values
    y = df['class_id'].values
    X = np.nan_to_num(X, 0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    importances = []
    fold=0
    for train_idx, test_idx in skf.split(Xs, y):
        fold+=1
        X_train, X_test = Xs[train_idx], Xs[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        rf = RandomForestClassifier(n_estimators=200, max_depth=max_depth, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        try:
            y_bin = label_binarize(y_test, classes=np.unique(y))
            proba = rf.predict_proba(X_test)
            auc = roc_auc_score(y_bin, proba, multi_class='ovr', average='macro')
        except Exception:
            auc = np.nan
        rows.append({'max_depth': str(max_depth), 'fold': fold, 'accuracy': acc, 'auc_macro': auc})
        importances.append(rf.feature_importances_)
    df_rows = pd.DataFrame(rows)
    imp_mean = np.mean(np.vstack(importances), axis=0)
    imp_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': imp_mean}).sort_values('importance', ascending=False)
    return df_rows, imp_df


def main():
    depths = [None, 6, 4]
    all_rows = []
    imps = {}
    for d in depths:
        rows, imp_df = evaluate(d)
        all_rows.append(rows)
        imps[str(d)] = imp_df
        rows.to_csv(TABLES / f'rf_reg_per_fold_maxdepth_{str(d)}.csv', index=False)
        imp_df.to_csv(TABLES / f'rf_reg_importances_maxdepth_{str(d)}.csv', index=False)

    per_fold = pd.concat(all_rows, ignore_index=True)
    per_fold.to_csv(TABLES / 'rf_reg_per_fold_all.csv', index=False)

    summary = per_fold.groupby('max_depth').agg({'accuracy':['mean','std'],'auc_macro':['mean','std']})
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.to_csv(TABLES / 'rf_reg_summary.csv')

    print('Regularization CV complete; tables in results/tables')

if __name__ == '__main__':
    main()
