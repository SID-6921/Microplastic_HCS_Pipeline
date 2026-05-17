"""Perform stratified 5-fold CV for Logistic Regression and Random Forest,
save per-fold metrics and averaged feature importances.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

OUT = Path('results')
OUT.mkdir(exist_ok=True)
TABLES = OUT / 'tables'
TABLES.mkdir(exist_ok=True)
FEATURE_COLS = [
    'nuclear_fragmentation_index','cell_shrinkage_ratio','membrane_blebbing_score','chromatin_condensation_proxy','cell_swelling_index','membrane_permeability_proxy','mean_intensity','total_intensity','intensity_variance','area_covered_ratio','cell_count','density_cells_per_10k_px','cell_area_mean','cell_area_std','cell_area_median','small_cell_fraction','medium_cell_fraction','large_cell_fraction'
]

def main():
    df = pd.read_csv(OUT / 'features.csv')
    X = df[FEATURE_COLS].values
    y = df['class_id'].values
    X = np.nan_to_num(X, 0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    rf_importances = []
    fold = 0
    for train_idx, test_idx in skf.split(Xs, y):
        fold += 1
        X_train, X_test = Xs[train_idx], Xs[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        for name, model in [('LR', lr), ('RF', rf)]:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            try:
                y_bin = label_binarize(y_test, classes=np.unique(y))
                proba = model.predict_proba(X_test)
                auc = roc_auc_score(y_bin, proba, multi_class='ovr', average='macro')
            except Exception:
                auc = np.nan

            rows.append({'fold': fold, 'model': name, 'accuracy': acc, 'auc_macro': auc})

        rf_importances.append(rf.feature_importances_)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(TABLES / 'cv_per_fold_metrics.csv', index=False)

    # Aggregate
    summary = metrics_df.groupby('model').agg({'accuracy': ['mean','std'], 'auc_macro': ['mean','std']})
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.to_csv(TABLES / 'cv_summary_metrics.csv')

    # Average feature importances
    importances_mean = np.mean(np.vstack(rf_importances), axis=0)
    feat_imp = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importances_mean})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    feat_imp.to_csv(TABLES / 'rf_feature_importances_cv.csv', index=False)

    print('CV complete. Tables written to results/tables')

if __name__ == '__main__':
    main()
