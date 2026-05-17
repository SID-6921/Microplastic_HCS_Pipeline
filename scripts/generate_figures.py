"""Generate publication-style figures from results/features.csv.
Produces 9 figures:
1. Class distribution
2. PCA (PC1 vs PC2)
3. Feature correlation heatmap
4. Random Forest feature importances (top 10)
5. ROC curves (LR vs RF, macro)
6. Confusion matrix (Random Forest)
7. Confusion matrix (Logistic Regression)
8. Calibration curve (Random Forest)
9. Boxplots for selected features by class
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve

SEED = 42
FEATURE_COLS = [
    "nuclear_fragmentation_index",
    "cell_shrinkage_ratio",
    "membrane_blebbing_score",
    "chromatin_condensation_proxy",
    "cell_swelling_index",
    "membrane_permeability_proxy",
    "mean_intensity",
    "total_intensity",
    "intensity_variance",
    "area_covered_ratio",
    "cell_count",
    "density_cells_per_10k_px",
    "cell_area_mean",
    "cell_area_std",
    "cell_area_median",
    "small_cell_fraction",
    "medium_cell_fraction",
    "large_cell_fraction",
]

OUT = Path("results")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(OUT / "features.csv")
    y = df["class_id"].values
    X = df[FEATURE_COLS].values
    X = np.nan_to_num(X, 0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_test = Xs[train_idx], Xs[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train LR and RF for plotting
    lr = LogisticRegression(max_iter=1000, random_state=SEED, solver='lbfgs')
    lr.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)

    lr_proba = lr.predict_proba(X_test)
    rf_proba = rf.predict_proba(X_test)
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    class_names = df["class_name"].astype(str).unique().tolist()
    n_classes = len(class_names)

    # 1 Class distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x=df['class_name'], order=class_names)
    plt.title('Class distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIG / 'figure_01_class_distribution.png', dpi=150)
    plt.close()

    # 2 PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=df['class_name'], palette='tab10', s=60)
    plt.title('PCA: PC1 vs PC2')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(FIG / 'figure_02_pca_pc1_pc2.png', dpi=150)
    plt.close()

    # 3 Correlation heatmap
    corr = pd.DataFrame(Xs, columns=FEATURE_COLS).corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='vlag', center=0, xticklabels=True, yticklabels=True)
    plt.title('Feature correlation (standardized)')
    plt.tight_layout()
    plt.savefig(FIG / 'figure_03_feature_correlation.png', dpi=150)
    plt.close()

    # 4 RF feature importance
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(6,4))
    sns.barplot(x=importances[idx], y=np.array(FEATURE_COLS)[idx], palette='mako')
    plt.title('Random Forest: Top 10 feature importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(FIG / 'figure_04_rf_feature_importance.png', dpi=150)
    plt.close()

    # 5 ROC curves (macro-average) — plot per-class ROC and macro
    y_bin = label_binarize(y_test, classes=range(n_classes))
    plt.figure(figsize=(6,5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:,i], rf_proba[:,i])
        plt.plot(fpr, tpr, lw=1, label=f'RF class {i}')
    # LR
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:,i], lr_proba[:,i])
        plt.plot(fpr, tpr, lw=1, linestyle='--', label=f'LR class {i}')
    plt.plot([0,1],[0,1], color='navy', lw=1, linestyle=':')
    plt.title('ROC curves (per-class)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(FIG / 'figure_05_roc_curves.png', dpi=150)
    plt.close()

    # 6 Confusion matrix RF
    cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix: Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(FIG / 'figure_06_confusion_rf.png', dpi=150)
    plt.close()

    # 7 Confusion matrix LR
    cm = confusion_matrix(y_test, lr_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix: Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(FIG / 'figure_07_confusion_lr.png', dpi=150)
    plt.close()

    # 8 Calibration curve RF
    prob_true, prob_pred = calibration_curve(label_binarize(y_test, classes=range(n_classes)).ravel(), rf_proba.ravel(), n_bins=10)
    plt.figure(figsize=(5,4))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.title('Calibration curve (RF, pooled)')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.tight_layout()
    plt.savefig(FIG / 'figure_08_calibration_rf.png', dpi=150)
    plt.close()

    # 9 Boxplots of selected features
    sel = ['nuclear_fragmentation_index', 'cell_area_mean', 'membrane_permeability_proxy']
    melt = df.melt(id_vars=['class_name'], value_vars=sel, var_name='feature', value_name='value')
    plt.figure(figsize=(8,5))
    sns.boxplot(x='feature', y='value', hue='class_name', data=melt)
    plt.title('Selected features by class')
    plt.tight_layout()
    plt.savefig(FIG / 'figure_09_feature_boxplots.png', dpi=150)
    plt.close()

    print('Generated 9 informative figures in results/figures')

if __name__ == '__main__':
    main()
