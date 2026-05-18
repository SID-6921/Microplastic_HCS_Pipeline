# Figure Captions

## Main figures

- Figure 1 (`fig_01_pipeline_workflow.png`): End-to-end computational workflow from synthetic image generation through preprocessing, feature extraction, model training, statistical validation, and artifact export.
- Figure 2 (`fig_02_cell_overlays.png`): Representative synthetic DAPI/PI channel patterns across the four class labels used in the feasibility pipeline.
- Figure 3 (`fig_03_roc_feature_models.png`): One-vs-rest ROC curves for logistic regression and random forest feature-based classifiers.
- Figure 4 (`fig_04_rf_feature_importance.png`): Random forest impurity-based feature importance across the 18 descriptors.
- Figure 5 (`fig_05_roc_dl_models.png`): One-vs-rest ROC curves for simulated CNN and ResNet model outputs.
- Figure 6 (`fig_06_calibration_curves.png`): Reliability curves with model-level calibration behavior and expected calibration error context.
- Figure 7 (`fig_07_pca_class_clusters.png`): PCA projection after normalization showing class-level separation in feature space.
- Figure 8 (`fig_08_feature_ablation.png`): AUC trend as top-ranked features are removed, including adjusted monotonic interpretation.
- Figure 9 (`fig_09_morphological_fingerprint.png`): Class-by-feature heatmap of normalized descriptor means.

## Supplementary figures

- Supplementary S1 (`supp_s1_apoptosis_features.png`): Distribution plots for apoptosis-oriented features.
- Supplementary S2 (`supp_s2_necrosis_features.png`): Distribution plots for necrosis/permeability and intensity features.
- Supplementary S3 (`supp_s3_morphology_features.png`): Distribution plots for morphology-heavy descriptors.
- Supplementary S4 (`supp_s4_pca_before_norm.png`): PCA before normalization, used as a qualitative batch-structure diagnostic.
- Supplementary S5 (`supp_s5_cm_logistic_regression.png`): Confusion matrix for logistic regression.
- Supplementary S6 (`supp_s6_cm_random_forest.png`): Confusion matrix for random forest.
- Supplementary S7 (`supp_s7_cm_resnet18_pretrained.png`): Confusion matrix for pretrained ResNet-18 (simulated output).
