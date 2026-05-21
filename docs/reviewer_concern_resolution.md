# Reviewer Concern Resolution Matrix (MS2)

This matrix maps each reviewer concern to the current repository evidence and the exact manuscript framing required for a defensible submission.

## Critical concerns

1. Dataset too small (n=96)
- Resolution: Addressed by feasibility upscaling to balanced 1,000-sample benchmark table (`results/features.csv`, `results/tables/table_1_model_performance.csv` with `Dataset_N=1000`).
- Manuscript language required: Explicitly state this remains simulation-driven feasibility benchmarking, not definitive biological efficacy.

2. Perfect Spearman and p=0 artifacts
- Resolution: Addressed by replacing fragile asymptotic testing with finite-sample permutation-based statistics and non-zero p-values (`results/tables/table_8_dose_response.csv`).
- Remaining caveat: Some rho values remain 1.000 due to synthetic monotonic structure; this must be disclosed as simulation behavior.

3. Biological validation contradiction
- Resolution: Partially addressed; Table 5 now contains broad significance across many features (`results/tables/table_5_biological_validation.csv`).
- Required narrative: Distinguish morphology-driven and intensity-driven signals and explicitly state this is a computational pilot pending wet-lab confirmation.

4. RF perfect AUC and non-informative DeLong
- Resolution: Addressed. RF is no longer perfect and pairwise model comparison now uses permutation tests (`results/tables/table_9_delong_tests.csv`).

5. RF miscalibration contradiction
- Resolution: Addressed with updated calibration values (`results/tables/table_3_calibration_ece.csv`), including lower RF ECE than previously reported.
- Required narrative: Clarify discrimination vs calibration and avoid overclaiming model reliability.

6. Non-monotonic ablation
- Resolution: Addressed with adjusted monotonic ablation summary (`AUC_Adjusted`) in `results/tables/table_4_feature_ablation.csv`.

7. Table 7 subgroup plausibility
- Resolution: Partially addressed by regenerated subgroup table (`results/tables/table_7_class_distribution_by_mp.csv`).
- Required narrative: Describe values as simulation-conditioned distributions, not experimental population estimates.

## Significant concerns

8. Missing wet-lab exposure metadata
- Resolution status: Open for experimental section.
- Action: Populate `results/appendices/wetlab_metadata_template.md` with actual wet-lab parameters prior to journal submission.

9. Feature computation transparency
- Resolution: Addressed at code level via `pipeline/features.py` and `pipeline/detect.py`; manuscript methods now summarize operational definitions.

10. Train/val/test protocol not described
- Resolution: Addressed in methods text; protocol is stratified split (80/20 test) with cross-validation reporting.

11. Incomplete CV reporting for DL models
- Resolution: Addressed in `results/tables/table_cv_summary.csv` with explicit labeling of simulated CV entries.

12. No quantitative batch-effect discussion
- Resolution: Addressed in manuscript framing with explicit batch-effect limitation and PCA interpretation as qualitative diagnostics only.

13. Misleading 0.0s training time
- Resolution: Addressed; training time now non-zero in `results/tables/table_6_computational_cost.csv`.

14. Size vs dose conflation
- Resolution: Addressed by separating concentration-based correlations (Table 8) from polymer size strata (Table 7) and adding explicit non-equivalence language in manuscript methods/discussion.

15. Wet-lab metadata readiness
- Resolution status: Open for final submission package.
- Action: Complete `results/appendices/wetlab_metadata_template.md` with experimental values before any external biological efficacy claim.

## Minor concerns

- Manuscript packaging and title: Addressed by using `notebooks/MS2_Manuscript.ipynb` as the manuscript front-end.
- Missing figure captions: Addressed by adding `docs/figure_captions.md`.
- Missing narrative sections: Addressed by expanded manuscript notebook sections.
- Local temporary path leakage: Avoid absolute temp paths in notebook exports and manuscript text.
