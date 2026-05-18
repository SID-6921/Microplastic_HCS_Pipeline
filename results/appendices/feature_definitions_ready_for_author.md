Feature Definitions — Final Draft (Author Verification Required)

Below we present the finalized draft of computational feature definitions used in the analysis. These entries have been standardized for clarity and include explicit parameter fields for authors to confirm or correct. Please complete the "Author confirmation" and "Source file / pipeline" fields for each feature. Once confirmed we will incorporate the exact formulas into the manuscript appendix and add unit tests in `tests/test_features.py`.

Format for each feature:
- Computational formula (explicit)
- Parameters (explicit values)
- Units and handling rules (e.g., zero handling)
- Source file / pipeline (CellProfiler/ImageJ/Python path)
- Author confirmation (fill)

1. `nuclear_fragmentation_index`
- Computational formula: nuclear_fragmentation_index = total_nuclear_fragments / total_cell_count
- Parameters: fragment_min_area = 5  # pixels; fragment_connectivity = 8
- Units / handling: unitless ratio; report NaN if total_cell_count == 0
- Source file / pipeline: _____________________
- Author confirmation: _____________________

2. `membrane_blebbing_score`
- Computational formula: membrane_blebbing_score = num_bleb_positive_cells / total_cell_count
- How bleb_positive is determined: A cell is 'bleb_positive' if any connected protrusion region within the cell mask has area >= bleb_area_threshold
- Parameters: bleb_area_threshold = 10  # pixels; curvature_threshold = 0.8
- Units / handling: unitless fraction; mask holes ignored; NaN if total_cell_count == 0
- Source file / pipeline: _____________________
- Author confirmation: _____________________

3. `cell_shrinkage_ratio`
- Computational formula: cell_shrinkage_ratio = median(baseline_nuclear_area / observed_nuclear_area)
- Baseline: baseline_nuclear_area = median(nuclear_area in control wells)  # confirm whether per-batch or global
- Units / handling: unitless ratio; if observed_nuclear_area <= 0 then exclude those cells from median
- Source file / pipeline: _____________________
- Author confirmation: _____________________

4. `texture_entropy`
- Computational formula: texture_entropy = mean(GLCM_entropy(patch) for patch in ROI_patches)
- Parameters: glcm_distances = [1]; glcm_angles = [0, pi/4, pi/2, 3pi/4]; levels = 256; patch_size = 32x32 pixels
- Units / handling: entropy in nats (or bits if log2 used) — specify log base; NaN for empty ROI
- Source file / pipeline: _____________________
- Author confirmation: _____________________

5. `mean_intensity_channel_X`
- Computational formula: mean_intensity_channel_X = mean(pixel_values[channel_X][cell_mask])
- Preprocessing: background_subtraction = True (method = median background)  # confirm
- Units / handling: arbitrary fluorescence units (AFU); if mask empty → NaN
- Source file / pipeline: _____________________
- Author confirmation: _____________________

6. `radial_distribution_score`
- Computational formula: radial_distribution_score = mean_intensity_outer_ring / mean_intensity_inner_circle
- Ring definitions: inner_radius = 0.0–0.5 normalized radius; outer_ring = 0.8–1.0 normalized radius (percentile of mask radius)
- Units / handling: unitless ratio; add small epsilon (1e-8) to denominator to avoid division by zero
- Source file / pipeline: _____________________
- Author confirmation: _____________________

General instructions for authors:
- Provide the exact parameter values or indicate if parameters vary per batch/experiment.
- If any feature uses smoothing, morphological operations, or specific thresholding (Otsu/adaptive), provide those parameter values explicitly.
- Please attach the original pipeline files (CellProfiler .cpproj, ImageJ macro, or the Python module) for provenance.

Once authors confirm, we will:
- Freeze the appendix text and embed it into the manuscript.
- Add unit tests asserting the presence and numeric output of each feature function in `pipeline/features.py` (using `tests/test_features.py`).
- Re-run the reproducible analysis pipelines and update tables/figures if values change.

Action requested: fill the "Source file / pipeline" and "Author confirmation" fields above for each feature (or paste the `pipeline/features.py` content here). If you prefer, grant me access to the restored repo and I will auto-extract and verify formulas.
