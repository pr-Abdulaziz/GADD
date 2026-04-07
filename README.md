# Arabic Deepfake Detection Research Pipeline

This repository contains the notebooks and helper scripts used for an Arabic deepfake-text detection study across two platforms:

- Twitter
- YouTube

The study compares three preprocessing views:

- `original`
- `manual`
- `deepfake_aware`

It evaluates three model families:

- Classical ML
- Sequence neural models
- Fine-tuned Arabic transformers

The project then analyzes:

- In-domain performance on each platform
- Preprocessing effects
- Misclassification behavior by deception type and metadata
- Cross-platform transfer between Twitter and YouTube
- Final hypothesis-level conclusions

Across the helper modules and analysis scripts, the binary class convention is:

- `1 = Real`
- `0 = Fake`

## Research Questions

The local analysis bundle is organized around three main hypotheses:

- `H1`: Fine-tuned contextual Arabic transformers achieve the strongest in-domain Macro-F1 on both platforms.
- `H2`: Preprocessing changes performance, with special attention to `manual` and `deepfake_aware` variants relative to `original`.
- `H3`: Cross-platform transfer is harder than in-domain evaluation, but some model families generalize better than others.

## What Is Versioned In GitHub

The GitHub repository is intended to contain the reproducible source side of the project:

```text

Final_Pipeline/
├── Arabic_Deepfake_Detection_Twitter.ipynb
├── Arabic_Deepfake_Detection_Youtube.ipynb
├── EDA.ipynb
├── results_and_analysis.ipynb
├── deceptive_type_error_analysis_helpers.py
├── exact_model_preprocessing_figures_helpers.py
├── generate_misclassification_analysis_artifacts.py
├── hypothesis_testing_analysis.py
├── misclassification_analysis_plot_helpers.py
└── deception_dist_twitter.png
```

The repository does not rely on GitHub to store large data or generated result bundles.

## Notebook Guide

### `Final_Pipeline/Arabic_Deepfake_Detection_Twitter.ipynb`

This is the full Twitter execution notebook. It is the main end-to-end experiment driver for the Twitter portion of the study.

Main responsibilities:

- Resolve local or Colab project paths
- Validate and de-leak train/validation/test splits
- Apply the `original`, `manual`, and `deepfake_aware` preprocessing views
- Train or load static embeddings
- Run classical ML models
- Run sequence models
- Fine-tune transformer models
- Aggregate metrics across families
- Export figures, tables, reports, and misclassification artifacts
- Run statistical testing and cross-platform evaluation

The notebook sections explicitly cover:

- Setup and configuration
- Data loading and split validation
- Preprocessing definitions and dataset caching
- Static embeddings and representations
- ML models
- Sequence models
- Transformer models
- Result aggregation
- Statistical testing
- Error analysis
- Cross-platform evaluation
- Final exports and artifact audit

### `Final_Pipeline/Arabic_Deepfake_Detection_Youtube.ipynb`

This notebook mirrors the Twitter notebook, but uses the YouTube dataset as the primary in-domain platform. It follows the same pipeline structure and produces a parallel output bundle under the YouTube output directory.

It handles:

- YouTube split loading and validation
- Preprocessing variants
- ML, sequence, and transformer experiments
- Statistical testing
- Error analysis
- Transfer evaluation against Twitter
- Exported tables and figures for the YouTube branch of the study

### `Final_Pipeline/EDA.ipynb`

This notebook is a lightweight, local-first exploratory analysis notebook focused on the training split only. Its purpose is to generate publication-ready descriptive views without retraining the models.

It covers:

- Dataset overview
- Label balance
- Text length distribution
- Signal prevalence comparisons for Real vs Fake
- Deception subtype distribution
- Preprocessing and tokenization explanation tables

It writes outputs to:

- `Final_Pipeline/Output/eda/figures/`
- `Final_Pipeline/Output/eda/tables/`

### `Final_Pipeline/results_and_analysis.ipynb`

This notebook is the compact reporting notebook. It does not retrain the full models. Instead, it consolidates exported CSV artifacts already produced by the execution notebooks and the hypothesis script.

It summarizes:

- In-domain performance
- `H1` transformer-family support
- `H2` preprocessing effects
- Error analysis by deceptive type
- Cross-platform transfer and `H3`
- Final decision summary

The notebook text explicitly states that its tables are derived from exported CSV artifacts rather than hand-entered values.

## Helper Scripts

### `Final_Pipeline/hypothesis_testing_analysis.py`

This script builds the compact post-hoc reporting layer. It reads previously exported output tables and writes:

- `Final_Pipeline/Output/hypothesis_tests/*.csv`
- `Final_Pipeline/Output/hypothesis_tests/hypothesis_testing_summary.md`
- `Final_Pipeline/results_and_analysis.ipynb`

This is the correct script to run after both platform execution notebooks have produced their output tables.

### `Final_Pipeline/generate_misclassification_analysis_artifacts.py`

This script standardizes raw misclassification exports and produces a cleaner cross-platform artifact bundle for deception-type error analysis.

It generates:

- Stacked deceptive-type error-rate figures
- FP/FN summary tables
- FP/FN-by-deception-type tables
- An artifact manifest and input inventory

By default it writes under `Final_Pipeline/Output/`, and it also accepts a custom `--output-dir`.

### `Final_Pipeline/deceptive_type_error_analysis_helpers.py`

This module defines deceptive-type normalization, plotting utilities, and selection logic used for deception-type error summaries and exact-model rosters.

### `Final_Pipeline/exact_model_preprocessing_figures_helpers.py`

This module contains the shared figure-building and comparison utilities for exact-model preprocessing figures, confidence intervals, and family-level rendering.

### `Final_Pipeline/misclassification_analysis_plot_helpers.py`

This module loads and standardizes misclassification exports, aligns configuration identifiers, computes FP/FN summaries, and renders stacked error-rate panels.

## Model Families And Study Configurations

The versioned code and notebook content show the following core families and configurations:

### Classical ML

- `LinearSVC`
- `LogisticRegression`
- `RandomForest`

These are evaluated with lexical and static-representation variants such as:

- `tfidf`
- `word2vec_cbow`
- `fasttext`

### Sequence Models

- `LSTM`
- `BiLSTM`

The sequence pipeline uses random or static embedding initializations, including:

- `random`
- `word2vec_cbow`
- `fasttext`

### Transformers

- `AraBERTv2`
- `MARBERTv2`
- `CAMeLBERT`

## Local Output Structure

`Final_Pipeline/Output/` is not stored on GitHub, but the local workspace currently follows a clear structure:

```text
Final_Pipeline/Output/
├── eda/
├── hypothesis_tests/
├── important/
├── misclassification_analysis_artifacts_error_rate_panels_tighter/
├── twitter/
└── youtube/
```

### `Final_Pipeline/Output/eda/`

EDA tables and publication-style descriptive figures are stored here.

Representative local files:

- `eda/figures/label_balance.png`
- `eda/figures/text_length_distribution.png`
- `eda/figures/deception_subtype_distribution.png`
- `eda/tables/dataset_overview_train_split.csv`
- `eda/tables/preprocessing_tokenization_comparison.csv`

### `Final_Pipeline/Output/twitter/` and `Final_Pipeline/Output/youtube/`

These are the main per-platform experiment bundles. Each currently contains the same high-level subdirectories:

- `cache/`
- `figures/`
- `misclass_analysis/`
- `ml_classifiers/`
- `sequence_models/`
- `tables/`

Typical contents include:

- Cached preprocessed train/validation/test splits
- Per-family result tables
- Classification reports
- Saved model artifacts
- Training-history exports
- Misclassification exports
- Aggregate experiment result tables
- Final figure exports in PNG and PDF

Representative local figure files include:

- `twitter/figures/twitter_exact_models_original_preprocessing_macro_f1_ci.png`
- `twitter/figures/twitter_family_preprocessing_comparison.png`
- `twitter/figures/twitter_deceptive_type_error_rate_original_exact_model_roster.png`
- `youtube/figures/youtube_exact_models_original_preprocessing_macro_f1_ci.png`
- `youtube/figures/youtube_family_preprocessing_comparison.png`
- `youtube/figures/youtube_deceptive_type_error_rate_original_exact_model_roster.png`

Representative local table files include:

- `twitter/tables/experiment_results_long.csv`
- `twitter/tables/in_domain_predictions_long.csv`
- `twitter/tables/best_family_preprocessing_summary.csv`
- `twitter/tables/transfer_significance_tests.csv`
- `youtube/tables/experiment_results_long.csv`
- `youtube/tables/in_domain_predictions_long.csv`
- `youtube/tables/best_family_preprocessing_summary.csv`
- `youtube/tables/transfer_significance_tests.csv`

### `Final_Pipeline/Output/important/`

This directory acts as a report-facing subset of especially useful per-platform artifacts. In the current local workspace it includes, for example:

- `important/twitter/deceptive_type_error_rate_summary.csv`
- `important/twitter/all_models_cross_platform_results.csv`
- `important/youtube/deceptive_type_error_rate_summary.csv`
- `important/youtube/all_models_preprocessing_summary.csv`

### `Final_Pipeline/Output/hypothesis_tests/`

This directory contains the compact tables used to support the final reporting notebook.

Representative local files:

- `hypothesis_tests/in_domain_exact_model_overview.csv`
- `hypothesis_tests/h1_family_support_tests.csv`
- `hypothesis_tests/h2_family_wilcoxon_summary.csv`
- `hypothesis_tests/h3_transfer_group_summary.csv`
- `hypothesis_tests/hypothesis_decision_summary.csv`
- `hypothesis_tests/hypothesis_testing_summary.md`

### `Final_Pipeline/Output/misclassification_analysis_artifacts_error_rate_panels_tighter/`

This is a generated post-processing bundle focused on tighter stacked error-rate panels and FP/FN summaries derived from raw misclassification exports.

Representative local files:

- `misclassification_analysis_artifacts_error_rate_panels_tighter/figures/twitter/twitter_deceptive_type_error_rate_all_preprocessing_stacked_exact_models.png`
- `misclassification_analysis_artifacts_error_rate_panels_tighter/figures/youtube/youtube_deceptive_type_error_rate_all_preprocessing_stacked_exact_models.png`
- `misclassification_analysis_artifacts_error_rate_panels_tighter/tables/all_platforms_fp_fn_summary.csv`
- `misclassification_analysis_artifacts_error_rate_panels_tighter/tables/all_platforms_fp_fn_by_deceptive_type.csv`
- `misclassification_analysis_artifacts_error_rate_panels_tighter/tables/misclassification_analysis_artifact_manifest.csv`

## Current Local Findings Snapshot

The current generated reporting notebook summarizes the local run as follows:

- `H1` is supported: the transformer family is the strongest in-domain family on both platforms.
- Best reported Twitter in-domain result: `CAMeLBERT` with `original` preprocessing, `Macro-F1 = 0.9912`.
- Best reported YouTube in-domain result: `MARBERTv2` with `original` preprocessing, `Macro-F1 = 0.9409`.
- `H2` is partially supported: preprocessing matters, but the effect is not uniformly beneficial across all families and all views.
- `H3` is supported: cross-platform transfer loss is real, and transformers are the most stable family overall.
- Hard deceptive types are not evenly distributed. In the current summary, `omission` is the hardest Twitter subtype, while `contradiction` and `omission` are the hardest YouTube subtypes.

These numbers and conclusions come from the generated local reporting notebook and its supporting CSV tables, so they may change if the full pipeline is rerun with modified data, settings, or exports.

## Recommended Execution Order

If you want to rebuild the local analysis bundle from scratch, use this order:

1. Place the dataset under `Dataset/` using the expected directory layout.
2. Run `Final_Pipeline/EDA.ipynb` if you want descriptive figures and preprocessing/tokenization tables.
3. Run `Final_Pipeline/Arabic_Deepfake_Detection_Twitter.ipynb`.
4. Run `Final_Pipeline/Arabic_Deepfake_Detection_Youtube.ipynb`.
5. Run `python Final_Pipeline/hypothesis_testing_analysis.py`.
6. Run `python Final_Pipeline/generate_misclassification_analysis_artifacts.py` after the per-platform misclassification exports exist.

## Environment Notes

### Execution notebooks

The two main execution notebooks are written with a Colab-style bootstrap section:

- The first code cell intentionally restarts the runtime with `os.kill(os.getpid(), 9)`.
- The next setup cell installs pinned dependencies with `pip`.
- The path-resolution logic supports both a local repository layout and Google Colab with Drive mounting.

Practical implication:

- In Google Colab, the bootstrap flow is already aligned with the notebook design.
- In a local Jupyter environment, you should create the environment yourself and skip the forced-restart/install cells if they are not appropriate for your setup.

### Core Python dependencies observed in the notebooks and scripts

The codebase imports these major packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `gensim`
- `torch`
- `transformers`
- `datasets`
- `plotly`
- `emoji`
- `joblib`
- `nbformat`
- `openpyxl`

There is no versioned `requirements.txt` in the current repository snapshot, so environment recreation is currently notebook-driven rather than package-file-driven.

## Consistency Notes

- This README intentionally describes `Dataset/` and `Final_Pipeline/Output/` as local expected paths, even though they are not part of the GitHub repository.
- Any file path listed inside those folders should be read as a generated or local-only artifact path.
- The versioned source of truth for the research logic is the notebooks and helper scripts under `Final_Pipeline/`.
- The compact reporting layer depends on previously exported CSV artifacts; it is not a substitute for the full execution notebooks.

## Summary

This repository is the source-code and notebook layer of a two-platform Arabic deepfake-text detection study. The dataset and generated outputs are deliberately kept out of GitHub, while the notebooks and helper modules preserve the experimental workflow, the statistical analysis logic, and the reporting pipeline needed to rebuild the local research artifacts.
