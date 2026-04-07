# Arabic Deepfake Text Detection on Social Media: Controlled Preprocessing and Cross-Platform Transferability

This repository contains the notebooks, helper scripts, and local analysis artifacts used for the study *Arabic Deepfake Text Detection on Social Media: Controlled Preprocessing and Cross-Platform Transferability*. The project evaluates Arabic deepfake text detection across Twitter/X and YouTube under three controlled preprocessing views (`original`, `manual`, and `deepfake_aware`) and three model families (classical machine learning, sequence models, and fine-tuned Arabic transformers).

The paper's main conclusion is straightforward: Arabic transformers are the strongest in-domain models, preprocessing materially changes results, and cross-platform transfer remains harder than same-platform evaluation.

> Note
>
> This README follows the manuscript for the study framing and uses the exported CSV artifacts under `Final_Pipeline/Output/` for the reproducible numbers in the current workspace. The local split summaries reflect cleaned and de-leaked working splits, so they do not match every high-level corpus count written in the manuscript exactly.

## Overview

- Task: binary classification of Arabic `Real` vs `Fake` text
- Platforms: Twitter/X and YouTube
- Evaluation settings: in-domain testing and cross-platform transfer without retraining
- Text views: `original`, `manual`, `deepfake_aware`
- Model families: ML, sequence models, Arabic transformers
- Label convention used by the code: `1 = Real`, `0 = Fake`

## Key Findings

| Finding | Result |
| --- | --- |
| Best Twitter in-domain result | `CAMeLBERT` with `original` preprocessing, Macro-F1 `0.9912` |
| Best YouTube in-domain result | `MARBERTv2` with `original` preprocessing, Macro-F1 `0.9409` |
| Best Twitter to YouTube transfer | `MARBERTv2` with `manual` preprocessing, Macro-F1 `0.8364` |
| Best YouTube to Twitter transfer | `AraBERTv2` with `original` preprocessing, Macro-F1 `0.9433` |
| Strongest family overall | Fine-tuned Arabic transformers |
| Preprocessing takeaway | `manual` cleaning hurts most; `deepfake_aware` is usually a better compromise but does not consistently beat `original` |
| Hardest deception types | `omission` on Twitter; `contradiction` and `omission` on YouTube |
| Hypothesis status | `H1` supported, `H2` partially supported, `H3` supported |

## Research Questions and Hypotheses

The study is organized around four research questions:

- `RQ1`: Which ML, sequence, and transformer configurations perform best under in-domain evaluation on Twitter and YouTube?
- `RQ2`: How do preprocessing variants affect in-domain performance across representations and model families?
- `RQ3`: How robust are source-trained models under cross-platform transfer between Twitter and YouTube?
- `RQ4`: Where do errors concentrate across deceptive types and metadata categories?

The manuscript tests three hypotheses:

- `H1`: Fine-tuned contextual Arabic transformers will achieve the highest performance on both platforms.
- `H2`: Preprocessing effects will be significant across representation strategies and model families.
- `H3`: Models perform better in-domain than under cross-platform transfer.

## Dataset and Benchmark Design

The benchmark pairs real Arabic social media text with prompt-generated deceptive rewrites. The Twitter/X portion contains tweets and tweet comments collected across domains such as education, health, advertisements, religion, politics, economy, and sports. The YouTube portion contains Arabic comments collected from public videos through the YouTube Data API v3. The paper describes the benchmark as platform-paired and approximately balanced between real and synthetic text.

### Corpus Summary

| Platform | Real data source | Paper-reported real texts | Paper-reported total texts | Current local modeling split summary |
| --- | --- | ---: | ---: | --- |
| Twitter/X | Tweets and post comments | 14,403 | 28,806 | train `19,800`, val `5,657`, test `2,829` |
| YouTube | Public video comments | 14,854 | 29,711 | train `15,846`, val `2,265`, test `5,654` |

The local split summaries come from `Output/twitter/tables/dataset_split_summary_twitter.csv` and `Output/youtube/tables/dataset_split_summary_youtube.csv`. The YouTube split reflects exact-text de-leakage in the exported workspace artifacts.

### Deception Types Used for Generation

The paper defines six deception strategies for synthetic rewriting:

| Deception type | Description | Twitter count | YouTube count |
| --- | --- | ---: | ---: |
| Exaggeration | Amplifies claims or emotion while preserving the topic | 2,289 | 2,543 |
| Omission | Removes important context to create a misleading reading | 2,286 | 2,403 |
| Contradiction | Introduces plausible statements that reverse the source meaning | 2,299 | 2,406 |
| Satirical tone | Rewrites the text with irony or mockery while keeping it misleading | 2,376 | 2,441 |
| Clickbait phrasing | Uses attention-grabbing or emotionally charged wording | 2,364 | 2,421 |
| Mixed truths | Blends correct information with fabricated or misleading details | 2,333 | 2,423 |

## Preprocessing Views

Preprocessing is treated as an experimental factor, not just routine cleanup.

| View | Intent | Behavior |
| --- | --- | --- |
| `original` | Reference condition | Uses the stored benchmark text as-is after initial corpus construction |
| `manual` | Strongest cleaning condition | Removes URLs, mentions, hashtags, emojis, diacritics, Latin spans, and aggressively normalizes repetition and punctuation |
| `deepfake_aware` | Selective normalization | Preserves or tokenizes potentially informative cues such as URLs, mentions, hashtags, and number spans depending on the platform |

In the paper's framing, `manual` cleaning tends to strip away useful platform and stylistic cues, while `deepfake_aware` tries to preserve signals that may help distinguish human and AI-generated text.

## Model Families

The pipeline compares multiple representation and modeling strategies under the same evaluation protocol.

| Family | Models | Representations / embeddings |
| --- | --- | --- |
| Classical ML | `LinearSVC`, `LogisticRegression`, `RandomForest` | `TF-IDF`, `Word2Vec-CBOW`, `FastText` |
| Sequence models | `LSTM`, `BiLSTM` | `random`, `Word2Vec-CBOW`, `FastText` |
| Transformers | `AraBERTv2`, `MARBERTv2`, `CAMeLBERT` | Contextual tokenization and fine-tuning |

Core protocol reported in the paper:

- maximum input length `128`
- fixed random seed `42`
- sequence models trained for `5` epochs
- transformer models trained for `5` epochs with learning rate `2e-5`
- validation Macro-F1 used for checkpoint selection
- exact-text de-leakage applied before model fitting

## Main Results

### In-Domain Performance

| Platform | View | ML mean | Sequence mean | Transformer mean | Best model | Best Macro-F1 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Twitter | `original` | 0.9581 | 0.9757 | 0.9906 | `CAMeLBERT` | 0.9912 |
| Twitter | `manual` | 0.9093 | 0.9344 | 0.9596 | `MARBERTv2` | 0.9625 |
| Twitter | `deepfake_aware` | 0.9548 | 0.9736 | 0.9857 | `MARBERTv2` | 0.9869 |
| YouTube | `original` | 0.8678 | 0.8898 | 0.9390 | `MARBERTv2` | 0.9409 |
| YouTube | `manual` | 0.8587 | 0.8826 | 0.9303 | `MARBERTv2` | 0.9354 |
| YouTube | `deepfake_aware` | 0.8645 | 0.8888 | 0.9318 | `MARBERTv2` | 0.9369 |

These results show a stable ranking on both platforms: transformers first, sequence models second, classical ML third. The `original` view is the strongest overall condition, and `manual` preprocessing causes the clearest performance drop.

### Cross-Platform Transfer

| Source | Target | View | Best ML | ML Macro-F1 | Best sequence | Sequence Macro-F1 | Best transformer | Transformer Macro-F1 |
| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: |
| Twitter | YouTube | `original` | `LinearSVC` | 0.6904 | `BiLSTM` | 0.6569 | `CAMeLBERT` | 0.7461 |
| Twitter | YouTube | `manual` | `LogisticRegression` | 0.7861 | `BiLSTM` | 0.8048 | `MARBERTv2` | 0.8364 |
| Twitter | YouTube | `deepfake_aware` | `LinearSVC` | 0.7613 | `BiLSTM` | 0.6506 | `CAMeLBERT` | 0.7206 |
| YouTube | Twitter | `original` | `LinearSVC` | 0.7453 | `BiLSTM` | 0.7358 | `AraBERTv2` | 0.9433 |
| YouTube | Twitter | `manual` | `RandomForest` | 0.7700 | `BiLSTM` | 0.7129 | `AraBERTv2` | 0.9070 |
| YouTube | Twitter | `deepfake_aware` | `LinearSVC` | 0.7552 | `LSTM` | 0.7030 | `AraBERTv2` | 0.9099 |

The local summary exports report an average Macro-F1 drop of `0.2549` for Twitter to YouTube transfer and `0.1941` for YouTube to Twitter transfer. Transformers remain the most transfer-robust family in both directions.

### Hypothesis Decisions

| Hypothesis | Verdict | Interpretation |
| --- | --- | --- |
| `H1` | Supported | Transformers significantly outperform the strongest ML and sequence baselines on both platforms |
| `H2` | Partially supported | Preprocessing effects are real, but they are selective rather than uniformly significant across all families |
| `H3` | Supported | Cross-platform transfer is consistently worse than in-domain testing, although transformers lose the least |

### Error Analysis by Deception Type

| Deception type | Twitter mean error (%) | YouTube mean error (%) |
| --- | ---: | ---: |
| Original real texts | 4.3 | 10.0 |
| Clickbait phrasing | 0.5 | 3.8 |
| Contradiction | 8.2 | 24.7 |
| Exaggeration | 1.6 | 7.1 |
| Mixed truths | 1.9 | 13.2 |
| Omission | 17.7 | 24.4 |
| Satirical tone | 1.3 | 7.0 |

The hardest cases are not the most obviously sensational ones. The dominant failure modes are omission on Twitter and omission plus contradiction on YouTube, which suggests that subtle semantic distortion remains harder than surface-level exaggeration or clickbait phrasing.

### Local Result Artifacts

The current workspace already contains exported figures and tables under `Final_Pipeline/Output/`. Representative local artifacts include:

Local README previews below render when the generated `Output/` directory is present in the workspace.

![Twitter exact-model Macro-F1 under original preprocessing](Output/twitter/figures/twitter_exact_models_original_preprocessing_macro_f1_ci.png)

![YouTube exact-model Macro-F1 under original preprocessing](Output/youtube/figures/youtube_exact_models_original_preprocessing_macro_f1_ci.png)

![Twitter family-level preprocessing comparison](Output/twitter/figures/twitter_family_preprocessing_comparison.png)

![YouTube family-level preprocessing comparison](Output/youtube/figures/youtube_family_preprocessing_comparison.png)

- `Output/twitter/figures/twitter_exact_models_original_preprocessing_macro_f1_ci.png`
- `Output/youtube/figures/youtube_exact_models_original_preprocessing_macro_f1_ci.png`
- `Output/twitter/figures/twitter_family_preprocessing_comparison.png`
- `Output/youtube/figures/youtube_family_preprocessing_comparison.png`
- `Output/misclassification_analysis_artifacts_error_rate_panels_tighter/figures/twitter/twitter_deceptive_type_error_rate_all_preprocessing_stacked_exact_models.png`
- `Output/misclassification_analysis_artifacts_error_rate_panels_tighter/figures/youtube/youtube_deceptive_type_error_rate_all_preprocessing_stacked_exact_models.png`

Because `Output/` is ignored by Git, these artifacts are available locally in this workspace but are not intended to be the version-controlled source of truth.

## Repository Guide

```text
Final_Pipeline/
|-- Arabic_Deepfake_Detection_Twitter.ipynb
|-- Arabic_Deepfake_Detection_Youtube.ipynb
|-- EDA.ipynb
|-- results_and_analysis.ipynb
|-- hypothesis_testing_analysis.py
|-- generate_misclassification_analysis_artifacts.py
|-- deceptive_type_error_analysis_helpers.py
|-- exact_model_preprocessing_figures_helpers.py
|-- misclassification_analysis_plot_helpers.py
`-- README.md
```

### Notebook Roles

- `Arabic_Deepfake_Detection_Twitter.ipynb`: full Twitter/X execution pipeline
- `Arabic_Deepfake_Detection_Youtube.ipynb`: full YouTube execution pipeline
- `EDA.ipynb`: descriptive analysis, label balance, text length, signal prevalence, and preprocessing examples
- `results_and_analysis.ipynb`: compact reporting notebook built from exported artifacts

### Script Roles

- `hypothesis_testing_analysis.py`: reads exported platform tables and writes the compact hypothesis-testing bundle under `Output/hypothesis_tests/`
- `generate_misclassification_analysis_artifacts.py`: builds stacked error-rate panels, FP/FN summaries, and misclassification manifests
- `deceptive_type_error_analysis_helpers.py`: deceptive-type normalization and error analysis helpers
- `exact_model_preprocessing_figures_helpers.py`: shared utilities for exact-model preprocessing comparisons and confidence intervals
- `misclassification_analysis_plot_helpers.py`: misclassification loading, alignment, aggregation, and plotting helpers

## Output Layout

The generated local output bundle is organized as follows:

```text
Final_Pipeline/Output/
|-- eda/
|-- hypothesis_tests/
|-- important/
|-- misclassification_analysis_artifacts_error_rate_panels_tighter/
|-- twitter/
`-- youtube/
```

Useful output locations:

- `Output/eda/`: descriptive figures and EDA tables
- `Output/twitter/` and `Output/youtube/`: per-platform experiment outputs, cached splits, training history, figures, and tables
- `Output/hypothesis_tests/`: final support tables for `H1`, `H2`, and `H3`
- `Output/important/`: compact report-facing tables
- `Output/misclassification_analysis_artifacts_error_rate_panels_tighter/`: cross-platform error analysis panels and FP/FN summaries

## Reproducing the Study

Recommended execution order:

1. Place the datasets under `Dataset/` using the expected local layout.
2. Run `Final_Pipeline/EDA.ipynb` to regenerate descriptive figures and preprocessing examples.
3. Run `Final_Pipeline/Arabic_Deepfake_Detection_Twitter.ipynb`.
4. Run `Final_Pipeline/Arabic_Deepfake_Detection_Youtube.ipynb`.
5. Run `python Final_Pipeline/hypothesis_testing_analysis.py`.
6. Run `python Final_Pipeline/generate_misclassification_analysis_artifacts.py`.

## Environment Notes

The two main execution notebooks were written with a Colab-oriented bootstrap flow:

- the first setup cell restarts the runtime
- the next setup cell installs pinned packages with `pip`
- path resolution supports both Google Colab and local execution

If you run the notebooks locally, create the Python environment yourself and skip the forced restart or install cells when they are not appropriate.

Observed core dependencies include:

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

There is currently no versioned `requirements.txt`, so environment recreation is notebook-driven.

## Data Availability

According to the manuscript, the datasets used in the study were developed in an earlier project phase and are being prepared for publication. Access is expected to be handled through the corresponding author and project stakeholders once release is approved.

## Citation

If you use this repository, cite the accompanying manuscript:

```text
Abdulaziz Alqahtani, Amal Sunba, and Tarek Helmy.
"Arabic Deepfake Text Detection on Social Media: Controlled Preprocessing and Cross-Platform Transferability."
2026 manuscript.
```
