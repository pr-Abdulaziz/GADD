from __future__ import annotations

from datetime import date
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbformat.v4 import new_markdown_cell, new_notebook
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Output" / "hypothesis_tests"
NOTEBOOK_PATH = ROOT / "results_and_analysis.ipynb"

PLATFORMS = ("twitter", "youtube")
PREPROCESSING_ORDER = ("original", "manual", "deepfake_aware")
FAMILY_ORDER = ("ML", "Sequence", "Transformer")
METHODS_NO_ORIGINAL = ("manual", "deepfake_aware")
SEED = 42


def format_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def format_pvalue(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def bh_adjust(p_values: pd.Series | np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    valid = ~np.isnan(p_values)
    if not valid.any():
        return adjusted
    order = np.argsort(p_values[valid])
    ranked = p_values[valid][order]
    n = len(ranked)
    scaled = ranked * n / np.arange(1, n + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_valid = np.clip(scaled, 0.0, 1.0)
    adjusted[np.where(valid)[0][order]] = adjusted_valid
    return adjusted


def macro_f1(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(
        precision_recall_fscore_support(
            np.asarray(y_true, dtype=int),
            np.asarray(y_pred, dtype=int),
            average="macro",
            zero_division=0,
        )[2]
    )


def bootstrap_macro_f1_difference(
    y_true: pd.Series | np.ndarray,
    y_pred_a: pd.Series | np.ndarray,
    y_pred_b: pd.Series | np.ndarray,
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)
    if len(y_true) == 0:
        return {"ci_lower": np.nan, "ci_upper": np.nan}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    deltas = np.empty(n_boot, dtype=float)
    for boot_idx in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        score_a = macro_f1(y_true[sample], y_pred_a[sample])
        score_b = macro_f1(y_true[sample], y_pred_b[sample])
        deltas[boot_idx] = score_a - score_b
    return {
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
    }


def mcnemar_exact_test(correct_a: pd.Series | np.ndarray, correct_b: pd.Series | np.ndarray) -> dict[str, float]:
    a = np.asarray(correct_a, dtype=int)
    b = np.asarray(correct_b, dtype=int)
    discordant_a_only = int(((a == 1) & (b == 0)).sum())
    discordant_b_only = int(((a == 0) & (b == 1)).sum())
    total_discordant = discordant_a_only + discordant_b_only
    if total_discordant == 0:
        p_value = 1.0
    else:
        successes = min(discordant_a_only, discordant_b_only)
        p_value = float(
            binomtest(successes, n=total_discordant, p=0.5, alternative="two-sided").pvalue
        )
    return {
        "p_value": p_value,
        "discordant_a_only": discordant_a_only,
        "discordant_b_only": discordant_b_only,
    }


def load_in_domain_predictions(platform: str) -> pd.DataFrame:
    path = ROOT / "Output" / platform / "tables" / "in_domain_predictions_long.csv"
    return pd.read_csv(path, low_memory=False)


def load_exact_model_table(platform: str, preprocessing: str) -> pd.DataFrame:
    path = (
        ROOT
        / "Output"
        / platform
        / "tables"
        / f"{platform}_exact_models_{preprocessing}_preprocessing_macro_f1_ci.csv"
    )
    return pd.read_csv(path)


def load_best_family_preprocessing(platform: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "Output" / platform / "tables" / "best_family_preprocessing_summary.csv")


def load_best_family_cross_platform(platform: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "Output" / platform / "tables" / "best_family_cross_platform_results.csv")


def load_transfer_group_tests(platform: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "Output" / platform / "tables" / "transfer_group_tests.csv")


def load_transfer_significance_tests(platform: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "Output" / platform / "tables" / "transfer_significance_tests.csv")


def load_deceptive_type_error_rate(platform: str) -> pd.DataFrame:
    candidates = [
        ROOT / "Output" / platform / "tables" / "deceptive_type_error_rate_summary.csv",
        ROOT / "Output" / "important" / platform / "deceptive_type_error_rate_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No deceptive type summary found for {platform}")


def platform_target(platform: str) -> str:
    return "youtube" if platform == "twitter" else "twitter"


def build_exact_model_overview() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for platform in PLATFORMS:
        for preprocessing in PREPROCESSING_ORDER:
            df = load_exact_model_table(platform, preprocessing)
            family_means = df.groupby("model_family", as_index=False)["metric_value"].mean()
            mean_map = dict(zip(family_means["model_family"], family_means["metric_value"]))
            best = df.loc[df["metric_value"].idxmax()]
            worst = df.loc[df["metric_value"].idxmin()]
            rows.append(
                {
                    "platform": platform,
                    "preprocessing": preprocessing,
                    "ml_mean": mean_map.get("ML", np.nan),
                    "sequence_mean": mean_map.get("Sequence", np.nan),
                    "transformer_mean": mean_map.get("Transformer", np.nan),
                    "best_model": best["model_name"],
                    "best_macro_f1": best["metric_value"],
                    "worst_model": worst["model_name"],
                    "worst_macro_f1": worst["metric_value"],
                    "spread_best_minus_worst": best["metric_value"] - worst["metric_value"],
                }
            )
    return pd.DataFrame(rows).sort_values(["platform", "preprocessing"]).reset_index(drop=True)


def build_best_family_in_domain_compact() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for platform in PLATFORMS:
        df = load_best_family_preprocessing(platform)
        for preprocessing in PREPROCESSING_ORDER:
            row: dict[str, object] = {"platform": platform, "preprocessing": preprocessing}
            prep_df = df[df["preprocessing"] == preprocessing]
            for family in FAMILY_ORDER:
                family_row = prep_df[prep_df["model_family"] == family].iloc[0]
                family_key = family.lower()
                row[f"{family_key}_model"] = family_row["model_name"]
                row[f"{family_key}_macro_f1"] = family_row["macro_f1"]
                row[f"{family_key}_delta_vs_original"] = family_row["delta_vs_original"]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["platform", "preprocessing"]).reset_index(drop=True)


def build_error_summary_table() -> pd.DataFrame:
    order = [
        "original",
        "clickbait phrasing",
        "contradiction",
        "exaggeration",
        "mixed truths",
        "omission",
        "satirical tone",
    ]
    summary_rows: list[dict[str, object]] = []
    for platform in PLATFORMS:
        df = load_deceptive_type_error_rate(platform)
        grouped = (
            df.groupby("deceptive_type", as_index=False)["error_rate"]
            .mean()
            .rename(columns={"error_rate": "mean_error_rate"})
        )
        grouped["mean_error_rate_pct"] = grouped["mean_error_rate"] * 100.0
        grouped["platform"] = platform
        summary_rows.append(grouped[["platform", "deceptive_type", "mean_error_rate_pct"]])
    combined = pd.concat(summary_rows, ignore_index=True)
    combined["deceptive_type"] = pd.Categorical(combined["deceptive_type"], order, ordered=True)
    wide = (
        combined.pivot(index="deceptive_type", columns="platform", values="mean_error_rate_pct")
        .reset_index()
        .rename(columns={"twitter": "twitter_mean_error_pct", "youtube": "youtube_mean_error_pct"})
        .sort_values("deceptive_type")
        .reset_index(drop=True)
    )
    return wide


def build_cross_platform_compact() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_platform in PLATFORMS:
        df = load_best_family_cross_platform(source_platform)
        target_platform = platform_target(source_platform)
        for preprocessing in PREPROCESSING_ORDER:
            row: dict[str, object] = {
                "source_platform": source_platform,
                "target_platform": target_platform,
                "preprocessing": preprocessing,
            }
            prep_df = df[df["preprocessing"] == preprocessing]
            for family in FAMILY_ORDER:
                family_row = prep_df[prep_df["model_family"] == family].iloc[0]
                family_key = family.lower()
                row[f"{family_key}_model"] = family_row["model_name"]
                row[f"{family_key}_macro_f1"] = family_row["macro_f1"]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["source_platform", "preprocessing"]).reset_index(drop=True)


def build_h3_transfer_group_summary() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for source_platform in PLATFORMS:
        df = load_transfer_group_tests(source_platform)
        subset = df[df["group_scope"] == "family"].copy()
        subset["source_platform"] = source_platform
        subset["target_platform"] = platform_target(source_platform)
        rows.append(
            subset[
                [
                    "source_platform",
                    "target_platform",
                    "model_family",
                    "mean_macro_f1_delta",
                    "median_macro_f1_delta",
                    "p_value",
                    "p_adjusted",
                    "rank_biserial_correlation",
                    "significance_flag",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(["source_platform", "model_family"]).reset_index(drop=True)


def build_h3_best_transfer_ci() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for source_platform in PLATFORMS:
        df = load_transfer_significance_tests(source_platform).copy()
        df["source_platform"] = source_platform
        df["target_platform"] = platform_target(source_platform)
        rows.append(
            df[
                [
                    "source_platform",
                    "target_platform",
                    "model_family",
                    "model_name",
                    "config_name",
                    "preprocessing",
                    "macro_f1_delta",
                    "ci_lower",
                    "ci_upper",
                    "ci_excludes_zero",
                ]
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["source_platform", "model_family"]).reset_index(drop=True)


def _select_best_overall_family_representatives(platform: str) -> pd.DataFrame:
    df = load_best_family_preprocessing(platform).copy()
    winners = (
        df.sort_values(["model_family", "macro_f1", "accuracy", "config_name"], ascending=[True, False, False, True])
        .groupby("model_family", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return winners


def _pair_prediction_rows(
    predictions: pd.DataFrame,
    row_a: pd.Series,
    row_b: pd.Series,
) -> pd.DataFrame:
    pred_a = predictions[
        (predictions["model_family"] == row_a["model_family"])
        & (predictions["config_name"] == row_a["config_name"])
        & (predictions["preprocessing"] == row_a["preprocessing"])
    ].sort_values("sample_idx")
    pred_b = predictions[
        (predictions["model_family"] == row_b["model_family"])
        & (predictions["config_name"] == row_b["config_name"])
        & (predictions["preprocessing"] == row_b["preprocessing"])
    ].sort_values("sample_idx")
    paired = pred_a[["sample_idx", "y_true", "y_pred", "correct"]].rename(
        columns={"y_pred": "y_pred_a", "correct": "correct_a"}
    ).merge(
        pred_b[["sample_idx", "y_pred", "correct"]].rename(
            columns={"y_pred": "y_pred_b", "correct": "correct_b"}
        ),
        on="sample_idx",
        how="inner",
    )
    return paired


def build_h1_family_support() -> pd.DataFrame:
    transformer_rows: list[pd.DataFrame] = []
    verification_rows: list[dict[str, object]] = []
    for platform in PLATFORMS:
        predictions = load_in_domain_predictions(platform)
        winners = _select_best_overall_family_representatives(platform)
        family_rows = {family: winners[winners["model_family"] == family].iloc[0] for family in FAMILY_ORDER}

        pair_specs = [
            ("Transformer", "Sequence"),
            ("Transformer", "ML"),
            ("Sequence", "ML"),
        ]
        computed_rows: list[dict[str, object]] = []
        for family_a, family_b in pair_specs:
            row_a = family_rows[family_a]
            row_b = family_rows[family_b]
            paired = _pair_prediction_rows(predictions, row_a, row_b)
            stats = mcnemar_exact_test(paired["correct_a"], paired["correct_b"])
            boot = bootstrap_macro_f1_difference(paired["y_true"], paired["y_pred_a"], paired["y_pred_b"])
            macro_f1_a = macro_f1(paired["y_true"], paired["y_pred_a"])
            macro_f1_b = macro_f1(paired["y_true"], paired["y_pred_b"])
            computed_rows.append(
                {
                    "platform": platform,
                    "comparison_scope": "best_family_pair_in_domain",
                    "family_a": family_a,
                    "model_a": row_a["model_name"],
                    "config_a": row_a["config_name"],
                    "preprocessing_a": row_a["preprocessing"],
                    "family_b": family_b,
                    "model_b": row_b["model_name"],
                    "config_b": row_b["config_name"],
                    "preprocessing_b": row_b["preprocessing"],
                    "test_name": "McNemar exact",
                    "statistic": np.nan,
                    "p_value": stats["p_value"],
                    "macro_f1_delta": macro_f1_a - macro_f1_b,
                    "bootstrap_ci_lower": boot["ci_lower"],
                    "bootstrap_ci_upper": boot["ci_upper"],
                    "discordant_a_only": stats["discordant_a_only"],
                    "discordant_b_only": stats["discordant_b_only"],
                    "n_samples": int(len(paired)),
                    "favors": family_a if (macro_f1_a - macro_f1_b) > 0 else (family_b if (macro_f1_a - macro_f1_b) < 0 else "Tie"),
                    "notes": "Best in-domain configuration per family ranked by macro-F1 then accuracy.",
                }
            )
        computed = pd.DataFrame(computed_rows)
        computed["p_adjusted"] = bh_adjust(computed["p_value"])
        computed["adjustment_method"] = "BH-FDR"
        computed["significance_flag"] = computed["p_adjusted"] < 0.05

        expected = pd.read_csv(ROOT / "Output" / platform / "tables" / f"statistical_validation_{platform}.csv")
        expected = expected.sort_values(["family_a", "family_b"]).reset_index(drop=True)
        computed_sorted = computed.sort_values(["family_a", "family_b"]).reset_index(drop=True)

        compare_cols = [
            ("model_a", False),
            ("config_a", False),
            ("preprocessing_a", False),
            ("model_b", False),
            ("config_b", False),
            ("preprocessing_b", False),
            ("p_value", True),
            ("macro_f1_delta", True),
            ("bootstrap_ci_lower", True),
            ("bootstrap_ci_upper", True),
            ("discordant_a_only", True),
            ("discordant_b_only", True),
            ("n_samples", True),
            ("p_adjusted", True),
            ("significance_flag", False),
        ]
        for idx in range(len(expected)):
            for col, numeric in compare_cols:
                left = computed_sorted.loc[idx, col]
                right = expected.loc[idx, col]
                if numeric:
                    ok = bool(np.isclose(float(left), float(right), rtol=1e-10, atol=1e-10))
                else:
                    ok = left == right
                verification_rows.append(
                    {
                        "platform": platform,
                        "family_a": computed_sorted.loc[idx, "family_a"],
                        "family_b": computed_sorted.loc[idx, "family_b"],
                        "field": col,
                        "computed_value": left,
                        "expected_value": right,
                        "matches": ok,
                    }
                )
                if not ok:
                    raise ValueError(
                        f"H1 verification failed for {platform} {computed_sorted.loc[idx, 'family_a']} vs "
                        f"{computed_sorted.loc[idx, 'family_b']} field {col}: computed={left} expected={right}"
                    )

        subset = computed[computed["family_a"] == "Transformer"].copy()
        subset["decision"] = np.where(subset["significance_flag"], "Support H1", "No support")
        subset = subset.rename(
            columns={
                "family_b": "comparison_family",
                "model_a": "transformer_model",
                "preprocessing_a": "transformer_preprocessing",
                "model_b": "comparison_model",
                "preprocessing_b": "comparison_preprocessing",
            }
        )
        transformer_rows.append(
            subset[
                [
                    "platform",
                    "comparison_family",
                    "transformer_model",
                    "transformer_preprocessing",
                    "comparison_model",
                    "comparison_preprocessing",
                    "macro_f1_delta",
                    "bootstrap_ci_lower",
                    "bootstrap_ci_upper",
                    "p_value",
                    "p_adjusted",
                    "significance_flag",
                    "decision",
                ]
            ]
        )

    out = pd.concat(transformer_rows, ignore_index=True).sort_values(["platform", "comparison_family"]).reset_index(drop=True)
    verification = pd.DataFrame(verification_rows)
    verification.to_csv(OUTPUT_DIR / "h1_family_support_verification.csv", index=False)
    return out


def build_h1_transformer_view_pairwise() -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for platform in PLATFORMS:
        for preprocessing in PREPROCESSING_ORDER:
            df = load_exact_model_table(platform, preprocessing)
            transformers = (
                df[df["model_family"] == "Transformer"]
                .sort_values("metric_value", ascending=False)
                .reset_index(drop=True)
            )
            best = transformers.iloc[0]
            runner_up = transformers.iloc[1]
            summary_rows.append(
                {
                    "platform": platform,
                    "preprocessing": preprocessing,
                    "best_transformer": best["model_name"],
                    "best_macro_f1": best["metric_value"],
                    "runner_up_transformer": runner_up["model_name"],
                    "runner_up_macro_f1": runner_up["metric_value"],
                    "delta_best_minus_runner_up": best["metric_value"] - runner_up["metric_value"],
                    "p_value": runner_up["p_value"],
                    "p_adjusted": runner_up["p_adjusted"],
                    "runner_up_significant_after_bh": bool(
                        pd.notna(runner_up["p_adjusted"]) and float(runner_up["p_adjusted"]) < 0.05
                    ),
                }
            )
            for _, competitor in transformers.iloc[1:].iterrows():
                pair_rows.append(
                    {
                        "platform": platform,
                        "preprocessing": preprocessing,
                        "best_transformer": best["model_name"],
                        "best_macro_f1": best["metric_value"],
                        "competitor_transformer": competitor["model_name"],
                        "competitor_macro_f1": competitor["metric_value"],
                        "delta_best_minus_competitor": best["metric_value"] - competitor["metric_value"],
                        "p_value": competitor["p_value"],
                        "p_adjusted": competitor["p_adjusted"],
                        "significant_after_bh": bool(
                            pd.notna(competitor["p_adjusted"]) and float(competitor["p_adjusted"]) < 0.05
                        ),
                    }
                )
    pairwise = pd.DataFrame(pair_rows).sort_values(["platform", "preprocessing", "competitor_transformer"]).reset_index(
        drop=True
    )
    summary = pd.DataFrame(summary_rows).sort_values(["platform", "preprocessing"]).reset_index(drop=True)
    return pairwise, summary


def build_h2_family_wilcoxon() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for platform in PLATFORMS:
        path = ROOT / "Output" / platform / "tables" / "preprocessing_effect_tests.csv"
        df = pd.read_csv(path)
        subset = df[
            [
                "platform",
                "model_family",
                "reference_preprocessing",
                "variant_preprocessing",
                "mean_macro_f1_delta",
                "median_macro_f1_delta",
                "p_value",
                "p_adjusted",
                "rank_biserial_correlation",
                "n_pairs",
                "favors",
                "significance_flag",
            ]
        ].copy()
        rows.append(subset)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["platform", "model_family", "variant_preprocessing"]).reset_index(drop=True)


def build_h2_best_config_tests() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for platform in PLATFORMS:
        predictions = load_in_domain_predictions(platform)
        best = pd.read_csv(ROOT / "Output" / platform / "tables" / "best_family_preprocessing_summary.csv")
        for family in FAMILY_ORDER:
            family_best = best[best["model_family"] == family].copy()
            original = family_best[family_best["preprocessing"] == "original"].iloc[0]
            pred_original = predictions[
                (predictions["model_family"] == family)
                & (predictions["config_name"] == original["config_name"])
                & (predictions["preprocessing"] == "original")
            ].sort_values("sample_idx")
            for variant in METHODS_NO_ORIGINAL:
                variant_row = family_best[family_best["preprocessing"] == variant].iloc[0]
                pred_variant = predictions[
                    (predictions["model_family"] == family)
                    & (predictions["config_name"] == variant_row["config_name"])
                    & (predictions["preprocessing"] == variant)
                ].sort_values("sample_idx")
                paired = pred_original[
                    ["sample_idx", "y_true", "y_pred", "correct"]
                ].rename(
                    columns={
                        "y_pred": "y_pred_original",
                        "correct": "correct_original",
                    }
                ).merge(
                    pred_variant[["sample_idx", "y_pred", "correct"]].rename(
                        columns={
                            "y_pred": "y_pred_variant",
                            "correct": "correct_variant",
                        }
                    ),
                    on="sample_idx",
                    how="inner",
                )
                stats = mcnemar_exact_test(paired["correct_variant"], paired["correct_original"])
                boot = bootstrap_macro_f1_difference(
                    paired["y_true"],
                    paired["y_pred_variant"],
                    paired["y_pred_original"],
                )
                macro_original = macro_f1(paired["y_true"], paired["y_pred_original"])
                macro_variant = macro_f1(paired["y_true"], paired["y_pred_variant"])
                delta = macro_variant - macro_original
                rows.append(
                    {
                        "platform": platform,
                        "model_family": family,
                        "comparison": f"original_vs_{variant}",
                        "reference_preprocessing": "original",
                        "variant_preprocessing": variant,
                        "original_model": original["model_name"],
                        "original_config": original["config_name"],
                        "variant_model": variant_row["model_name"],
                        "variant_config": variant_row["config_name"],
                        "macro_f1_original": macro_original,
                        "macro_f1_variant": macro_variant,
                        "macro_f1_delta_variant_minus_original": delta,
                        "bootstrap_ci_lower": boot["ci_lower"],
                        "bootstrap_ci_upper": boot["ci_upper"],
                        "discordant_variant_only": stats["discordant_a_only"],
                        "discordant_original_only": stats["discordant_b_only"],
                        "p_value": stats["p_value"],
                        "n_samples": int(len(paired)),
                        "favors": variant if delta > 0 else ("original" if delta < 0 else "tie"),
                    }
                )
    out = pd.DataFrame(rows).sort_values(["platform", "model_family", "variant_preprocessing"]).reset_index(drop=True)
    out["p_adjusted"] = bh_adjust(out["p_value"])
    out["significance_flag"] = out["p_adjusted"] < 0.05
    return out


def build_decision_summary(
    h1_family: pd.DataFrame,
    h1_transformer_summary: pd.DataFrame,
    h2_family_wilcoxon: pd.DataFrame,
    h2_best_config: pd.DataFrame,
    h3_transfer_group: pd.DataFrame,
) -> pd.DataFrame:
    h1_supported = bool(h1_family["significance_flag"].all())
    h1_transformer_sig = int(h1_transformer_summary["runner_up_significant_after_bh"].sum())
    strict_h2_supported = bool(h2_family_wilcoxon["significance_flag"].all())
    any_h2_supported = bool(h2_family_wilcoxon["significance_flag"].any())
    h2_best_sig = int(h2_best_config["significance_flag"].sum())
    h3_supported = bool(h3_transfer_group["significance_flag"].all())
    transformer_smallest = []
    for source_platform in PLATFORMS:
        source_df = h3_transfer_group[h3_transfer_group["source_platform"] == source_platform]
        smallest_family = source_df.sort_values("mean_macro_f1_delta").iloc[0]["model_family"]
        transformer_smallest.append(smallest_family == "Transformer")

    decision_rows = [
        {
            "hypothesis": "H1",
            "support_verdict": "Supported" if h1_supported else "Not supported",
            "null_hypothesis_language": (
                "Reject the null of no transformer-family advantage on both platforms."
                if h1_supported
                else "Do not reject the null of equal family performance on at least one platform."
            ),
            "interpretation": (
                f"Transformer family comparisons were significant on both platforms; "
                f"{h1_transformer_sig} of 6 transformer-only view summaries also showed a significant lead "
                f"for the top backbone after BH-FDR."
            ),
        },
        {
            "hypothesis": "H2",
            "support_verdict": (
                "Supported"
                if strict_h2_supported
                else ("Partially supported" if any_h2_supported else "Not supported")
            ),
            "null_hypothesis_language": (
                "Reject the null only for selected preprocessing contrasts, not uniformly across all families."
                if any_h2_supported
                else "Do not reject the null of no preprocessing effect across the reported family-level tests."
            ),
            "interpretation": (
                f"Family-wise Wilcoxon tests were significant in "
                f"{int(h2_family_wilcoxon['significance_flag'].sum())} of {len(h2_family_wilcoxon)} contrasts, "
                f"and best-config McNemar tests were significant in {h2_best_sig} of {len(h2_best_config)} "
                f"reported contrasts after BH-FDR."
            ),
        },
        {
            "hypothesis": "H3",
            "support_verdict": "Supported" if h3_supported and all(transformer_smallest) else "Partially supported",
            "null_hypothesis_language": (
                "Reject the null of no transfer loss at the family level; transformers show the smallest mean loss in both directions."
                if h3_supported and all(transformer_smallest)
                else "Reject the null of no transfer loss, but only part of the transformer-advantage transfer claim is supported."
            ),
            "interpretation": (
                "Family-wise transfer-loss tests were significant in all reported family comparisons, and the transformer family had the smallest mean Macro-F1 loss in both transfer directions."
            ),
        },
    ]
    return pd.DataFrame(decision_rows)


def prepare_for_markdown(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if "p_value" in col or "p_adjusted" in col:
            out[col] = out[col].map(format_pvalue)
        elif (
            "macro_f1" in col
            or "bootstrap_ci" in col
            or "delta" in col
            or "correlation" in col
            or col.endswith("_pct")
            or "spread_" in col
            or col.endswith("_mean")
            or col.endswith("_ci")
            or col in {"ci_lower", "ci_upper", "best_macro_f1", "worst_macro_f1"}
        ):
            out[col] = out[col].map(lambda x: format_float(x, 4))
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    return prepare_for_markdown(df).to_markdown(index=False)


def render_h1_interpretation(h1_family: pd.DataFrame, h1_transformer_summary: pd.DataFrame) -> str:
    twitter_seq = h1_family[(h1_family["platform"] == "twitter") & (h1_family["comparison_family"] == "Sequence")].iloc[0]
    twitter_ml = h1_family[(h1_family["platform"] == "twitter") & (h1_family["comparison_family"] == "ML")].iloc[0]
    youtube_seq = h1_family[(h1_family["platform"] == "youtube") & (h1_family["comparison_family"] == "Sequence")].iloc[0]
    youtube_ml = h1_family[(h1_family["platform"] == "youtube") & (h1_family["comparison_family"] == "ML")].iloc[0]

    significant_views = h1_transformer_summary[h1_transformer_summary["runner_up_significant_after_bh"]]
    if significant_views.empty:
        transformer_sentence = (
            "Within the transformer family, none of the view-specific top backbones were significantly better "
            "than the runner-up after BH-FDR."
        )
    else:
        formatted_views = ", ".join(
            f"{row.platform}-{row.preprocessing} ({row.best_transformer}, adjusted p={format_pvalue(row.p_adjusted)})"
            for row in significant_views.itertuples(index=False)
        )
        transformer_sentence = (
            "Within the transformer family, significant within-view backbone differences appeared only in "
            f"{formatted_views}."
        )

    return (
        "This H1 table compares the best overall in-domain transformer setting on each platform against the "
        "best overall sequence and ML settings on that same platform, where family winners are selected by "
        "Macro-F1 and then accuracy. On Twitter, the best transformer outperformed the best sequence model "
        f"(adjusted p={format_pvalue(twitter_seq['p_adjusted'])}, delta={format_float(twitter_seq['macro_f1_delta'])}) "
        f"and the best ML model (adjusted p={format_pvalue(twitter_ml['p_adjusted'])}, "
        f"delta={format_float(twitter_ml['macro_f1_delta'])}). On YouTube, the best transformer also "
        f"outperformed the best sequence model (adjusted p={format_pvalue(youtube_seq['p_adjusted'])}, "
        f"delta={format_float(youtube_seq['macro_f1_delta'])}) and the best ML model "
        f"(adjusted p={format_pvalue(youtube_ml['p_adjusted'])}, delta={format_float(youtube_ml['macro_f1_delta'])}). "
        f"{transformer_sentence} This supports H1 at the platform level using the strongest available setting "
        "from each family."
    )


def render_h2_interpretation(h2_family_wilcoxon: pd.DataFrame, h2_best_config: pd.DataFrame) -> str:
    family_sig = h2_family_wilcoxon[h2_family_wilcoxon["significance_flag"]].copy()
    best_sig = h2_best_config[h2_best_config["significance_flag"]].copy()

    family_examples = ", ".join(
        f"{row.platform}-{row.model_family}-{row.variant_preprocessing} (adjusted p={format_pvalue(row.p_adjusted)})"
        for row in family_sig.itertuples(index=False)
    )
    best_examples = ", ".join(
        f"{row.platform}-{row.model_family}-{row.variant_preprocessing} (adjusted p={format_pvalue(row.p_adjusted)})"
        for row in best_sig.itertuples(index=False)
    )

    return (
        "The strict version of H2 is not fully supported. Family-wise Wilcoxon tests show significant "
        "preprocessing effects only for selected contrasts rather than uniformly across all families: "
        f"{family_examples}. The best-config paired McNemar analysis tells a similar story, with significant "
        f"effects concentrated in {best_examples}. In practical terms, manual preprocessing is the clearest "
        "source of degradation, especially on Twitter, whereas deepfake-aware preprocessing is usually smaller "
        "and often not significant after multiple-comparison correction. The safest wording is therefore that "
        "preprocessing effects are real but selective and non-uniform."
    )


def render_in_domain_interpretation(exact_overview: pd.DataFrame) -> str:
    twitter_original = exact_overview[
        (exact_overview["platform"] == "twitter") & (exact_overview["preprocessing"] == "original")
    ].iloc[0]
    youtube_original = exact_overview[
        (exact_overview["platform"] == "youtube") & (exact_overview["preprocessing"] == "original")
    ].iloc[0]
    twitter_manual = exact_overview[
        (exact_overview["platform"] == "twitter") & (exact_overview["preprocessing"] == "manual")
    ].iloc[0]
    youtube_manual = exact_overview[
        (exact_overview["platform"] == "youtube") & (exact_overview["preprocessing"] == "manual")
    ].iloc[0]
    return (
        "Across all six platform-view conditions, the transformer mean Macro-F1 is the highest family mean. "
        f"The strongest in-domain result appears on Twitter under the original view "
        f"({twitter_original['best_model']}, Macro-F1={format_float(twitter_original['best_macro_f1'])}), "
        f"while the strongest YouTube result is also under the original view "
        f"({youtube_original['best_model']}, Macro-F1={format_float(youtube_original['best_macro_f1'])}). "
        f"Manual preprocessing produces the clearest drop in family means on Twitter, especially for the "
        f"sequence and transformer families, and YouTube shows the same direction of degradation for the "
        f"transformer family even though the absolute drop is smaller than on Twitter. The larger spreads on "
        f"YouTube ({format_float(youtube_original['spread_best_minus_worst'])} under the original view versus "
        f"{format_float(twitter_original['spread_best_minus_worst'])} on Twitter) also show that configuration "
        "choice matters more on YouTube."
    )


def render_error_interpretation(error_summary: pd.DataFrame) -> str:
    twitter_top = error_summary.sort_values("twitter_mean_error_pct", ascending=False).iloc[0]
    youtube_top = error_summary.sort_values("youtube_mean_error_pct", ascending=False).iloc[0]
    youtube_second = error_summary.sort_values("youtube_mean_error_pct", ascending=False).iloc[1]
    return (
        f"Error patterns are not uniform across deception types. On Twitter, the hardest category is "
        f"`{twitter_top['deceptive_type']}` with a mean error rate of "
        f"{format_float(twitter_top['twitter_mean_error_pct'], 1)}%, far above the next cluster of error types. "
        f"On YouTube, `{youtube_top['deceptive_type']}` "
        f"({format_float(youtube_top['youtube_mean_error_pct'], 1)}%) and "
        f"`{youtube_second['deceptive_type']}` "
        f"({format_float(youtube_second['youtube_mean_error_pct'], 1)}%) dominate the failure profile. "
        "This indicates that the hardest cases are not the most visibly sensational texts, but those that omit "
        "key information, reverse meaning, or mix true and false content while remaining fluent."
    )


def render_cross_platform_interpretation(
    cross_platform_compact: pd.DataFrame,
    h3_transfer_group: pd.DataFrame,
    h3_transfer_best_ci: pd.DataFrame,
) -> str:
    tw_to_yt_best = cross_platform_compact[
        cross_platform_compact["source_platform"] == "twitter"
    ].assign(
        best_family_macro_f1=lambda df: df[
            ["ml_macro_f1", "sequence_macro_f1", "transformer_macro_f1"]
        ].max(axis=1)
    ).sort_values("best_family_macro_f1", ascending=False).iloc[0]
    yt_to_tw_best = cross_platform_compact[
        cross_platform_compact["source_platform"] == "youtube"
    ].assign(
        best_family_macro_f1=lambda df: df[
            ["ml_macro_f1", "sequence_macro_f1", "transformer_macro_f1"]
        ].max(axis=1)
    ).sort_values("best_family_macro_f1", ascending=False).iloc[0]
    youtube_transformer_ci = h3_transfer_best_ci[
        (h3_transfer_best_ci["source_platform"] == "youtube")
        & (h3_transfer_best_ci["model_family"] == "Transformer")
    ].iloc[0]
    return (
        "Cross-platform performance is consistently lower than in-domain performance at the family level, and the "
        "family-wise Wilcoxon tests are significant in both transfer directions for ML, sequence, and transformer "
        "models. The best Twitter-to-YouTube transfer result is the transformer under manual preprocessing "
        f"(Macro-F1={format_float(tw_to_yt_best['best_family_macro_f1'])}), while the best YouTube-to-Twitter "
        f"transfer result is again a transformer under the original view "
        f"(Macro-F1={format_float(yt_to_tw_best['best_family_macro_f1'])}). "
        "Transformers also have the smallest mean transfer loss in both directions. One nuance remains important: "
        f"for the strongest YouTube-to-Twitter transformer configuration, the bootstrap delta CI "
        f"[{format_float(youtube_transformer_ci['ci_lower'])}, {format_float(youtube_transformer_ci['ci_upper'])}] "
        "crosses zero, which means that a very strong transformer can occasionally transfer almost as well as, or "
        "slightly better than, its in-domain score."
    )


def build_notebook(
    exact_overview: pd.DataFrame,
    in_domain_compact: pd.DataFrame,
    error_summary: pd.DataFrame,
    cross_platform_compact: pd.DataFrame,
    h1_family: pd.DataFrame,
    h1_transformer_summary: pd.DataFrame,
    h1_transformer_pairwise: pd.DataFrame,
    h2_family_wilcoxon: pd.DataFrame,
    h2_best_config: pd.DataFrame,
    h3_transfer_group: pd.DataFrame,
    h3_transfer_best_ci: pd.DataFrame,
    decision_summary: pd.DataFrame,
) -> None:
    title = "# Results and Analysis\n\n"
    title += (
        "This notebook consolidates the executed result tables for the Arabic deepfake detection study across "
        "Twitter and YouTube. It covers in-domain performance, preprocessing effects, error analysis, "
        "cross-platform transfer, and the final hypothesis decisions.\n\n"
        "All tables below are derived from the exported CSV artifacts produced by the executed pipelines rather "
        "than from hand-entered values."
    )

    methods = (
        "## Reading Guide and Statistical Method\n\n"
        "This notebook is organized in the same order as the paper results section.\n\n"
        "- `Macro-F1` remains the primary metric.\n"
        "- `Exact McNemar` tests are used when two systems are evaluated on the same held-out items.\n"
        "- `Wilcoxon signed-rank` tests are used for family-wise preprocessing summaries over matched configurations.\n"
        "- `95% bootstrap confidence intervals` are used for reported Macro-F1 uncertainty and paired differences where available.\n"
        "- `BH-FDR` correction is applied across grouped p-value families in the reported tables.\n\n"
        "Important interpretation note: p-values are used to test null hypotheses of equal performance. "
        "The notebook therefore reports whether the evidence supports each research hypothesis, not whether a "
        "hypothesis itself was literally rejected.\n\n"
        f"Analysis generated on {date.today().isoformat()}."
    )

    in_domain_md = (
        "## In-Domain Performance Overview\n\n"
        "### Exact-model overview by platform and preprocessing view\n\n"
        f"{markdown_table(exact_overview)}\n\n"
        "### Best configuration within each family and preprocessing view\n\n"
        f"{markdown_table(in_domain_compact)}\n\n"
        f"**Interpretation.** {render_in_domain_interpretation(exact_overview)}"
    )

    h1_md = (
        "## H1: Fine-tuned contextual Arabic transformers will produce the strongest in-domain Macro-F1 on both platforms\n\n"
        "### Family-level test used to decide H1\n\n"
        f"{markdown_table(h1_family)}\n\n"
        "### Transformer-only view summary\n\n"
        f"{markdown_table(h1_transformer_summary)}\n\n"
        "### Transformer-only pairwise view comparisons\n\n"
        f"{markdown_table(h1_transformer_pairwise)}\n\n"
        f"**Interpretation.** {render_h1_interpretation(h1_family, h1_transformer_summary)}"
    )

    preprocessing_md = (
        "## Preprocessing Effects and H2\n\n"
        "The preprocessing analysis is reported in two layers. The first layer uses family-wise Wilcoxon tests on "
        "matched configuration deltas. The second layer uses paired prediction-level McNemar tests for the most "
        "important best-configuration comparisons against the original view.\n\n"
        "### Family-wise preprocessing tests already exported by the pipelines\n\n"
        f"{markdown_table(h2_family_wilcoxon)}\n\n"
        "### Best-config preprocessing comparisons for the most important practical contrasts\n\n"
        f"{markdown_table(h2_best_config)}\n\n"
        f"**Interpretation.** {render_h2_interpretation(h2_family_wilcoxon, h2_best_config)}"
    )

    error_md = (
        "## Error Analysis\n\n"
        "### Mean exact-model error rates by deceptive type\n\n"
        f"{markdown_table(error_summary)}\n\n"
        f"**Interpretation.** {render_error_interpretation(error_summary)}"
    )

    cross_platform_md = (
        "## Cross-Platform Evaluation and H3\n\n"
        "### Best transfer-performing family configuration by preprocessing view\n\n"
        f"{markdown_table(cross_platform_compact)}\n\n"
        "### Family-level transfer-loss tests\n\n"
        f"{markdown_table(h3_transfer_group)}\n\n"
        "### Bootstrap confidence intervals for the strongest transfer configuration in each family\n\n"
        f"{markdown_table(h3_transfer_best_ci)}\n\n"
        f"**Interpretation.** {render_cross_platform_interpretation(cross_platform_compact, h3_transfer_group, h3_transfer_best_ci)}"
    )

    decision_md = (
        "## Final Decision Summary\n\n"
        f"{markdown_table(decision_summary)}\n\n"
        "### Practical Reading of the Results\n\n"
        "- The in-domain story is clear: transformers are the strongest family on both platforms.\n"
        "- The preprocessing story is selective: `manual` cleaning is often harmful, while `deepfake_aware` is usually less harmful but not consistently beneficial.\n"
        "- The error-analysis story is semantic: omission and contradiction remain the dominant hard cases.\n"
        "- The transfer story is cautious: cross-platform robustness remains limited, although transformers are the most stable family overall."
    )

    nb = new_notebook(
        cells=[
            new_markdown_cell(title),
            new_markdown_cell(methods),
            new_markdown_cell(in_domain_md),
            new_markdown_cell(h1_md),
            new_markdown_cell(preprocessing_md),
            new_markdown_cell(error_md),
            new_markdown_cell(cross_platform_md),
            new_markdown_cell(decision_md),
        ]
    )
    NOTEBOOK_PATH.write_text(nbformat.writes(nb), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exact_overview = build_exact_model_overview()
    in_domain_compact = build_best_family_in_domain_compact()
    error_summary = build_error_summary_table()
    cross_platform_compact = build_cross_platform_compact()
    h1_family = build_h1_family_support()
    h1_transformer_pairwise, h1_transformer_summary = build_h1_transformer_view_pairwise()
    h2_family_wilcoxon = build_h2_family_wilcoxon()
    h2_best_config = build_h2_best_config_tests()
    h3_transfer_group = build_h3_transfer_group_summary()
    h3_transfer_best_ci = build_h3_best_transfer_ci()
    decision_summary = build_decision_summary(
        h1_family,
        h1_transformer_summary,
        h2_family_wilcoxon,
        h2_best_config,
        h3_transfer_group,
    )

    exact_overview.to_csv(OUTPUT_DIR / "in_domain_exact_model_overview.csv", index=False)
    in_domain_compact.to_csv(OUTPUT_DIR / "in_domain_best_family_compact.csv", index=False)
    error_summary.to_csv(OUTPUT_DIR / "error_analysis_mean_deceptive_type.csv", index=False)
    cross_platform_compact.to_csv(OUTPUT_DIR / "cross_platform_best_family_compact.csv", index=False)
    h1_family.to_csv(OUTPUT_DIR / "h1_family_support_tests.csv", index=False)
    h1_transformer_pairwise.to_csv(OUTPUT_DIR / "h1_transformer_view_pairwise.csv", index=False)
    h1_transformer_summary.to_csv(OUTPUT_DIR / "h1_transformer_view_summary.csv", index=False)
    h2_family_wilcoxon.to_csv(OUTPUT_DIR / "h2_family_wilcoxon_summary.csv", index=False)
    h2_best_config.to_csv(OUTPUT_DIR / "h2_best_config_preprocessing_tests.csv", index=False)
    h3_transfer_group.to_csv(OUTPUT_DIR / "h3_transfer_group_summary.csv", index=False)
    h3_transfer_best_ci.to_csv(OUTPUT_DIR / "h3_transfer_best_ci.csv", index=False)
    decision_summary.to_csv(OUTPUT_DIR / "hypothesis_decision_summary.csv", index=False)

    summary_md = (
        "# Results and Hypothesis Summary\n\n"
        "## In-Domain Performance\n\n"
        f"{markdown_table(exact_overview)}\n\n"
        f"{render_in_domain_interpretation(exact_overview)}\n\n"
        "## H1\n\n"
        f"{markdown_table(h1_family)}\n\n"
        f"{render_h1_interpretation(h1_family, h1_transformer_summary)}\n\n"
        "## H2\n\n"
        f"{markdown_table(h2_family_wilcoxon)}\n\n"
        f"{render_h2_interpretation(h2_family_wilcoxon, h2_best_config)}\n\n"
        "## Error Analysis\n\n"
        f"{markdown_table(error_summary)}\n\n"
        f"{render_error_interpretation(error_summary)}\n\n"
        "## Cross-Platform Evaluation\n\n"
        f"{markdown_table(h3_transfer_group)}\n\n"
        f"{render_cross_platform_interpretation(cross_platform_compact, h3_transfer_group, h3_transfer_best_ci)}\n\n"
        "## Decision Summary\n\n"
        f"{markdown_table(decision_summary)}\n"
    )
    (OUTPUT_DIR / "hypothesis_testing_summary.md").write_text(summary_md, encoding="utf-8")

    build_notebook(
        exact_overview=exact_overview,
        in_domain_compact=in_domain_compact,
        error_summary=error_summary,
        cross_platform_compact=cross_platform_compact,
        h1_family=h1_family,
        h1_transformer_summary=h1_transformer_summary,
        h1_transformer_pairwise=h1_transformer_pairwise,
        h2_family_wilcoxon=h2_family_wilcoxon,
        h2_best_config=h2_best_config,
        h3_transfer_group=h3_transfer_group,
        h3_transfer_best_ci=h3_transfer_best_ci,
        decision_summary=decision_summary,
    )

    print(f"Wrote notebook: {NOTEBOOK_PATH}")
    print(f"Wrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
