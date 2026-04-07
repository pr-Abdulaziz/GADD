from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_fscore_support


SHARED_EMBEDDING_POOL = ["word2vec_cbow", "fasttext"]
ML_LEXICAL_CONFIGS = [
    ("linearsvc", "tfidf"),
    ("logisticregression", "tfidf"),
    ("randomforest", "tfidf"),
]
ML_DENSE_MODELS = ["linearsvc", "logisticregression", "randomforest"]
SEQUENCE_ARCH_ORDER = ["lstm", "bilstm"]
SEQUENCE_BASELINES = ["random"]
TRANSFORMER_ORDER = ["arabertv2", "marbertv2", "camelbert"]
FAMILY_ORDER = ["ML", "Sequence", "Transformer"]
FAMILY_COLORS = {"ML": "#C44E52", "Sequence": "#55A868", "Transformer": "#4C72B0"}
MODEL_SHORT = {
    "linearsvc": "LSVC",
    "logisticregression": "LR",
    "randomforest": "RF",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "arabertv2": "AraBERT",
    "marbertv2": "MARBERTv2",
    "camelbert": "CAMeLBERT",
}
EMBED_SHORT = {
    "tfidf": "TFIDF",
    "word2vec_cbow": "CBOW",
    "fasttext": "FT",
    "random": "Rand",
    "arabertv2": "AraBERT",
    "marbertv2": "MARBERTv2",
    "camelbert": "CAMeLBERT",
}


@dataclass(frozen=True)
class FigureRenderContext:
    preprocessing_order: list[str]
    pipeline_name: str
    display_name: str
    table_dir: object
    fig_dir: object
    save_bundle: Callable
    export_csv: Callable
    apply_axis_style: Callable
    preprocessing_label: Callable
    significance_stars: Callable
    family_labels: dict[str, str]
    rc_params: dict
    seed: int = 42
    default_max_len: int = 128
    figure_size: tuple[float, float] = (8.6, 6.0)


def normalize_pair_value(value):
    if pd.isna(value):
        return "__NA__"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def attach_config_uid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    id_cols = [
        col
        for col in [
            "model_family",
            "model_name",
            "config_name",
            "preprocessing",
            "embedding_family",
            "embedding_name",
            "representation_type",
            "max_len",
        ]
        if col in out.columns
    ]
    if out.empty:
        out["config_uid"] = pd.Series(dtype="object")
        return out
    out["config_uid"] = out[id_cols].apply(
        lambda row: "||".join(f"{col}={normalize_pair_value(row[col])}" for col in id_cols),
        axis=1,
    )
    return out


def ensure_evaluation_scope(df: pd.DataFrame, *, default: str = "in_domain") -> pd.DataFrame:
    out = df.copy()
    if "evaluation_scope" not in out.columns:
        out["evaluation_scope"] = default
    else:
        out["evaluation_scope"] = out["evaluation_scope"].fillna(default)
    return out


def bootstrap_macro_f1_ci(y_true, y_pred, *, n_boot: int = 2000, seed: int = 42):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return {"ci_lower": np.nan, "ci_upper": np.nan}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    scores = np.empty(n_boot, dtype=float)
    for boot_idx in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        scores[boot_idx] = precision_recall_fscore_support(
            y_true[sample],
            y_pred[sample],
            average="macro",
            zero_division=0,
        )[2]
    return {
        "ci_lower": float(np.percentile(scores, 2.5)),
        "ci_upper": float(np.percentile(scores, 97.5)),
    }


def bootstrap_macro_f1_difference(y_true, pred_a, pred_b, *, n_boot: int = 2000, seed: int = 42):
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    if len(y_true) == 0:
        return {"mean_diff": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    diffs = np.empty(n_boot, dtype=float)
    for boot_idx in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        score_a = precision_recall_fscore_support(
            y_true[sample],
            pred_a[sample],
            average="macro",
            zero_division=0,
        )[2]
        score_b = precision_recall_fscore_support(
            y_true[sample],
            pred_b[sample],
            average="macro",
            zero_division=0,
        )[2]
        diffs[boot_idx] = score_a - score_b
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
    }


def bh_adjust(p_values):
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
    adjusted_indices = np.where(valid)[0][order]
    adjusted[adjusted_indices] = adjusted_valid
    return adjusted


def build_in_domain_prediction_frame(
    *,
    eval_results: dict,
    ml_results: dict,
    sequence_model_runs: dict,
    datasets_by_prep: dict,
    pipeline_name: str,
) -> pd.DataFrame:
    rows = []
    for (_, preprocessing, max_len), metrics in eval_results.items():
        preds = np.asarray(metrics["predictions"], dtype=int)
        gold = np.asarray(metrics["labels"], dtype=int)
        for sample_idx, (y_true, y_pred) in enumerate(zip(gold, preds)):
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "y_true": int(y_true),
                    "y_pred": int(y_pred),
                    "correct": int(y_true == y_pred),
                    "platform": pipeline_name,
                    "comparison_scope": "in_domain",
                    "model_family": "Transformer",
                    "model_name": metrics["model_name"],
                    "config_name": metrics["model_name"],
                    "preprocessing": preprocessing,
                    "embedding_family": "contextual_transformer",
                    "embedding_name": metrics["model_name"],
                    "representation_type": "contextual_transformer",
                    "max_len": max_len,
                }
            )
    for preprocessing, config_dict in ml_results.items():
        if preprocessing not in datasets_by_prep:
            continue
        y_true = np.asarray(datasets_by_prep[preprocessing]["test"]["label"], dtype=int)
        for config_name, result in config_dict.items():
            preds = np.asarray(result["test_predictions"], dtype=int)
            for sample_idx, (gold, pred) in enumerate(zip(y_true, preds)):
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "y_true": int(gold),
                        "y_pred": int(pred),
                        "correct": int(gold == pred),
                        "platform": pipeline_name,
                        "comparison_scope": "in_domain",
                        "model_family": "ML",
                        "model_name": result["model_name"],
                        "config_name": config_name,
                        "preprocessing": preprocessing,
                        "embedding_family": result["embedding_family"],
                        "embedding_name": result["embedding_name"],
                        "representation_type": result["representation_type"],
                        "max_len": np.nan,
                    }
                )
    for preprocessing, config_dict in sequence_model_runs.items():
        for config_name, run_data in config_dict.items():
            gold = np.asarray(run_data["y_test"], dtype=int)
            preds = np.asarray(run_data["test_predictions"], dtype=int)
            for sample_idx, (y_true, y_pred) in enumerate(zip(gold, preds)):
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "y_true": int(y_true),
                        "y_pred": int(y_pred),
                        "correct": int(y_true == y_pred),
                        "platform": pipeline_name,
                        "comparison_scope": "in_domain",
                        "model_family": "Sequence",
                        "model_name": run_data["model_name"],
                        "config_name": config_name,
                        "preprocessing": preprocessing,
                        "embedding_family": run_data["embedding_family"],
                        "embedding_name": run_data["embedding_name"],
                        "representation_type": run_data["representation_type"],
                        "max_len": run_data.get("max_len", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def shared_embedding_rank_table(results_df: pd.DataFrame) -> pd.DataFrame:
    ranked = ensure_evaluation_scope(results_df)
    ranked = ranked[ranked["evaluation_scope"] == "in_domain"].copy()
    shared_df = ranked[
        ranked["model_family"].isin(["ML", "Sequence"])
        & ranked["embedding_name"].isin(SHARED_EMBEDDING_POOL)
    ].copy()
    if shared_df.empty:
        return pd.DataFrame(columns=["embedding_name", "mean_macro_f1", "best_macro_f1", "pool_rank"])
    rank_df = (
        shared_df.groupby("embedding_name", as_index=False)
        .agg(mean_macro_f1=("macro_f1", "mean"), best_macro_f1=("macro_f1", "max"))
        .reset_index(drop=True)
    )
    pool_rank = {name: idx for idx, name in enumerate(SHARED_EMBEDDING_POOL)}
    rank_df["pool_rank"] = rank_df["embedding_name"].map(pool_rank)
    return rank_df.sort_values(
        ["mean_macro_f1", "best_macro_f1", "pool_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def select_shared_top_embeddings(results_df: pd.DataFrame, *, top_n: int = 3) -> list[str]:
    rank_df = shared_embedding_rank_table(results_df)
    selected = rank_df["embedding_name"].tolist()[:top_n]
    for embedding_name in SHARED_EMBEDDING_POOL:
        if embedding_name not in selected:
            selected.append(embedding_name)
    return selected[:top_n]


def exact_label(model_family: str, model_name: str, embedding_name: str) -> str:
    if model_family == "Transformer":
        return MODEL_SHORT.get(model_name, model_name)
    return f"{MODEL_SHORT.get(model_name, model_name)}-{EMBED_SHORT.get(embedding_name, embedding_name)}"


def exact_tick_label(model_family: str, model_name: str, embedding_name: str) -> str:
    if model_family == "Transformer":
        return MODEL_SHORT.get(model_name, model_name)
    return f"{MODEL_SHORT.get(model_name, model_name)}\n{EMBED_SHORT.get(embedding_name, embedding_name)}"


def build_exact_model_comparison_roster(
    results_df: pd.DataFrame,
    *,
    preprocessing_order: list[str],
    pipeline_name: str,
    default_max_len: int = 128,
    top_n_embeddings: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_columns = [
        "platform",
        "preprocessing",
        "model_family",
        "model_name",
        "config_name",
        "embedding_family",
        "embedding_name",
        "representation_type",
        "max_len",
        "accuracy",
        "macro_f1",
        "config_uid",
        "template_key",
        "exact_label",
        "tick_label",
        "family_order",
        "base_plot_order",
        "plot_order",
        "selection_note",
        "selection_embedding_rank",
        "selection_embedding_pool",
        "figure_bar_count_target",
    ]
    if results_df is None or results_df.empty:
        return pd.DataFrame(columns=base_columns), pd.DataFrame()

    ranked = ensure_evaluation_scope(attach_config_uid(results_df.copy()))
    ranked = ranked[ranked["evaluation_scope"] == "in_domain"].copy()
    ranked = ranked.sort_values(["macro_f1", "accuracy", "config_uid"], ascending=[False, False, True]).reset_index(drop=True)
    rank_df = shared_embedding_rank_table(ranked)
    selected_embeddings = select_shared_top_embeddings(ranked, top_n=top_n_embeddings)
    selected_rank = {name: idx + 1 for idx, name in enumerate(selected_embeddings)}
    family_rank = {name: idx for idx, name in enumerate(FAMILY_ORDER)}

    specs: list[dict] = []
    for model_name, embedding_name in ML_LEXICAL_CONFIGS:
        specs.append(
            {
                "model_family": "ML",
                "model_name": model_name,
                "embedding_name": embedding_name,
                "max_len": np.nan,
                "selection_note": "tfidf_lexical_baseline",
            }
        )
    for embedding_name in selected_embeddings:
        for model_name in ML_DENSE_MODELS:
            specs.append(
                {
                    "model_family": "ML",
                    "model_name": model_name,
                    "embedding_name": embedding_name,
                    "max_len": np.nan,
                    "selection_note": "shared_top_embedding",
                }
            )
    for model_name in SEQUENCE_ARCH_ORDER:
        for embedding_name in [*SEQUENCE_BASELINES, *selected_embeddings]:
            specs.append(
                {
                    "model_family": "Sequence",
                    "model_name": model_name,
                    "embedding_name": embedding_name,
                    "max_len": default_max_len,
                    "selection_note": "sequence_baseline_or_shared_top_embedding",
                }
            )
    for model_name in TRANSFORMER_ORDER:
        specs.append(
            {
                "model_family": "Transformer",
                "model_name": model_name,
                "embedding_name": model_name,
                "max_len": default_max_len,
                "selection_note": "contextual_transformer_full_set",
            }
        )

    rows: list[dict] = []
    for prep_idx, preprocessing in enumerate(preprocessing_order):
        prep_df = ranked[ranked["preprocessing"].astype(str) == str(preprocessing)].copy()
        for base_order, spec in enumerate(specs):
            family_df = prep_df[prep_df["model_family"].astype(str) == spec["model_family"]].copy()
            family_df = family_df[family_df["model_name"].astype(str) == spec["model_name"]].copy()
            family_df = family_df[family_df["embedding_name"].astype(str) == spec["embedding_name"]].copy()
            if spec["model_family"] in {"Sequence", "Transformer"}:
                family_df = family_df[
                    family_df["max_len"].fillna(spec["max_len"]).astype(float) == float(spec["max_len"])
                ]
            if family_df.empty:
                continue
            chosen = family_df.sort_values(["macro_f1", "accuracy", "config_uid"], ascending=[False, False, True]).iloc[0].copy()
            chosen["template_key"] = f"{chosen['model_family']}::{chosen['model_name']}::{chosen['embedding_name']}"
            chosen["exact_label"] = exact_label(chosen["model_family"], chosen["model_name"], chosen["embedding_name"])
            chosen["tick_label"] = exact_tick_label(chosen["model_family"], chosen["model_name"], chosen["embedding_name"])
            chosen["family_order"] = family_rank.get(str(chosen["model_family"]), 999)
            chosen["base_plot_order"] = int(base_order)
            chosen["plot_order"] = int(prep_idx * len(specs) + base_order)
            chosen["selection_note"] = spec["selection_note"]
            chosen["selection_embedding_rank"] = selected_rank.get(str(chosen["embedding_name"]), np.nan)
            chosen["selection_embedding_pool"] = ",".join(selected_embeddings)
            chosen["figure_bar_count_target"] = int(len(specs))
            chosen["platform"] = pipeline_name
            rows.append(chosen.to_dict())

    roster_df = pd.DataFrame(rows)
    if roster_df.empty:
        return pd.DataFrame(columns=base_columns), rank_df
    roster_df = roster_df[base_columns].sort_values(["preprocessing", "base_plot_order"]).reset_index(drop=True)
    return roster_df, rank_df


def compute_macro_f1_ci_from_predictions(pred_df: pd.DataFrame, *, seed: int) -> dict:
    pred_df = pred_df.sort_values("sample_idx").reset_index(drop=True)
    macro_f1 = float(
        precision_recall_fscore_support(
            pred_df["y_true"],
            pred_df["y_pred"],
            average="macro",
            zero_division=0,
        )[2]
    )
    ci = bootstrap_macro_f1_ci(pred_df["y_true"], pred_df["y_pred"], seed=seed)
    return {
        "macro_f1": macro_f1,
        "ci_lower": float(ci["ci_lower"]),
        "ci_upper": float(ci["ci_upper"]),
        "n_samples": int(len(pred_df)),
    }


def build_preprocessing_exact_model_source(
    *,
    preprocessing: str,
    roster_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    pipeline_name: str,
    seed: int,
) -> pd.DataFrame:
    prep_roster = roster_df[roster_df["preprocessing"].astype(str) == str(preprocessing)].copy()
    rows = []
    for rank_idx, row in enumerate(prep_roster.sort_values("base_plot_order").itertuples(index=False), start=1):
        config_pred = pred_df[pred_df["config_uid"].astype(str) == str(row.config_uid)].sort_values("sample_idx")
        if config_pred.empty:
            continue
        metric_row = compute_macro_f1_ci_from_predictions(config_pred, seed=seed)
        rows.append(
            {
                "platform": pipeline_name,
                "preprocessing": preprocessing,
                "model_family": row.model_family,
                "model_name": row.model_name,
                "config_name": row.config_name,
                "embedding_family": row.embedding_family,
                "embedding_name": row.embedding_name,
                "representation_type": row.representation_type,
                "max_len": row.max_len,
                "config_uid": row.config_uid,
                "template_key": row.template_key,
                "exact_label": row.exact_label,
                "tick_label": row.tick_label,
                "metric_name": "Macro-F1",
                "metric_value": metric_row["macro_f1"],
                "ci_lower": metric_row["ci_lower"],
                "ci_upper": metric_row["ci_upper"],
                "n_samples": metric_row["n_samples"],
                "base_plot_order": int(row.base_plot_order),
                "rank_within_figure": int(rank_idx),
                "selection_note": row.selection_note,
                "selection_embedding_rank": row.selection_embedding_rank,
                "selection_embedding_pool": row.selection_embedding_pool,
                "figure_bar_count_target": int(row.figure_bar_count_target),
            }
        )
    return pd.DataFrame(rows).sort_values("base_plot_order").reset_index(drop=True)


def build_preprocessing_exact_model_significance(
    *,
    plot_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    pipeline_name: str,
    significance_stars_fn: Callable,
    seed: int,
) -> pd.DataFrame:
    columns = [
        "platform",
        "preprocessing",
        "model_family",
        "comparison_a",
        "comparison_b",
        "significance_reference",
        "config_uid_a",
        "config_uid_b",
        "model_name_a",
        "model_name_b",
        "test_name",
        "p_value",
        "p_adjusted",
        "adjustment_method",
        "macro_f1_delta",
        "comparison_ci_lower",
        "comparison_ci_upper",
        "discordant_reference_only",
        "discordant_variant_only",
        "n_samples",
        "favors",
        "significance_flag",
        "significance_label",
        "significance_source",
        "is_reference_bar",
    ]
    if plot_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for family_label, family_df in plot_df.groupby("model_family", sort=False):
        family_df = family_df.sort_values(["metric_value", "base_plot_order"], ascending=[False, True]).reset_index(drop=True)
        reference = family_df.iloc[0]
        reference_pred = pred_df[pred_df["config_uid"].astype(str) == str(reference["config_uid"])].sort_values("sample_idx")
        rows.append(
            {
                "platform": pipeline_name,
                "preprocessing": reference["preprocessing"],
                "model_family": family_label,
                "comparison_a": reference["exact_label"],
                "comparison_b": reference["exact_label"],
                "significance_reference": reference["exact_label"],
                "config_uid_a": reference["config_uid"],
                "config_uid_b": reference["config_uid"],
                "model_name_a": reference["model_name"],
                "model_name_b": reference["model_name"],
                "test_name": "McNemar exact",
                "p_value": np.nan,
                "p_adjusted": np.nan,
                "adjustment_method": "BH-FDR",
                "macro_f1_delta": 0.0,
                "comparison_ci_lower": 0.0,
                "comparison_ci_upper": 0.0,
                "discordant_reference_only": 0,
                "discordant_variant_only": 0,
                "n_samples": int(reference_pred.shape[0]) if not reference_pred.empty else 0,
                "favors": reference["exact_label"],
                "significance_flag": False,
                "significance_label": "ref",
                "significance_source": "within_family_reference",
                "is_reference_bar": True,
            }
        )
        for _, variant in family_df.iloc[1:].iterrows():
            variant_pred = pred_df[pred_df["config_uid"].astype(str) == str(variant["config_uid"])].sort_values("sample_idx")
            if reference_pred.empty or variant_pred.empty:
                continue
            pair_df = reference_pred[["sample_idx", "y_true", "y_pred", "correct"]].merge(
                variant_pred[["sample_idx", "y_true", "y_pred", "correct"]],
                on=["sample_idx", "y_true"],
                suffixes=("_reference", "_variant"),
                how="inner",
            ).sort_values("sample_idx")
            if pair_df.empty:
                continue
            discordant_reference_only = int(np.sum(pair_df["correct_reference"] & ~pair_df["correct_variant"]))
            discordant_variant_only = int(np.sum(~pair_df["correct_reference"] & pair_df["correct_variant"]))
            n_discordant = discordant_reference_only + discordant_variant_only
            p_value = 1.0 if n_discordant == 0 else float(binomtest(min(discordant_reference_only, discordant_variant_only), n=n_discordant, p=0.5).pvalue)
            diff_ci = bootstrap_macro_f1_difference(
                pair_df["y_true"],
                pair_df["y_pred_variant"],
                pair_df["y_pred_reference"],
                seed=seed,
            )
            delta = float(variant["metric_value"] - reference["metric_value"])
            rows.append(
                {
                    "platform": pipeline_name,
                    "preprocessing": variant["preprocessing"],
                    "model_family": family_label,
                    "comparison_a": reference["exact_label"],
                    "comparison_b": variant["exact_label"],
                    "significance_reference": reference["exact_label"],
                    "config_uid_a": reference["config_uid"],
                    "config_uid_b": variant["config_uid"],
                    "model_name_a": reference["model_name"],
                    "model_name_b": variant["model_name"],
                    "test_name": "McNemar exact",
                    "p_value": p_value,
                    "p_adjusted": np.nan,
                    "adjustment_method": "BH-FDR",
                    "macro_f1_delta": delta,
                    "comparison_ci_lower": float(diff_ci["ci_lower"]),
                    "comparison_ci_upper": float(diff_ci["ci_upper"]),
                    "discordant_reference_only": discordant_reference_only,
                    "discordant_variant_only": discordant_variant_only,
                    "n_samples": int(pair_df.shape[0]),
                    "favors": variant["exact_label"] if delta > 0 else reference["exact_label"],
                    "significance_flag": False,
                    "significance_label": "",
                    "significance_source": "McNemar exact vs within-family best; BH-FDR adjusted within figure",
                    "is_reference_bar": False,
                }
            )
    sig_df = pd.DataFrame(rows, columns=columns)
    non_ref = ~sig_df["is_reference_bar"].astype(bool)
    if non_ref.any():
        sig_df.loc[non_ref, "p_adjusted"] = bh_adjust(sig_df.loc[non_ref, "p_value"].to_numpy(dtype=float))
        sig_df.loc[non_ref, "significance_flag"] = sig_df.loc[non_ref, "p_adjusted"] < 0.05
        sig_df.loc[non_ref, "significance_label"] = sig_df.loc[non_ref, "p_adjusted"].map(significance_stars_fn)
        sig_df.loc[sig_df["significance_label"] == "ns", "significance_label"] = ""
    return sig_df


def compute_shared_exact_model_ylim(source_frames: list[pd.DataFrame]) -> tuple[float, float]:
    valid_frames = [frame for frame in source_frames if frame is not None and not frame.empty]
    if not valid_frames:
        return (0.60, 1.00)
    combined = pd.concat(valid_frames, ignore_index=True)
    lower = float(combined["ci_lower"].min())
    upper = float(combined["ci_upper"].max())
    y_min = max(0.60, np.floor((lower - 0.015) / 0.02) * 0.02)
    y_max = min(1.00, np.ceil((upper + 0.03) / 0.02) * 0.02)
    if y_max <= y_min:
        y_max = min(1.00, y_min + 0.12)
    return float(y_min), float(y_max)


def _blend_color(hex_color: str, blend: float) -> str:
    rgb = np.array(matplotlib.colors.to_rgb(hex_color), dtype=float)
    return matplotlib.colors.to_hex(rgb + (1.0 - rgb) * float(blend))


def build_exact_model_color_map(roster_df: pd.DataFrame) -> dict[str, str]:
    color_map: dict[str, str] = {}
    template_rows = (
        roster_df[["template_key", "model_family", "base_plot_order"]]
        .drop_duplicates()
        .sort_values(["model_family", "base_plot_order"])
        .reset_index(drop=True)
    )
    for family_label, family_df in template_rows.groupby("model_family", sort=False):
        blends = np.linspace(0.05, 0.55, num=max(1, len(family_df)))
        for blend, row in zip(blends, family_df.itertuples(index=False)):
            color_map[row.template_key] = _blend_color(FAMILY_COLORS.get(family_label, "#666666"), float(blend))
    return color_map


def _x_positions(plot_df: pd.DataFrame):
    bar_step = 0.88
    family_gap = 0.34
    positions = []
    family_centers = {}
    boundaries = []
    cursor = 0.0
    for family_index, family_label in enumerate(FAMILY_ORDER):
        family_df = plot_df[plot_df["model_family"].astype(str) == family_label].sort_values("base_plot_order")
        if family_df.empty:
            continue
        family_positions = []
        for _ in family_df.itertuples(index=False):
            positions.append(cursor)
            family_positions.append(cursor)
            cursor += bar_step
        family_centers[family_label] = float(np.mean(family_positions))
        if family_index < len(FAMILY_ORDER) - 1:
            boundaries.append(cursor - (bar_step / 2.0) + (family_gap / 2.0))
            cursor += family_gap
    return np.asarray(positions, dtype=float), family_centers, boundaries


def plot_preprocessing_exact_model_bars(
    *,
    plot_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    context: FigureRenderContext,
    stem: str,
    y_limits: tuple[float, float],
    color_map: dict[str, str],
) -> pd.DataFrame | None:
    if plot_df.empty:
        return None
    export_df = plot_df.merge(
        sig_df[
            [
                "comparison_b",
                "significance_reference",
                "p_value",
                "p_adjusted",
                "significance_label",
                "significance_flag",
                "favors",
                "is_reference_bar",
            ]
        ],
        left_on="exact_label",
        right_on="comparison_b",
        how="left",
    )
    export_df["significance_reference"] = export_df["significance_reference"].fillna("")
    export_df["significance_label"] = export_df["significance_label"].fillna("")
    export_cols = [
        "platform",
        "preprocessing",
        "model_family",
        "model_name",
        "embedding_name",
        "exact_label",
        "metric_name",
        "metric_value",
        "ci_lower",
        "ci_upper",
        "significance_reference",
        "p_value",
        "p_adjusted",
        "significance_label",
        "rank_within_figure",
        "config_name",
        "embedding_family",
        "representation_type",
        "max_len",
        "config_uid",
        "selection_note",
        "selection_embedding_rank",
        "selection_embedding_pool",
        "figure_bar_count_target",
    ]
    context.export_csv(export_df[export_cols], stem)

    with plt.rc_context(context.rc_params):
        n_bars = int(plot_df.shape[0])
        fig_width = max(float(context.figure_size[0]), min(14.0, 2.0 + (0.34 * n_bars)))
        fig_height = float(context.figure_size[1])
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        positions, family_centers, boundaries = _x_positions(plot_df)
        values = plot_df["metric_value"].to_numpy(dtype=float)
        lower = plot_df["ci_lower"].to_numpy(dtype=float)
        upper = plot_df["ci_upper"].to_numpy(dtype=float)
        bar_colors = [color_map.get(key, "#666666") for key in plot_df["template_key"]]
        ref_labels = set(sig_df.loc[sig_df["is_reference_bar"].astype(bool), "comparison_b"].astype(str))
        linewidths = [1.8 if label in ref_labels else 1.0 for label in plot_df["exact_label"].astype(str)]
        edgecolors = ["#24313B" if label in ref_labels else "#3D4A56" for label in plot_df["exact_label"].astype(str)]

        ax.bar(positions, values, width=0.80, color=bar_colors, edgecolor=edgecolors, linewidth=linewidths, zorder=3)
        ax.errorbar(
            positions,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="none",
            ecolor="#26313B",
            elinewidth=0.9,
            capsize=2.8,
            capthick=0.9,
            zorder=4,
        )

        for boundary in boundaries:
            ax.axvline(boundary, color="#D3D8DE", linewidth=0.8, linestyle=":", zorder=1)

        ax.set_xticks(positions)
        ax.set_xticklabels(plot_df["tick_label"], rotation=58, ha="right", fontsize=7.2, linespacing=0.95)
        ax.tick_params(axis="x", pad=2)
        context.apply_axis_style(
            ax,
            ylabel="Macro-F1",
            # title=f"{context.display_name}: Exact-Model Macro-F1 ({context.preprocessing_label(plot_df['preprocessing'].iloc[0])})",
        )
        # if ax.get_title():
        #     ax.set_title(ax.get_title(), pad=16)
        ax.set_ylim(*y_limits)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.set_xlim(positions.min() - 0.55, positions.max() + 0.55)
        ax.legend(
            handles=[Patch(facecolor=FAMILY_COLORS[fam], edgecolor="#2F3A45", label=context.family_labels.get(fam, fam)) for fam in FAMILY_ORDER],
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            fontsize=8.0,
            columnspacing=1.2,
            handlelength=1.4,
        )
        # for family_label, center in family_centers.items():
        #     ax.text(
        #         center,
        #         -0.34,
        #         context.family_labels.get(family_label, family_label),
        #         transform=ax.get_xaxis_transform(),
        #         ha="center",
        #         va="top",
        #         fontsize=8.0,
        #         color="#43515C",
        #         fontweight="semibold",
        #     )
        fig.tight_layout(rect=(0.02, 0.20, 0.99, 0.92))
        context.save_bundle(fig, stem)
        plt.show()
    return export_df[export_cols]


def render_exact_model_preprocessing_figure_set(
    *,
    results_df: pd.DataFrame,
    eval_results: dict,
    ml_results: dict,
    sequence_model_runs: dict,
    datasets_by_prep: dict,
    context: FigureRenderContext,
) -> dict[str, pd.DataFrame]:
    roster_df, rank_df = build_exact_model_comparison_roster(
        results_df,
        preprocessing_order=context.preprocessing_order,
        pipeline_name=context.pipeline_name,
        default_max_len=context.default_max_len,
    )
    pred_df = attach_config_uid(
        build_in_domain_prediction_frame(
            eval_results=eval_results,
            ml_results=ml_results,
            sequence_model_runs=sequence_model_runs,
            datasets_by_prep=datasets_by_prep,
            pipeline_name=context.pipeline_name,
        )
    )
    sources = {
        preprocessing: build_preprocessing_exact_model_source(
            preprocessing=preprocessing,
            roster_df=roster_df,
            pred_df=pred_df,
            pipeline_name=context.pipeline_name,
            seed=context.seed,
        )
        for preprocessing in context.preprocessing_order
    }
    y_limits = compute_shared_exact_model_ylim(list(sources.values()))
    color_map = build_exact_model_color_map(roster_df)
    manifest_rows = []
    for preprocessing in context.preprocessing_order:
        plot_df = sources[preprocessing]
        if plot_df.empty:
            continue
        sig_df = build_preprocessing_exact_model_significance(
            plot_df=plot_df,
            pred_df=pred_df,
            pipeline_name=context.pipeline_name,
            significance_stars_fn=context.significance_stars,
            seed=context.seed,
        )
        stem = f"{context.pipeline_name}_exact_models_{preprocessing}_preprocessing_macro_f1_ci"
        plot_preprocessing_exact_model_bars(plot_df=plot_df, sig_df=sig_df, context=context, stem=stem, y_limits=y_limits, color_map=color_map)
        manifest_rows.append(
            {
                "platform": context.pipeline_name,
                "preprocessing": preprocessing,
                "stem": stem,
                "n_bars": int(plot_df.shape[0]),
                "figure_bar_count_target": int(plot_df["figure_bar_count_target"].iloc[0]),
                "shared_embedding_pool": str(plot_df["selection_embedding_pool"].iloc[0]),
            }
        )
    return {
        "roster": roster_df,
        "embedding_rank": rank_df,
        "manifest": pd.DataFrame(manifest_rows),
        "predictions": pred_df,
    }
