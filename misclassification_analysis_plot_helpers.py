from __future__ import annotations

from pathlib import Path
import logging
import math
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

from exact_model_preprocessing_figures_helpers import (
    EMBED_SHORT,
    FAMILY_ORDER,
    MODEL_SHORT,
    attach_config_uid,
    build_exact_model_comparison_roster,
    ensure_evaluation_scope,
)

try:
    from deceptive_type_error_analysis_helpers import (
        DECEPTIVE_TYPE_NORMALIZATION as EXISTING_DECEPTIVE_TYPE_NORMALIZATION,
        DECEPTIVE_TYPE_ORDER as EXISTING_DECEPTIVE_TYPE_ORDER,
        FAMILY_COLOR_MAP as EXISTING_FAMILY_COLOR_MAP,
        resolve_deception_type_column as existing_resolve_deception_type_column,
    )
except Exception:
    EXISTING_DECEPTIVE_TYPE_ORDER = [
        "original",
        "clickbait phrasing",
        "contradiction",
        "exaggeration",
        "mixed truths",
        "omission",
        "satirical tone",
    ]
    EXISTING_DECEPTIVE_TYPE_NORMALIZATION = {
        "none": "original",
        "orginal": "original",
    }
    EXISTING_FAMILY_COLOR_MAP = {
        "ML": ["#C44E52", "#DD8452", "#E17C05", "#B07AA1", "#937860", "#DA8BC3", "#8C8C8C", "#F28E2B", "#BAB0AC"],
        "Sequence": ["#55A868", "#59A14F", "#7BC47F", "#2F7D4A", "#8CD17D", "#76B7B2"],
        "Transformer": ["#4C72B0", "#64B5CD", "#8172B2"],
    }
    existing_resolve_deception_type_column = None


LOGGER = logging.getLogger(__name__)

SUPPORTED_TABULAR_SUFFIXES = {".csv", ".xls", ".xlsx"}
PREFERRED_PREPROCESSING_ORDER = ["original", "manual", "deepfake_aware"]
PREFERRED_EVALUATION_SCOPE = "in_domain"

POSITIVE_CLASS_CODE = 1
POSITIVE_CLASS_LABEL = "Real"
NEGATIVE_CLASS_CODE = 0
NEGATIVE_CLASS_LABEL = "Fake"

DECEPTIVE_TYPE_ORDER = list(EXISTING_DECEPTIVE_TYPE_ORDER)
DECEPTIVE_TYPE_NORMALIZATION = dict(EXISTING_DECEPTIVE_TYPE_NORMALIZATION)
FAMILY_COLOR_MAP = {family: list(colors) for family, colors in EXISTING_FAMILY_COLOR_MAP.items()}

DECEPTIVE_TYPE_SHORT_LABELS = {
    "original": "Original",
    "clickbait phrasing": "Clickbait\nphrasing",
    "contradiction": "Contradiction",
    "exaggeration": "Exaggeration",
    "mixed truths": "Mixed truths",
    "omission": "Omission",
    "satirical tone": "Satirical\ntone",
}

PAPER_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "font.size": 11.0,
    "axes.titlesize": 13.0,
    "axes.labelsize": 11.8,
    "xtick.labelsize": 9.6,
    "ytick.labelsize": 9.6,
    "legend.fontsize": 8.8,
}

STANDARDIZED_COLUMNS = [
    "platform",
    "evaluation_scope",
    "source_platform",
    "target_platform",
    "sample_idx",
    "text",
    "y_true",
    "y_pred",
    "correct",
    "error",
    "model_identifier",
    "model_name",
    "config_name",
    "model_family",
    "preprocessing",
    "max_len",
    "embedding_family",
    "embedding_name",
    "representation_type",
    "dialect",
    "sector",
    "field",
    "deception_type",
    "deception_type_raw",
    "input_file",
    "input_format",
]

PLOT_SUMMARY_COLUMNS = [
    "platform",
    "evaluation_scope",
    "source_platform",
    "target_platform",
    "preprocessing",
    "model_family",
    "model_name",
    "model_identifier",
    "config_name",
    "embedding_family",
    "embedding_name",
    "representation_type",
    "max_len",
    "deception_type",
    "total_samples",
    "error_count",
    "false_positive_count",
    "false_negative_count",
    "error_rate",
    "config_uid",
    "exact_label",
    "tick_label",
    "plot_order",
    "selection_note",
    "selection_rule",
]

FP_FN_GROUP_COLUMNS = [
    "platform",
    "evaluation_scope",
    "source_platform",
    "target_platform",
    "model_family",
    "model_name",
    "model_identifier",
    "config_name",
    "preprocessing",
    "embedding_family",
    "embedding_name",
    "representation_type",
    "max_len",
]

REQUIRED_INPUT_COLUMNS = {"sample_idx", "y_true", "y_pred", "config_name"}
REQUIRED_STANDARDIZED_COLUMNS = {
    "platform",
    "evaluation_scope",
    "sample_idx",
    "y_true",
    "y_pred",
    "correct",
    "error",
    "model_identifier",
    "model_name",
    "config_name",
    "model_family",
    "preprocessing",
}


def _norm(value) -> str:
    return str(value).strip().lower()


def platform_display_name(platform: str) -> str:
    mapping = {
        "twitter": "Twitter",
        "youtube": "YouTube",
    }
    return mapping.get(_norm(platform), str(platform).replace("_", " ").title())


def preprocessing_display_name(preprocessing: str) -> str:
    mapping = {
        "original": "Original",
        "manual": "Manual",
        "deepfake_aware": "Deepfake-aware",
    }
    return mapping.get(_norm(preprocessing), str(preprocessing).replace("_", " ").title())


def normalize_model_family(value) -> str:
    mapping = {
        "ml": "ML",
        "machine learning": "ML",
        "sequence": "Sequence",
        "dl": "Sequence",
        "deep learning": "Sequence",
        "transformer": "Transformer",
    }
    cleaned = _norm(value)
    if not cleaned:
        return ""
    return mapping.get(cleaned, str(value).strip())


def normalize_preprocessing(value) -> str:
    mapping = {
        "none": "original",
        "orginal": "original",
    }
    cleaned = _norm(value).replace("-", "_").replace(" ", "_")
    if not cleaned:
        return ""
    return mapping.get(cleaned, cleaned)


def normalize_deception_type(value) -> str:
    if pd.isna(value):
        return ""
    cleaned = " ".join(str(value).strip().lower().replace("_", " ").split())
    if not cleaned or cleaned == "nan":
        return ""
    return DECEPTIVE_TYPE_NORMALIZATION.get(cleaned, cleaned)


def ordered_preprocessing_values(values: pd.Series | list[str]) -> list[str]:
    observed = [normalize_preprocessing(value) for value in pd.Series(values).dropna().astype(str)]
    unique = list(dict.fromkeys(value for value in observed if value))
    preferred = [value for value in PREFERRED_PREPROCESSING_ORDER if value in unique]
    remainder = sorted(value for value in unique if value not in preferred)
    return preferred + remainder


def ordered_deception_type_values(values: pd.Series | list[str]) -> list[str]:
    observed = [normalize_deception_type(value) for value in pd.Series(values).dropna().astype(str)]
    unique = list(dict.fromkeys(value for value in observed if value))
    preferred = [value for value in DECEPTIVE_TYPE_ORDER if value in unique]
    remainder = sorted(value for value in unique if value not in preferred)
    return preferred + remainder


def read_tabular_file(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_TABULAR_SUFFIXES:
        raise ValueError(f"Unsupported file suffix for {path}: {suffix}")
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    return pd.read_excel(path, nrows=nrows)


def peek_tabular_columns(path: Path) -> list[str]:
    return list(read_tabular_file(path, nrows=0).columns)


def resolve_deception_type_column(df: pd.DataFrame) -> str | None:
    if existing_resolve_deception_type_column is not None:
        resolved = existing_resolve_deception_type_column(df)
        if resolved is not None:
            return resolved
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in ["deception_type", "deceptive_type", "deception category", "deceptive category"]:
        if candidate in normalized:
            return normalized[candidate]
    for col in df.columns:
        key = str(col).strip().lower()
        if ("deception" in key or "deceptive" in key) and "type" in key:
            return col
    return None


def infer_platform_from_path(path: Path) -> str:
    for part in path.parts:
        normalized = _norm(part)
        if normalized in {"twitter", "youtube"}:
            return normalized
    return ""


def infer_family_from_frame(df: pd.DataFrame, path: Path) -> str:
    if "model_family" in df.columns:
        families = [normalize_model_family(value) for value in df["model_family"].dropna().astype(str).unique()]
        families = [value for value in families if value]
        unique = list(dict.fromkeys(families))
        if len(unique) == 1:
            return unique[0]
    if "classifier" in df.columns:
        return "ML"
    stem = _norm(path.stem)
    if "transformer" in stem or "trans" in stem:
        return "Transformer"
    if "sequence" in stem or "seq" in stem:
        return "Sequence"
    if "ml" in stem:
        return "ML"
    return ""


def inspect_misclassification_tabular_files(project_root: Path) -> pd.DataFrame:
    output_root = project_root / "Output"
    if not output_root.exists():
        raise FileNotFoundError(f"Output directory not found: {output_root}")

    records: list[dict] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_TABULAR_SUFFIXES:
            continue
        if "misclass_analysis" not in {_norm(part) for part in path.parts}:
            continue
        try:
            columns = peek_tabular_columns(path)
            normalized_columns = {_norm(column) for column in columns}
            selected_for_loading = REQUIRED_INPUT_COLUMNS.issubset(normalized_columns)
            reason = (
                "prediction_export"
                if selected_for_loading
                else "missing item-level prediction columns required for misclassification loading"
            )
        except Exception as exc:
            columns = []
            selected_for_loading = False
            reason = f"column inspection failed: {exc}"
        records.append(
            {
                "platform": infer_platform_from_path(path),
                "relative_path": path.relative_to(project_root).as_posix(),
                "file_format": path.suffix.lower().lstrip("."),
                "selected_for_loading": bool(selected_for_loading),
                "selection_reason": reason,
                "column_count": len(columns),
                "columns": " | ".join(str(column) for column in columns),
            }
        )
    inventory_df = pd.DataFrame(records)
    if inventory_df.empty:
        raise FileNotFoundError(f"No tabular files were found under {output_root}/*/misclass_analysis")
    return inventory_df.sort_values(["platform", "relative_path"]).reset_index(drop=True)


def _coalesce_series(df: pd.DataFrame, columns: list[str], *, default=np.nan) -> pd.Series:
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    result = df[available[0]].copy()
    for column in available[1:]:
        result = result.where(result.notna(), df[column])
    return result


def _to_int_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64")


def standardize_prediction_frame(df: pd.DataFrame, *, source_path: Path, project_root: Path) -> pd.DataFrame:
    out = df.copy()

    inferred_platform = infer_platform_from_path(source_path)
    out["platform"] = _coalesce_series(out, ["platform"], default=inferred_platform).astype(str).str.strip().str.lower()
    out["platform"] = out["platform"].replace({"": inferred_platform})

    out["evaluation_scope"] = _coalesce_series(out, ["evaluation_scope"], default=PREFERRED_EVALUATION_SCOPE)
    out["evaluation_scope"] = out["evaluation_scope"].fillna(PREFERRED_EVALUATION_SCOPE).astype(str).str.strip().str.lower()

    out["source_platform"] = _coalesce_series(out, ["source_platform"], default=np.nan)
    out["source_platform"] = out["source_platform"].where(out["source_platform"].notna(), out["platform"])
    out["source_platform"] = out["source_platform"].astype(str).str.strip().str.lower()

    out["target_platform"] = _coalesce_series(out, ["target_platform"], default=np.nan)
    out["target_platform"] = out["target_platform"].where(out["target_platform"].notna(), out["platform"])
    out["target_platform"] = out["target_platform"].astype(str).str.strip().str.lower()

    out["sample_idx"] = _to_int_series(_coalesce_series(out, ["sample_idx"]))
    out["text"] = _coalesce_series(out, ["text"], default="").fillna("").astype(str)
    out["y_true"] = _to_int_series(_coalesce_series(out, ["y_true"]))
    out["y_pred"] = _to_int_series(_coalesce_series(out, ["y_pred"]))

    out["model_identifier"] = _coalesce_series(out, ["classifier", "model", "model_name"], default=np.nan)
    out["model_identifier"] = out["model_identifier"].astype(str).str.strip().str.lower()
    out["model_name"] = _coalesce_series(out, ["model_name", "classifier", "model"], default=np.nan)
    out["model_name"] = out["model_name"].astype(str).str.strip().str.lower()
    out["config_name"] = _coalesce_series(out, ["config_name"], default=np.nan)
    out["config_name"] = out["config_name"].where(out["config_name"].notna(), out["model_name"])
    out["config_name"] = out["config_name"].astype(str).str.strip().str.lower()

    out["model_family"] = _coalesce_series(out, ["model_family"], default=infer_family_from_frame(out, source_path))
    out["model_family"] = out["model_family"].map(normalize_model_family)

    out["preprocessing"] = _coalesce_series(out, ["preprocessing"], default=np.nan).map(normalize_preprocessing)
    out["max_len"] = pd.to_numeric(_coalesce_series(out, ["max_len"]), errors="coerce")

    out["embedding_family"] = _coalesce_series(out, ["embedding_family"], default=np.nan)
    out["embedding_family"] = out["embedding_family"].astype(str).str.strip().str.lower()
    out["embedding_name"] = _coalesce_series(out, ["embedding_name"], default=np.nan)
    out["embedding_name"] = out["embedding_name"].astype(str).str.strip().str.lower()
    out["representation_type"] = _coalesce_series(out, ["representation_type"], default=np.nan)
    out["representation_type"] = out["representation_type"].astype(str).str.strip().str.lower()
    out["dialect"] = _coalesce_series(out, ["dialect"], default=np.nan)
    out["dialect"] = out["dialect"].astype(str).str.strip()

    out["field"] = _coalesce_series(out, ["field"], default=np.nan)
    out["sector"] = _coalesce_series(out, ["sector", "field"], default=np.nan)
    out["sector"] = out["sector"].astype(str).str.strip()

    deception_col = resolve_deception_type_column(out)
    if deception_col is None:
        out["deception_type_raw"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    else:
        out["deception_type_raw"] = out[deception_col]
    out["deception_type"] = out["deception_type_raw"].map(normalize_deception_type)

    out["correct"] = _to_int_series(_coalesce_series(out, ["correct"]))
    computed_correct = (out["y_true"] == out["y_pred"]).astype("Int64")
    out["correct"] = out["correct"].where(out["correct"].notna(), computed_correct)

    out["error"] = _to_int_series(_coalesce_series(out, ["error"]))
    computed_error = (out["y_true"] != out["y_pred"]).astype("Int64")
    out["error"] = out["error"].where(out["error"].notna(), computed_error)

    out["input_file"] = source_path.relative_to(project_root).as_posix()
    out["input_format"] = source_path.suffix.lower().lstrip(".")

    standardized = out.reindex(columns=STANDARDIZED_COLUMNS).copy()
    validate_standardized_prediction_frame(standardized, source_path=source_path)
    return standardized


def validate_standardized_prediction_frame(df: pd.DataFrame, *, source_path: Path | None = None) -> None:
    missing = sorted(REQUIRED_STANDARDIZED_COLUMNS.difference(df.columns))
    if missing:
        location = f" in {source_path}" if source_path is not None else ""
        raise ValueError(f"Standardized frame is missing required columns{location}: {missing}")
    empty_required = [column for column in REQUIRED_STANDARDIZED_COLUMNS if column in df.columns and df[column].isna().all()]
    if empty_required:
        location = f" in {source_path}" if source_path is not None else ""
        raise ValueError(f"Required standardized columns are fully empty{location}: {empty_required}")


def load_standardized_prediction_exports(project_root: Path, candidate_inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = candidate_inventory[candidate_inventory["selected_for_loading"]].copy()
    if selected.empty:
        raise FileNotFoundError("No item-level misclassification exports were selected for loading.")

    frames: list[pd.DataFrame] = []
    loaded_rows: list[dict] = []
    for record in selected.itertuples(index=False):
        source_path = project_root / record.relative_path
        LOGGER.info("Loading misclassification export: %s", source_path)
        raw_df = read_tabular_file(source_path)
        standardized = standardize_prediction_frame(raw_df, source_path=source_path, project_root=project_root)
        family_label = infer_family_from_frame(raw_df, source_path)
        frames.append(standardized)
        loaded_rows.append(
            {
                "relative_path": record.relative_path,
                "loaded_row_count": int(len(standardized)),
                "loaded_platforms": ", ".join(sorted(standardized["platform"].dropna().astype(str).unique())),
                "loaded_model_families": ", ".join(sorted(standardized["model_family"].dropna().astype(str).unique())),
                "loaded_evaluation_scopes": ", ".join(sorted(standardized["evaluation_scope"].dropna().astype(str).unique())),
                "loaded_preprocessing": ", ".join(ordered_preprocessing_values(standardized["preprocessing"])),
                "detected_model_family": family_label,
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    validate_standardized_prediction_frame(combined)
    loaded_df = pd.DataFrame(loaded_rows)
    inventory_df = candidate_inventory.merge(loaded_df, on="relative_path", how="left")
    inventory_df["loaded_row_count"] = pd.to_numeric(inventory_df["loaded_row_count"], errors="coerce").astype("Int64")
    return combined, inventory_df


def select_plot_evaluation_scope(standardized_df: pd.DataFrame) -> str:
    scopes = [scope for scope in standardized_df["evaluation_scope"].dropna().astype(str).unique() if scope]
    if PREFERRED_EVALUATION_SCOPE in scopes:
        return PREFERRED_EVALUATION_SCOPE
    if len(scopes) == 1:
        LOGGER.warning(
            "Preferred evaluation scope '%s' was not found; using only available scope '%s'.",
            PREFERRED_EVALUATION_SCOPE,
            scopes[0],
        )
        return scopes[0]
    raise ValueError(f"Unable to resolve a single evaluation scope for plotting from: {scopes}")


def load_experiment_results(platform: str, project_root: Path) -> tuple[pd.DataFrame, Path]:
    tables_dir = project_root / "Output" / platform / "tables"
    candidates = [
        tables_dir / "experiment_results_long.csv",
        tables_dir / "eval_summary_all_metrics_long.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            results_df = pd.read_csv(candidate)
            results_df = ensure_evaluation_scope(results_df)
            if "platform" not in results_df.columns:
                results_df["platform"] = platform
            results_df["platform"] = results_df["platform"].fillna(platform).astype(str).str.strip().str.lower()
            if "model_family" in results_df.columns:
                results_df["model_family"] = results_df["model_family"].map(normalize_model_family)
            if "preprocessing" in results_df.columns:
                results_df["preprocessing"] = results_df["preprocessing"].map(normalize_preprocessing)
            if "model_name" in results_df.columns:
                results_df["model_name"] = results_df["model_name"].astype(str).str.strip().str.lower()
            if "embedding_name" in results_df.columns:
                results_df["embedding_name"] = results_df["embedding_name"].astype(str).str.strip().str.lower()
            if "representation_type" in results_df.columns:
                results_df["representation_type"] = results_df["representation_type"].astype(str).str.strip().str.lower()
            if "max_len" in results_df.columns:
                results_df["max_len"] = pd.to_numeric(results_df["max_len"], errors="coerce")
            return results_df, candidate
    raise FileNotFoundError(
        f"No experiment results table was found for platform '{platform}' under {tables_dir}."
    )


def build_exact_model_selection(results_df: pd.DataFrame, *, platform: str, preprocessing_order: list[str]) -> pd.DataFrame:
    roster_df, _ = build_exact_model_comparison_roster(
        results_df,
        preprocessing_order=preprocessing_order,
        pipeline_name=platform,
    )
    if roster_df.empty:
        raise ValueError(f"Exact-model selection is empty for platform '{platform}'.")
    return roster_df.sort_values(["preprocessing", "plot_order"]).reset_index(drop=True)


def exact_model_label(model_family: str, model_name: str, embedding_name: str) -> str:
    model_label = MODEL_SHORT.get(_norm(model_name), str(model_name))
    if normalize_model_family(model_family) == "Transformer":
        return model_label
    embedding_label = EMBED_SHORT.get(_norm(embedding_name), str(embedding_name))
    return f"{model_label}-{embedding_label}"


def selection_rule_text() -> str:
    return (
        "Within each preprocessing, the best in-domain configuration for each exact study-roster item is selected "
        "by Macro-F1, then accuracy, then config_uid."
    )


def build_plot_summary(
    standardized_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    *,
    evaluation_scope: str,
) -> pd.DataFrame:
    if standardized_df.empty or selection_df.empty:
        return pd.DataFrame(columns=PLOT_SUMMARY_COLUMNS)

    working = attach_config_uid(standardized_df.copy())
    working = working[working["evaluation_scope"].astype(str) == str(evaluation_scope)].copy()
    working = working[working["deception_type"].astype(str).ne("")].copy()

    selection = ensure_evaluation_scope(selection_df.copy())
    selection = selection[selection["evaluation_scope"].astype(str) == str(evaluation_scope)].copy()
    selection = selection.drop_duplicates("config_uid")

    if working.empty or selection.empty:
        return pd.DataFrame(columns=PLOT_SUMMARY_COLUMNS)

    working = working[working["config_uid"].isin(selection["config_uid"])].copy()
    if working.empty:
        return pd.DataFrame(columns=PLOT_SUMMARY_COLUMNS)

    working["false_positive_count"] = (
        (working["y_true"] == NEGATIVE_CLASS_CODE) & (working["y_pred"] == POSITIVE_CLASS_CODE)
    ).astype(int)
    working["false_negative_count"] = (
        (working["y_true"] == POSITIVE_CLASS_CODE) & (working["y_pred"] == NEGATIVE_CLASS_CODE)
    ).astype(int)

    group_cols = [
        "platform",
        "evaluation_scope",
        "source_platform",
        "target_platform",
        "preprocessing",
        "model_family",
        "model_name",
        "model_identifier",
        "config_name",
        "embedding_family",
        "embedding_name",
        "representation_type",
        "max_len",
        "config_uid",
        "deception_type",
    ]
    grouped = (
        working.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            total_samples=("sample_idx", "count"),
            error_count=("error", "sum"),
            false_positive_count=("false_positive_count", "sum"),
            false_negative_count=("false_negative_count", "sum"),
        )
    )
    grouped["error_rate"] = grouped["error_count"] / grouped["total_samples"]
    grouped["selection_rule"] = selection_rule_text()

    merged = grouped.merge(
        selection[
            [
                "config_uid",
                "exact_label",
                "tick_label",
                "plot_order",
                "selection_note",
            ]
        ].drop_duplicates("config_uid"),
        on="config_uid",
        how="left",
    )
    merged["exact_label"] = merged["exact_label"].where(
        merged["exact_label"].notna(),
        merged.apply(
            lambda row: exact_model_label(row["model_family"], row["model_name"], row["embedding_name"]),
            axis=1,
        ),
    )
    merged["plot_order"] = pd.to_numeric(merged["plot_order"], errors="coerce")
    merged = merged[PLOT_SUMMARY_COLUMNS].sort_values(
        ["platform", "preprocessing", "plot_order", "deception_type"],
        ascending=[True, True, True, True],
    )
    return merged.reset_index(drop=True)


def build_series_color_map(selection_df: pd.DataFrame) -> dict[str, str]:
    color_map: dict[str, str] = {}
    for family_label in FAMILY_ORDER:
        family_selection = (
            selection_df[selection_df["model_family"].astype(str) == str(family_label)]
            .sort_values(["plot_order", "model_name", "embedding_name", "config_name"])
            .drop_duplicates("exact_label")
        )
        palette = FAMILY_COLOR_MAP.get(family_label, ["#666666"])
        for idx, row in enumerate(family_selection.itertuples(index=False)):
            color_map[str(row.exact_label)] = palette[idx % len(palette)]
    return color_map


def _wrap_label(label: str, width: int = 13) -> str:
    wrapped = textwrap.wrap(str(label), width=width, break_long_words=False, break_on_hyphens=False)
    return "\n".join(wrapped) if wrapped else str(label)


def _short_deception_label(label: str) -> str:
    normalized = normalize_deception_type(label)
    if normalized in DECEPTIVE_TYPE_SHORT_LABELS:
        return DECEPTIVE_TYPE_SHORT_LABELS[normalized]
    return _wrap_label(str(label).replace("_", " ").title(), width=13)


def save_figure_bundle(fig: plt.Figure, path_stem: Path, *, dpi: int = 400) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")


def write_dataframe_with_markdown(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    if df.empty:
        md_text = "_No rows available._\n"
    else:
        try:
            md_text = df.to_markdown(index=False) + "\n"
        except Exception:
            md_text = "```text\n" + df.to_string(index=False) + "\n```\n"
    md_path.write_text(md_text, encoding="utf-8")


def plot_platform_deceptive_type_error_rate_panels(
    plot_df: pd.DataFrame,
    *,
    selection_df: pd.DataFrame,
    platform: str,
    preprocessing_order: list[str],
    evaluation_scope: str,
    png_pdf_stem: Path,
) -> dict[str, float | int | str]:
    if plot_df.empty:
        raise ValueError("No plot rows were supplied for stacked deceptive-type figure generation.")

    ordered_types = ordered_deception_type_values(plot_df["deception_type"])
    series_df = (
        selection_df[
            [
                "model_family",
                "model_name",
                "embedding_name",
                "exact_label",
                "plot_order",
            ]
        ]
        .sort_values(["plot_order", "model_family", "model_name", "embedding_name"])
        .drop_duplicates("exact_label")
        .reset_index(drop=True)
    )
    color_map = build_series_color_map(selection_df)

    n_types = len(ordered_types)
    n_series = len(series_df)
    active_preprocessing = [prep for prep in preprocessing_order if prep in set(plot_df["preprocessing"].astype(str))]
    if n_types == 0 or n_series == 0 or not active_preprocessing:
        raise ValueError("Stacked deceptive-type figure requires deceptive types, model series, and preprocessing panels.")

    group_width = 0.84
    bar_width = min(0.18, group_width / max(n_series, 1))
    figure_width = min(16.6, max(13.6, 5.8 + 0.76 * n_types + 0.23 * n_series))
    figure_height = max(11.1, 3.45 * len(active_preprocessing) + 1.2)
    legend_cols = 6 if n_series >= 16 else 5 if n_series >= 12 else 4 if n_series >= 8 else max(1, n_series)
    legend_rows = int(math.ceil(n_series / legend_cols))
    legend_top = 0.988
    legend_bottom = legend_top - 0.033 * legend_rows
    title_y = legend_bottom - 0.011
    axes_top = max(0.73, title_y - 0.03)

    x_positions = np.arange(n_types, dtype=float)
    legend_handles = [
        Patch(
            facecolor=color_map.get(str(row.exact_label), "#666666"),
            edgecolor="white",
            linewidth=0.35,
            label=str(row.exact_label),
        )
        for row in series_df.itertuples(index=False)
    ]
    legend_labels = [str(row.exact_label) for row in series_df.itertuples(index=False)]

    global_max = float(plot_df["error_rate"].max()) if not plot_df.empty else 0.0
    y_limit = max(0.05, min(1.0, global_max * 1.12 if global_max > 0 else 0.05))

    with plt.rc_context(PAPER_RCPARAMS):
        fig, axes = plt.subplots(
            nrows=len(active_preprocessing),
            ncols=1,
            figsize=(figure_width, figure_height),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        for axis_idx, (ax, preprocessing) in enumerate(zip(axes, active_preprocessing)):
            prep_df = plot_df[plot_df["preprocessing"].astype(str) == str(preprocessing)].copy()
            for series_idx, row in enumerate(series_df.itertuples(index=False)):
                offset = -group_width / 2 + (series_idx + 0.5) * bar_width
                ordered = (
                    prep_df[prep_df["exact_label"].astype(str) == str(row.exact_label)]
                    .set_index("deception_type")
                    .reindex(ordered_types)
                )
                ax.bar(
                    x_positions + offset,
                    ordered["error_rate"].fillna(0.0).to_numpy(dtype=float),
                    width=bar_width * 0.94,
                    color=color_map.get(str(row.exact_label), "#666666"),
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.94,
                    zorder=3,
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.05)
            ax.spines["bottom"].set_linewidth(1.05)
            ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25, color="#8C96A3", zorder=1)
            ax.set_axisbelow(True)
            ax.set_ylim(0, y_limit)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.margins(x=0.02)
            ax.set_title(
                preprocessing_display_name(preprocessing),
                loc="left",
                fontweight="semibold",
                pad=8,
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [_short_deception_label(label) for label in ordered_types],
                rotation=0,
                ha="center",
            )
            ax.tick_params(axis="x", labelbottom=True, pad=2)
            ax.set_xlabel("Deceptive type", fontweight="semibold", labelpad=4)

        fig.supylabel("Error rate", x=0.03, fontweight="semibold")
        fig.legend(
            legend_handles,
            legend_labels,
            ncol=legend_cols,
            loc="upper right",
            bbox_to_anchor=(0.985, legend_top),
            frameon=False,
            columnspacing=0.95,
            handlelength=1.2,
            handletextpad=0.35,
            borderaxespad=0.0,
            labelspacing=0.35,
        )
        fig.suptitle(platform_display_name(platform), y=title_y, fontweight="semibold")
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=axes_top, hspace=0.38)
        save_figure_bundle(fig, png_pdf_stem)
        plt.close(fig)

    return {
        "platform": platform,
        "evaluation_scope": evaluation_scope,
        "n_deceptive_types": n_types,
        "n_series": n_series,
        "n_preprocessing_panels": len(active_preprocessing),
        "figure_width": figure_width,
        "figure_height": figure_height,
        "bar_width": float(bar_width),
        "legend_cols": legend_cols,
        "legend_rows": legend_rows,
    }


def plot_platform_deceptive_type_error_count_panels(
    plot_df: pd.DataFrame,
    *,
    selection_df: pd.DataFrame,
    platform: str,
    preprocessing_order: list[str],
    evaluation_scope: str,
    png_pdf_stem: Path,
) -> dict[str, float | int | str]:
    # Backward-compatible wrapper for the superseded count-based plotting entry point.
    return plot_platform_deceptive_type_error_rate_panels(
        plot_df,
        selection_df=selection_df,
        platform=platform,
        preprocessing_order=preprocessing_order,
        evaluation_scope=evaluation_scope,
        png_pdf_stem=png_pdf_stem,
    )


def compute_fp_fn_summary(standardized_df: pd.DataFrame, *, include_deception_type: bool) -> pd.DataFrame:
    if standardized_df.empty:
        columns = FP_FN_GROUP_COLUMNS + (["deception_type"] if include_deception_type else [])
        return pd.DataFrame(
            columns=columns
            + [
                "positive_class_code",
                "positive_class_label",
                "negative_class_code",
                "negative_class_label",
                "false_positive_count",
                "false_negative_count",
                "total_errors",
                "total_samples",
                "fp_rate",
                "fn_rate",
                "overall_error_rate",
            ]
        )

    working = standardized_df.copy()
    if include_deception_type:
        working = working[working["deception_type"].astype(str).ne("")].copy()
    working["false_positive_count"] = (
        (working["y_true"] == NEGATIVE_CLASS_CODE) & (working["y_pred"] == POSITIVE_CLASS_CODE)
    ).astype(int)
    working["false_negative_count"] = (
        (working["y_true"] == POSITIVE_CLASS_CODE) & (working["y_pred"] == NEGATIVE_CLASS_CODE)
    ).astype(int)

    group_cols = FP_FN_GROUP_COLUMNS.copy()
    if include_deception_type:
        group_cols.append("deception_type")

    summary = (
        working.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            false_positive_count=("false_positive_count", "sum"),
            false_negative_count=("false_negative_count", "sum"),
            total_errors=("error", "sum"),
            total_samples=("sample_idx", "count"),
        )
    )
    summary["fp_rate"] = summary["false_positive_count"] / summary["total_samples"]
    summary["fn_rate"] = summary["false_negative_count"] / summary["total_samples"]
    summary["overall_error_rate"] = summary["total_errors"] / summary["total_samples"]
    summary.insert(len(group_cols), "positive_class_code", POSITIVE_CLASS_CODE)
    summary.insert(len(group_cols) + 1, "positive_class_label", POSITIVE_CLASS_LABEL)
    summary.insert(len(group_cols) + 2, "negative_class_code", NEGATIVE_CLASS_CODE)
    summary.insert(len(group_cols) + 3, "negative_class_label", NEGATIVE_CLASS_LABEL)

    ordered_columns = (
        group_cols
        + [
            "positive_class_code",
            "positive_class_label",
            "negative_class_code",
            "negative_class_label",
            "false_positive_count",
            "false_negative_count",
            "total_errors",
            "total_samples",
            "fp_rate",
            "fn_rate",
            "overall_error_rate",
        ]
    )
    summary = summary[ordered_columns].copy()

    sort_cols = [
        column
        for column in [
            "platform",
            "evaluation_scope",
            "source_platform",
            "target_platform",
            "preprocessing",
            "model_family",
            "model_name",
            "embedding_name",
            "config_name",
        ]
        if column in summary.columns
    ]
    if include_deception_type:
        sort_cols.append("deception_type")
    summary = summary.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return summary


def build_artifact_manifest(manifest_rows: list[dict]) -> pd.DataFrame:
    columns = [
        "artifact_type",
        "format",
        "filename",
        "relative_path",
        "platform",
        "preprocessing",
        "evaluation_scope",
        "description",
    ]
    if not manifest_rows:
        return pd.DataFrame(columns=columns)
    manifest = pd.DataFrame(manifest_rows)
    for column in columns:
        if column not in manifest.columns:
            manifest[column] = ""
    return manifest[columns].sort_values(["artifact_type", "platform", "preprocessing", "filename"]).reset_index(drop=True)
