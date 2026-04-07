from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exact_model_preprocessing_figures_helpers import (
    EMBED_SHORT,
    FAMILY_ORDER,
    MODEL_SHORT,
    build_exact_model_comparison_roster,
    ensure_evaluation_scope,
)


PREPROCESSING_KEEP = {"original", "manual", "deepfake_aware"}
DECEPTIVE_TYPE_ORDER = [
    "original",
    "clickbait phrasing",
    "contradiction",
    "exaggeration",
    "mixed truths",
    "omission",
    "satirical tone",
]
DECEPTIVE_TYPE_NORMALIZATION = {
    "none": "original",
    "orginal": "original",
}
FAMILY_COLOR_MAP = {
    "ML": ["#C44E52", "#DD8452", "#E17C05", "#B07AA1", "#937860", "#DA8BC3", "#8C8C8C", "#F28E2B", "#BAB0AC"],
    "Sequence": ["#55A868", "#59A14F", "#7BC47F", "#2F7D4A", "#8CD17D", "#76B7B2"],
    "Transformer": ["#4C72B0", "#64B5CD", "#8172B2"],
}


def _norm(value) -> str:
    return str(value).strip().lower()


def _normalize_deceptive_type(value) -> str:
    cleaned = " ".join(str(value).strip().lower().replace("_", " ").split())
    return DECEPTIVE_TYPE_NORMALIZATION.get(cleaned, cleaned)


def resolve_deception_type_column(df: pd.DataFrame) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in ["deception_type", "deceptive_type", "deception category", "deceptive category"]:
        if candidate in normalized:
            return normalized[candidate]
    for col in df.columns:
        col_key = str(col).strip().lower()
        if ("deception" in col_key or "deceptive" in col_key) and "type" in col_key:
            return col
    return None


def _series_sort_key(model_family, model_name, embedding_name):
    return (
        FAMILY_ORDER.index(str(model_family)) if str(model_family) in FAMILY_ORDER else 99,
        str(model_name),
        str(embedding_name),
    )


def _series_label(model_family, model_name, embedding_name) -> str:
    model_label = MODEL_SHORT.get(_norm(model_name), str(model_name))
    embedding_label = EMBED_SHORT.get(_norm(embedding_name), str(embedding_name))
    if str(model_family) == "Transformer":
        return model_label
    return f"{model_label}-{embedding_label}"


def _wrap_axis_label(label: str, width: int = 14) -> str:
    wrapped = textwrap.wrap(str(label), width=width, break_long_words=False, break_on_hyphens=False)
    return "\n".join(wrapped) if wrapped else str(label)


def _base_columns():
    return [
        "platform",
        "preprocessing",
        "model_family",
        "model_name",
        "config_name",
        "embedding_name",
        "deceptive_type",
        "total_samples",
        "error_count",
        "error_rate",
        "false_positive_count",
        "false_negative_count",
        "config_uid",
        "deception_type_field",
        "selection_rule",
        "selection_note",
        "plot_order",
        "exact_label",
        "tick_label",
    ]


def _selection_rule_text() -> str:
    return (
        "Within each preprocessing, the best in-domain configuration for each exact study-roster item is selected "
        "by Macro-F1, then accuracy, then config_uid."
    )


def _build_exact_model_selection(results_df: pd.DataFrame, *, preprocessing_order: list[str], pipeline_name: str) -> pd.DataFrame:
    roster_df, _ = build_exact_model_comparison_roster(
        results_df,
        preprocessing_order=preprocessing_order,
        pipeline_name=pipeline_name,
    )
    if roster_df.empty:
        return roster_df
    selection = roster_df.copy()
    selection["selection_rule"] = _selection_rule_text()
    return selection.sort_values(["preprocessing", "plot_order"]).reset_index(drop=True)


def _summarize(prediction_df: pd.DataFrame, selected_configs: pd.DataFrame, deception_col: str, attach_config_uid) -> pd.DataFrame:
    base_columns = _base_columns()
    if prediction_df.empty or selected_configs.empty or deception_col not in prediction_df.columns:
        return pd.DataFrame(columns=base_columns)

    working = attach_config_uid(prediction_df.copy())
    selection = ensure_evaluation_scope(selected_configs.copy())
    selection = selection[selection["evaluation_scope"] == "in_domain"].drop_duplicates("config_uid")
    if selection.empty:
        return pd.DataFrame(columns=base_columns)

    working = working[working["config_uid"].isin(selection["config_uid"])].copy()
    working = working.dropna(subset=[deception_col]).copy()
    if working.empty:
        return pd.DataFrame(columns=base_columns)

    working["deceptive_type"] = working[deception_col].map(_normalize_deceptive_type)
    working = working[working["deceptive_type"].ne("")].copy()
    if working.empty:
        return pd.DataFrame(columns=base_columns)

    # Positive class is class 1 = Real throughout the notebooks.
    working["false_positive_count"] = ((working["y_true"] == 0) & (working["y_pred"] == 1)).astype(int)
    working["false_negative_count"] = ((working["y_true"] == 1) & (working["y_pred"] == 0)).astype(int)

    grouped = (
        working.groupby(["config_uid", "deceptive_type"], dropna=False, as_index=False)
        .agg(
            total_samples=("sample_idx", "count"),
            error_count=("error", "sum"),
            false_positive_count=("false_positive_count", "sum"),
            false_negative_count=("false_negative_count", "sum"),
        )
    )
    grouped["error_rate"] = grouped["error_count"] / grouped["total_samples"]
    grouped["deception_type_field"] = deception_col

    merged = grouped.merge(
        selection[
            [
                "platform",
                "preprocessing",
                "model_family",
                "model_name",
                "config_name",
                "embedding_name",
                "config_uid",
                "selection_note",
                "plot_order",
                "exact_label",
                "tick_label",
                "selection_rule",
            ]
        ].drop_duplicates("config_uid"),
        on="config_uid",
        how="left",
    )
    merged["sort_key"] = merged.apply(lambda row: _series_sort_key(row["model_family"], row["model_name"], row["embedding_name"]), axis=1)
    merged = merged.sort_values(
        ["preprocessing", "plot_order", "sort_key", "deceptive_type"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return merged.drop(columns=["sort_key"])[base_columns]


def _ordered_deceptive_types(summary_df: pd.DataFrame) -> list[str]:
    observed = [value for value in summary_df["deceptive_type"].dropna().astype(str).tolist() if value]
    unique = list(dict.fromkeys(observed))
    preferred = [value for value in DECEPTIVE_TYPE_ORDER if value in unique]
    remainder = sorted(value for value in unique if value not in preferred)
    return preferred + remainder


def _family_color_map(family_df: pd.DataFrame) -> dict[str, str]:
    series_df = (
        family_df[["config_uid", "model_family", "model_name", "embedding_name"]]
        .drop_duplicates()
        .sort_values(["model_name", "embedding_name", "config_uid"])
        .reset_index(drop=True)
    )
    palette = FAMILY_COLOR_MAP.get(str(family_df["model_family"].iloc[0]), ["#666666"])
    return {row.config_uid: palette[idx % len(palette)] for idx, row in enumerate(series_df.itertuples(index=False))}


def _plot_preprocessing_facets(
    summary_df: pd.DataFrame,
    *,
    preprocessing: str,
    stem: str,
    fig_dir: Path,
    display_name: str,
    paper_preprocessing_label,
    paper_rcparams: dict,
    save_matplotlib_figure,
) -> None:
    plot_df = summary_df[summary_df["preprocessing"].astype(str) == str(preprocessing)].copy()
    if plot_df.empty:
        return

    deceptive_type_order = _ordered_deceptive_types(plot_df)
    plot_df["deceptive_type"] = pd.Categorical(plot_df["deceptive_type"], deceptive_type_order, ordered=True)
    figure_width = max(12.0, 1.05 * len(deceptive_type_order) + 6.2)

    with plt.rc_context(paper_rcparams):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(figure_width, 10.8), sharex=True)
        axes = np.atleast_1d(axes)

        for axis_idx, family_label in enumerate(FAMILY_ORDER):
            ax = axes[axis_idx]
            family_df = plot_df[plot_df["model_family"].astype(str) == family_label].copy()
            if family_df.empty:
                ax.axis("off")
                continue

            series_df = (
                family_df[["config_uid", "model_name", "embedding_name", "tick_label"]]
                .drop_duplicates()
                .sort_values(["tick_label", "config_uid"])
                .reset_index(drop=True)
            )
            color_map = _family_color_map(family_df)
            bar_width = min(0.84 / max(len(series_df), 1), 0.2)
            x_positions = np.arange(len(deceptive_type_order), dtype=float)

            for idx, row in enumerate(series_df.itertuples(index=False)):
                offset = (idx - (len(series_df) - 1) / 2) * bar_width
                ordered = (
                    family_df[family_df["config_uid"] == row.config_uid]
                    .set_index("deceptive_type")
                    .reindex(deceptive_type_order)
                )
                ax.bar(
                    x_positions + offset,
                    ordered["error_rate"].to_numpy(dtype=float),
                    width=bar_width * 0.94,
                    color=color_map.get(row.config_uid, "#666666"),
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.92,
                    label=_series_label(family_label, row.model_name, row.embedding_name),
                    zorder=3,
                )

            ax.set_ylabel("Error rate")
            ax.set_title(f"{family_label} Models", fontweight="semibold", fontsize=10.5, pad=8)
            ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25, color="#8C96A3", zorder=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(x=0.02)
            y_max = float(family_df["error_rate"].max()) if not family_df.empty else 0.0
            ax.set_ylim(0.0, max(0.08, y_max * 1.18 if y_max > 0 else 0.08))
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ncol = 3 if len(handles) >= 6 else max(1, len(handles))
                ax.legend(
                    handles,
                    labels,
                    loc="lower left",
                    bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
                    ncol=ncol,
                    mode="expand",
                    frameon=False,
                    fontsize=8.2,
                    columnspacing=0.9,
                    handlelength=1.2,
                    handletextpad=0.4,
                    borderaxespad=0.0,
                )

        axes[-1].set_xticks(np.arange(len(deceptive_type_order), dtype=float))
        axes[-1].set_xticklabels([_wrap_axis_label(label, width=14) for label in deceptive_type_order], rotation=0, ha="center")
        axes[-1].set_xlabel("Deceptive type")
        fig.suptitle(
            f"{display_name}: {paper_preprocessing_label(preprocessing)} deceptive-type error rate",
            fontsize=12,
            fontweight="semibold",
            y=0.995,
        )
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.985))
        save_matplotlib_figure(fig, fig_dir / stem)
        plt.show()


def run_deceptive_type_analysis(
    *,
    experiment_results_long: pd.DataFrame,
    in_domain_predictions: pd.DataFrame,
    df_trans: pd.DataFrame,
    df_sequence: pd.DataFrame,
    df_ml: pd.DataFrame,
    attach_config_uid,
    write_dataframe_with_markdown,
    table_dir: Path,
    fig_dir: Path,
    pipeline_name: str,
    display_name: str,
    preprocessing_order: list[str],
    paper_rcparams: dict,
    paper_preprocessing_label,
    save_matplotlib_figure,
) -> dict:
    kept_preprocessing = [prep for prep in preprocessing_order if prep in PREPROCESSING_KEEP]
    results_long = ensure_evaluation_scope(attach_config_uid(experiment_results_long.copy()))
    results_long = results_long[results_long["evaluation_scope"] == "in_domain"].copy()
    deception_col = resolve_deception_type_column(in_domain_predictions)
    artifacts: list[str] = []

    empty = pd.DataFrame(columns=_base_columns())
    if deception_col is None:
        return {
            "deception_type_field": None,
            "exact_model_table": empty,
            "summary": pd.DataFrame(columns=_base_columns() + ["analysis_group"]),
            "manifest": pd.DataFrame(),
            "selection": pd.DataFrame(),
            "artifacts": artifacts,
        }

    selection = _build_exact_model_selection(results_long, preprocessing_order=kept_preprocessing, pipeline_name=pipeline_name)
    all_predictions = pd.concat(
        [frame for frame in [df_ml, df_sequence, df_trans] if frame is not None and not frame.empty],
        ignore_index=True,
        sort=False,
    ) if any(frame is not None and not frame.empty for frame in [df_ml, df_sequence, df_trans]) else pd.DataFrame()

    exact_model_table = _summarize(all_predictions, selection, deception_col, attach_config_uid)
    exact_model_path = table_dir / f"{pipeline_name}_deceptive_type_error_rate_exact_model_roster.csv"
    exact_model_table.to_csv(exact_model_path, index=False)

    selection_path = table_dir / f"{pipeline_name}_deceptive_type_error_rate_exact_model_selection.csv"
    selection.to_csv(selection_path, index=False)

    summary = exact_model_table.assign(analysis_group="exact_model_roster") if not exact_model_table.empty else pd.DataFrame(columns=_base_columns() + ["analysis_group"])
    write_dataframe_with_markdown(
        summary,
        table_dir / "deceptive_type_error_rate_summary.csv",
        table_dir / "deceptive_type_error_rate_summary.md",
    )

    manifest_rows = []
    for preprocessing in kept_preprocessing:
        prep_df = exact_model_table[exact_model_table["preprocessing"].astype(str) == str(preprocessing)].copy()
        if prep_df.empty:
            continue

        csv_name = f"{pipeline_name}_deceptive_type_error_rate_{preprocessing}_exact_model_roster.csv"
        csv_path = table_dir / csv_name
        prep_df.to_csv(csv_path, index=False)

        stem = f"{pipeline_name}_deceptive_type_error_rate_{preprocessing}_exact_model_roster"
        _plot_preprocessing_facets(
            prep_df,
            preprocessing=preprocessing,
            stem=stem,
            fig_dir=fig_dir,
            display_name=display_name,
            paper_preprocessing_label=paper_preprocessing_label,
            paper_rcparams=paper_rcparams,
            save_matplotlib_figure=save_matplotlib_figure,
        )
        manifest_rows.append(
            {
                "platform": pipeline_name,
                "preprocessing": preprocessing,
                "analysis_group": "exact_model_roster",
                "deception_type_field": deception_col,
                "figure_filename": f"{stem}.png",
                "figure_pdf_filename": f"{stem}.pdf",
                "csv_filename": csv_name,
                "figure_output_path": str(fig_dir / f"{stem}.png"),
                "csv_output_path": str(csv_path),
                "section_name": "Error Analysis / Deceptive-Type Error Rates",
                "purpose": "Faceted deceptive-type error-rate comparison over the full exact study roster.",
                "n_deceptive_types": int(prep_df["deceptive_type"].nunique()),
                "n_series": int(prep_df["config_uid"].nunique()),
                "selection_rule": _selection_rule_text(),
            }
        )
        artifacts.extend([csv_name, f"{stem}.png", f"{stem}.pdf"])

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(table_dir / "deceptive_type_error_rate_figures_manifest.csv", index=False)
    artifacts.extend(
        [
            "deceptive_type_error_rate_summary.csv",
            "deceptive_type_error_rate_summary.md",
            "deceptive_type_error_rate_figures_manifest.csv",
            exact_model_path.name,
            selection_path.name,
        ]
    )
    return {
        "deception_type_field": deception_col,
        "exact_model_table": exact_model_table,
        "summary": summary,
        "manifest": manifest,
        "selection": selection,
        "artifacts": artifacts,
    }
