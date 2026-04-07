from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from misclassification_analysis_plot_helpers import (
    build_artifact_manifest,
    build_exact_model_selection,
    build_plot_summary,
    compute_fp_fn_summary,
    inspect_misclassification_tabular_files,
    load_experiment_results,
    load_standardized_prediction_exports,
    ordered_preprocessing_values,
    platform_display_name,
    plot_platform_deceptive_type_error_rate_panels,
    select_plot_evaluation_scope,
    write_dataframe_with_markdown,
)


LOGGER = logging.getLogger("misclassification_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate misclassification-based deceptive-type figures and FP/FN summary tables.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root that contains Output/, helper modules, and notebook exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for the new script-based analysis artifacts.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def manifest_row(
    *,
    artifact_type: str,
    fmt: str,
    path: Path,
    project_root: Path,
    platform: str,
    preprocessing: str,
    evaluation_scope: str,
    description: str,
) -> dict:
    return {
        "artifact_type": artifact_type,
        "format": fmt,
        "filename": path.name,
        "relative_path": path.relative_to(project_root).as_posix(),
        "platform": platform,
        "preprocessing": preprocessing,
        "evaluation_scope": evaluation_scope,
        "description": description,
    }


def print_run_summary(
    *,
    inventory_df: pd.DataFrame,
    standardized_df: pd.DataFrame,
    plot_scope: str,
    figure_count: int,
    tables_root: Path,
    figures_root: Path,
    manifest_path: Path,
) -> None:
    selected_files = inventory_df[inventory_df["selected_for_loading"]]["relative_path"].tolist()
    platforms = sorted(standardized_df["platform"].dropna().astype(str).unique())
    preprocessing_values = ordered_preprocessing_values(standardized_df["preprocessing"])
    scopes = sorted(standardized_df["evaluation_scope"].dropna().astype(str).unique())

    print("\nMisclassification analysis summary")
    print(f"- Verified raw misclassification exports loaded: {len(selected_files)}")
    for relative_path in selected_files:
        print(f"  - {relative_path}")
    print(f"- Platforms present: {', '.join(platforms)}")
    print(f"- Evaluation scopes present in raw exports: {', '.join(scopes)}")
    print(f"- Plot evaluation scope used: {plot_scope}")
    print(f"- Preprocessing approaches present: {', '.join(preprocessing_values)}")
    print("- Verified class convention: class 1 = Real, class 0 = Fake")
    print(f"- Logical merged figures generated: {figure_count}")
    print(f"- Tables directory: {tables_root}")
    print(f"- Figures directory: {figures_root}")
    print(f"- Artifact manifest: {manifest_path}")


def main() -> None:
    args = parse_args()
    configure_logging()

    project_root = args.project_root.resolve()
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (project_root / "Output" / "misclassification_analysis_artifacts").resolve()
    )
    tables_root = output_root / "tables"
    figures_root = output_root / "figures"
    tables_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []

    candidate_inventory = inspect_misclassification_tabular_files(project_root)
    standardized_df, inventory_df = load_standardized_prediction_exports(project_root, candidate_inventory)
    plot_scope = select_plot_evaluation_scope(standardized_df)

    inventory_path = tables_root / "misclassification_input_inventory.csv"
    inventory_df.to_csv(inventory_path, index=False, encoding="utf-8")
    manifest_rows.append(
        manifest_row(
            artifact_type="table",
            fmt="csv",
            path=inventory_path,
            project_root=project_root,
            platform="all",
            preprocessing="all",
            evaluation_scope="all",
            description="Verified inventory of tabular files discovered under Output/*/misclass_analysis and whether each file was loaded.",
        )
    )

    figure_count = 0
    platforms = sorted(standardized_df["platform"].dropna().astype(str).unique())
    for platform in platforms:
        platform_df = standardized_df[standardized_df["platform"].astype(str) == str(platform)].copy()
        platform_preprocessing = ordered_preprocessing_values(platform_df["preprocessing"])
        platform_tables_dir = tables_root / platform
        platform_figures_dir = figures_root / platform
        platform_tables_dir.mkdir(parents=True, exist_ok=True)
        platform_figures_dir.mkdir(parents=True, exist_ok=True)

        results_df, results_source = load_experiment_results(platform, project_root)
        LOGGER.info("Using experiment results source for %s: %s", platform, results_source)
        selection_df = build_exact_model_selection(
            results_df,
            platform=platform,
            preprocessing_order=platform_preprocessing,
        )

        selection_path = (
            platform_tables_dir
            / f"{platform}_deceptive_type_error_rate_all_preprocessing_stacked_exact_model_selection.csv"
        )
        selection_df.to_csv(selection_path, index=False, encoding="utf-8")
        manifest_rows.append(
            manifest_row(
                artifact_type="table",
                fmt="csv",
                path=selection_path,
                project_root=project_root,
                platform=platform,
                preprocessing="all",
                evaluation_scope=plot_scope,
                description=f"Exact-model roster selected for the stacked deceptive-type error-rate figure on {platform_display_name(platform)}.",
            )
        )

        plot_summary_df = build_plot_summary(
            platform_df,
            selection_df,
            evaluation_scope=plot_scope,
        )
        if plot_summary_df.empty:
            LOGGER.warning("No plot rows available for %s; skipping stacked figure.", platform)
            continue

        stem = f"{platform}_deceptive_type_error_rate_all_preprocessing_stacked_exact_models"
        plot_csv_path = platform_tables_dir / f"{stem}.csv"
        plot_summary_df.to_csv(plot_csv_path, index=False, encoding="utf-8")
        manifest_rows.append(
            manifest_row(
                artifact_type="table",
                fmt="csv",
                path=plot_csv_path,
                project_root=project_root,
                platform=platform,
                preprocessing="all",
                evaluation_scope=plot_scope,
                description=f"Plotted values for the vertically stacked deceptive-type error-rate figure on {platform_display_name(platform)} across all preprocessing approaches.",
            )
        )

        plot_platform_deceptive_type_error_rate_panels(
            plot_summary_df,
            selection_df=selection_df,
            platform=platform,
            preprocessing_order=platform_preprocessing,
            evaluation_scope=plot_scope,
            png_pdf_stem=platform_figures_dir / stem,
        )
        for fmt in ["png", "pdf"]:
            figure_path = (platform_figures_dir / stem).with_suffix(f".{fmt}")
            manifest_rows.append(
                manifest_row(
                    artifact_type="figure",
                    fmt=fmt,
                    path=figure_path,
                    project_root=project_root,
                    platform=platform,
                    preprocessing="all",
                    evaluation_scope=plot_scope,
                    description=f"Vertically stacked deceptive-type error-rate figure for {platform_display_name(platform)} with one shared top-right legend across all preprocessing approaches.",
                )
            )
        figure_count += 1

    overall_fp_fn = compute_fp_fn_summary(standardized_df, include_deception_type=False)
    overall_fp_fn_csv = tables_root / "all_platforms_fp_fn_summary.csv"
    overall_fp_fn_md = tables_root / "all_platforms_fp_fn_summary.md"
    write_dataframe_with_markdown(overall_fp_fn, overall_fp_fn_csv, overall_fp_fn_md)
    manifest_rows.extend(
        [
            manifest_row(
                artifact_type="table",
                fmt="csv",
                path=overall_fp_fn_csv,
                project_root=project_root,
                platform="all",
                preprocessing="all",
                evaluation_scope="all",
                description="Overall FP/FN summary across all platforms, preprocessors, models, embeddings, and model families.",
            ),
            manifest_row(
                artifact_type="table",
                fmt="md",
                path=overall_fp_fn_md,
                project_root=project_root,
                platform="all",
                preprocessing="all",
                evaluation_scope="all",
                description="Markdown companion for the overall FP/FN summary across all platforms.",
            ),
        ]
    )

    fp_fn_by_type = compute_fp_fn_summary(standardized_df, include_deception_type=True)
    fp_fn_by_type_csv = tables_root / "all_platforms_fp_fn_by_deceptive_type.csv"
    fp_fn_by_type_md = tables_root / "all_platforms_fp_fn_by_deceptive_type.md"
    write_dataframe_with_markdown(fp_fn_by_type, fp_fn_by_type_csv, fp_fn_by_type_md)
    manifest_rows.extend(
        [
            manifest_row(
                artifact_type="table",
                fmt="csv",
                path=fp_fn_by_type_csv,
                project_root=project_root,
                platform="all",
                preprocessing="all",
                evaluation_scope="all",
                description="FP/FN summary grouped by the normalized deception_type field in addition to the overall model configuration keys.",
            ),
            manifest_row(
                artifact_type="table",
                fmt="md",
                path=fp_fn_by_type_md,
                project_root=project_root,
                platform="all",
                preprocessing="all",
                evaluation_scope="all",
                description="Markdown companion for the deception-type FP/FN summary across all platforms.",
            ),
        ]
    )

    manifest_path = tables_root / "misclassification_analysis_artifact_manifest.csv"
    manifest_rows.append(
        manifest_row(
            artifact_type="table",
            fmt="csv",
            path=manifest_path,
            project_root=project_root,
            platform="all",
            preprocessing="all",
            evaluation_scope="all",
            description="Manifest of all figures and tables generated by the script-based misclassification analysis pipeline.",
        )
    )
    manifest_df = build_artifact_manifest(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")

    print_run_summary(
        inventory_df=inventory_df,
        standardized_df=standardized_df,
        plot_scope=plot_scope,
        figure_count=figure_count,
        tables_root=tables_root,
        figures_root=figures_root,
        manifest_path=manifest_path,
    )


if __name__ == "__main__":
    main()
