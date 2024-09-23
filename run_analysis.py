import json
from typing import Any
from pathlib import Path

import click

from analysis.shared import batch_convert, create_aggregated_df, create_aggregated_selection_df
from analysis.sub_eval_comparison import compare_sub_evals
from analysis.sub_eval_plotting import plot_sub_evals
from analysis.selection_plotting import plot_selection


def run_compare_sub_evals(tests_dir: Path, encoding_cfg: dict[str, Any]):
    for seed_dir in tests_dir.glob("[0-9]*"):
        compare_sub_evals(encoding_cfg, seed_dir)


@click.command()
@click.argument(
    "analysis_type",
    type=click.Choice(
        [
            "convert_tests",
            "aggregate",
            "compare_sub_evals",
            "plot_sub_evals",
            "save_combined_sub_evals",
            "create_selection_df",
            "plot_selection",
        ]
    ),
)
@click.argument("encoding_cfg_fpath", type=click.Path(exists=True), nargs=1)
@click.argument("tests_dir_str", type=click.Path(exists=True))
def main(analysis_type: str, encoding_cfg_fpath: str, tests_dir_str: str) -> None:
    with open(encoding_cfg_fpath, encoding="utf-8") as j_file:
        encoding_cfg = json.load(j_file)

    tests_dir = Path(tests_dir_str)
    match analysis_type:
        case "convert_tests":
            batch_convert(tests_dir, encoding_cfg)
        case "aggregate":
            aggregated_df = create_aggregated_df(tests_dir, encoding_cfg)
            aggregated_df.to_pickle(tests_dir / "aggregated.pkl")
        case "compare_sub_evals":
            run_compare_sub_evals(tests_dir, encoding_cfg)
        case "plot_sub_evals":
            plot_sub_evals(tests_dir)
        case "create_selection_df":
            selection_df = create_aggregated_selection_df(tests_dir)
            selection_df.to_pickle(tests_dir / "selection.pkl")
        case "plot_selection":
            plot_selection(tests_dir)
        case _:
            raise ValueError(f"analysis type unknown: {analysis_type}")


if __name__ == "__main__":
    main()

#  LocalWords:  eval fpath str evals
