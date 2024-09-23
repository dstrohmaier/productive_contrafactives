from typing import Any
from pathlib import Path

import pandas as pd
import seaborn as sns

from analysis.shared import create_aggregated_selection_df, SEED_TO_SETTING


def plot_selection(tests_dir: Path) -> Any:
    selection_df: pd.DataFrame = create_aggregated_selection_df(tests_dir)

    melted_df = selection_df.melt(
        id_vars=["options", "seed", "batch", "sample size"],
        var_name="selection",
        value_name="proportion",
    )
    melted_df = melted_df.fillna(0.0)
    melted_df["batch size"] = melted_df["sample size"] / melted_df["batch"]

    melted_df["hyperparameter setting"] = melted_df["seed"].apply(
        lambda bs: SEED_TO_SETTING[bs]
    )

    sns.set_theme(style="darkgrid")
    fg = sns.catplot(
        data=melted_df[melted_df.batch >= 3000],
        x="selection",
        y="proportion",
        col="options",
        col_order=["factive-contrafactive", "factive", "contrafactive", "non-factive"],
        row="batch size",
        kind="box",
    )

    analysis_dir = tests_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    fg.savefig(analysis_dir / "selection_all.svg")

    fg = sns.catplot(
        data=melted_df[
            (melted_df.batch >= 3000) & (melted_df.options == "factive-contrafactive")
        ],
        x="selection",
        y="proportion",
        order=["factive", "contrafactive"],
        col="hyperparameter setting",
        kind="box",
    )

    analysis_dir = tests_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    fg.savefig(analysis_dir / "selection_multi_options.svg")
