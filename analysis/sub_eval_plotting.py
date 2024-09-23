from pathlib import Path
from itertools import product

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.shared import SEED_TO_SETTING


def convert_for_plotting(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    attitudes = (
        "non-factive",
        "factive",
        "contrafactive",
    )
    indicators = {"": "all", " (matching)": "matching", " (~matching)": "non-matching"}

    collected_dfs = []

    all_cols = [att + ind for att, ind in product(attitudes, indicators.keys())]

    aggregated_df["hyperparameter setting"] = aggregated_df["seed"].apply(
        lambda bs: SEED_TO_SETTING[bs]
    )

    for att, ind in product(attitudes, indicators.keys()):
        selected_col = att + ind

        to_drop = [other_att for other_att in all_cols if other_att != selected_col]
        to_drop.append("overall")

        sub_df = aggregated_df.drop(columns=to_drop)
        sub_df["attitude verb"] = att
        sub_df["selection"] = indicators[ind]
        sub_df = sub_df.rename(columns={selected_col: "proportion correct"})

        collected_dfs.append(sub_df)

    return pd.concat(collected_dfs)


def plot_sub_evals(tests_dir: Path) -> None:
    aggregated_df = pd.read_pickle(tests_dir / "aggregated.pkl")
    converted_df = convert_for_plotting(aggregated_df)

    analysis_dir = tests_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    sns.set_theme(style="darkgrid")
    for selector in converted_df.selection.unique():
        selected_df = converted_df[converted_df.selection == selector]

        fg = sns.relplot(
            x="batch",
            y="proportion correct",
            hue="attitude verb",
            hue_order=["factive", "contrafactive", "non-factive"],
            data=selected_df,
            col="hyperparameter setting",
            kind="line",
        )
        sns.move_legend(
            fg,
            "lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=3,
            title=None,
            frameon=False,
        )

        fg.savefig(analysis_dir / f"sub_eval_plot_{selector}_correct.svg")

    selected_df = converted_df[(converted_df["hyperparameter setting"] == 1)]
    fg = sns.relplot(
        x="batch",
        y="proportion correct",
        hue="attitude verb",
        hue_order=["factive", "contrafactive", "non-factive"],
        data=selected_df,
        col="selection",
        kind="line",
    )
    sns.move_legend(
        fg,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        title=None,
        frameon=False,
    )

    fg.savefig(analysis_dir / "sub_eval_plot_setting_1_correct.svg")


#  LocalWords:  darkgrid eval pkl
