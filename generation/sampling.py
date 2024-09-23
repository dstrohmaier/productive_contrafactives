from pathlib import Path
from itertools import product

import click
import numpy as np
import pandas as pd


def balance(all_df: pd.DataFrame, base_size: int = 54):
    A = np.array([[-2, 18], [1, 6]])
    b = np.array([base_size, base_size])
    true_size, other_size = np.linalg.solve(A, b)

    main_vals = ("true", "false", "pfailure")
    sub_vals = ("true", "false", "unknown")
    mw = ("=", "!=", "?")

    sub_dfs = []

    for mw_r, m_val, s_val in product(mw, main_vals, sub_vals):
        size = round(other_size * 1000)

        if (mw_r == "?") and (m_val == "true") and (s_val == "unknown"):
            size = base_size * 1000
        elif m_val == "true":
            size = round(true_size * 1000)

        selected_df = all_df[
            (all_df.mind_world_relation == mw_r)
            & (all_df.main_value == m_val)
            & (all_df.sub_value == s_val)
        ]

        if len(selected_df) == 0:
            continue

        assert len(selected_df) >= size, f"too small size {len(selected_df)}"

        sub_dfs.append(selected_df.sample(size))

    return pd.concat(sub_dfs)


@click.command()
@click.argument("input_fpath", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=True))
@click.option("--seed", default=1848, type=int)
def sample_df(input_fpath: str, output_dir: str, seed: int) -> None:
    np.random.seed(seed)

    df = pd.read_csv(input_fpath, sep="\t", keep_default_na=False)

    balanced_df = balance(df)
    balanced_df.to_csv(Path(output_dir) / "balanced.tsv", sep="\t", index=False)


if __name__ == "__main__":
    sample_df()  # pylint: disable=no-value-for-parameter
