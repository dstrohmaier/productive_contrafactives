import re

from pathlib import Path
from typing import Any
from functools import partial
from multiprocessing import Pool
from itertools import combinations

import numpy as np
import pandas as pd

from analysis.shared import process_test, get_batch_num


def compare_func(r: pd.Series, key_1: str, key_2: str):
    if r[key_1] > r[key_2]:
        return key_1
    elif r[key_1] < r[key_2]:
        return key_2
    else:
        return np.nan


def get_comparison(df: pd.DataFrame, cols: tuple[str, str, str]) -> pd.DataFrame:
    col_series = {}
    for k1, k2 in combinations(cols, 2):
        s = df.apply(lambda r: compare_func(r, k1, k2), axis=1).value_counts()
        col_series[f"{k1} -- {k2}"] = s
    compare_df = pd.DataFrame(col_series).transpose()
    return compare_df


def compare_sub_evals(encoding_cfg: dict[str, Any], test_output_dir: Path) -> None:
    sorted_paths = sorted(
        test_output_dir.glob("sub_eval/sub_test_*_converted.pkl"), key=get_batch_num
    )

    # print(sorted_paths)

    func = partial(process_test, encoding_cfg=encoding_cfg)
    with Pool(20) as p:
        rows = p.map(func, sorted_paths)
        # rows = list(map(func, sorted_paths))

    # print(rows[0])
    df = pd.DataFrame(rows)
    # df = df.loc[:, COLS]

    df.to_csv(test_output_dir / "batch_analysis.tsv", sep="\t", index=False)

    overall_df = get_comparison(df, ("non-factive", "factive", "contrafactive"))
    matching_df = get_comparison(
        df, ("non-factive (matching)", "factive (matching)", "contrafactive (matching)")
    )
    matching_df.columns = pd.Index([c[:-4] for c in matching_df.columns])
    non_matching_df = get_comparison(
        df,
        ("non-factive (~matching)", "factive (~matching)", "contrafactive (~matching)"),
    )
    non_matching_df.columns = pd.Index([col[:-5] for col in non_matching_df.columns])

    compare_df = pd.concat((overall_df, matching_df, non_matching_df))
    compare_df = compare_df.loc[:, ("non-factive", "factive", "contrafactive")]
    compare_df.to_csv(test_output_dir / "compare_attitudes_analysis.tsv", sep="\t")
