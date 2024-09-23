import re
import json

from typing import Any
from pathlib import Path
from functools import partial
from multiprocessing import Pool

from bidict import bidict
import pandas as pd


SEED_TO_SETTING = {
    9545076: 2,
    8433255: 1,
    7283613: 1,
    5463433: 2,
    4932797: 2,
    2724542: 1,
    1403899: 1,
    1116077: 2,
    445741: 2,
    81708: 1
}


def check_for_attitude_allowed(row, output_map, attitude: str):
    att_1 = output_map.inv[row.attitude_1]
    att_2 = output_map.inv.get(row.attitude_2, None)

    return (att_1 == attitude) or (att_2 == attitude)


def get_prop_correct(df: pd.DataFrame, selector: pd.Series) -> float:
    return (
        df[selector]
        .overall_correctness.value_counts(normalize=True)
        .get(True, 0)  # True? 0 ?
    )


def get_batch_num(filepath: Path) -> int:
    """Extract batch number from sub_test file path.

    :param filepath: file path from which to extract batch number.
    :returns: batch number.

    """
    extract_pattern = re.compile(r"sub_test_(\d+)_converted$")
    filename_stem = filepath.stem
    # print(filename_stem)

    match_object = extract_pattern.match(filename_stem)
    assert match_object, f"not matching {filename_stem} / {filepath}"
    batch_num = match_object.groups()[0]

    return int(batch_num)


def process_test(df: pd.DataFrame, encoding_cfg: dict[str, Any]) -> pd.DataFrame:
    # input_map = bidict(encoding_cfg["inupt_map"])
    output_map = bidict(encoding_cfg["output_map"])

    converted_df = df.apply(
        lambda r: pd.Series(
            [
                check_for_attitude_allowed(r, output_map, "non-factive"),
                check_for_attitude_allowed(r, output_map, "factive"),
                check_for_attitude_allowed(r, output_map, "contrafactive"),
                output_map.inv[int(r.output.split()[1])],
                r.matching,
                r.overall_correctness,
            ],
            index=[
                "non-factive allowed",
                "factive allowed",
                "contrafactive allowed",
                "selected",
                "matching",
                "overall_correctness",
            ],
        ),
        axis=1,
    )

    return converted_df


def converted_df_to_row(df: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, float | int] = {
        attitude: get_prop_correct(df, df[f"{attitude} allowed"])
        for attitude in ("non-factive", "factive", "contrafactive")
    }
    for attitude in ("non-factive", "factive", "contrafactive"):
        row[f"{attitude} (matching)"] = get_prop_correct(
            df, df[f"{attitude} allowed"] & df.matching
        )
        row[f"{attitude} (~matching)"] = get_prop_correct(
            df, df[f"{attitude} allowed"] & ~df.matching
        )

    overall_values = df.overall_correctness.value_counts(normalize=True)
    if True not in overall_values.index:
        print(overall_values)
        row["overall"] = 0.0
    else:
        row["overall"] = overall_values.loc[True]

    return row


def converted_fp_to_row(fp: Path, encoding_cfg: dict[str, Any]) -> dict[str, Any]:
    with open(fp.parent.parent / "parameters.json") as j_file:
        parameters = json.load(j_file)["train_parameters"]
    batch_size = parameters["batch_size"]

    converted_df = pd.read_pickle(fp)

    row = converted_df_to_row(converted_df)

    row["seed"] = int(fp.parent.parent.stem)

    n_batch = get_batch_num(fp)
    row["batch"] = n_batch
    row["sample size"] = n_batch * batch_size
    return row


def convert_test_fp(fp: Path, encoding_cfg: dict[str, Any]) -> None:
    original_df = pd.read_csv(
        fp,
        sep="\t",
        dtype={
            "matching": bool,
        },
    )

    assert "matching" in original_df.columns, f"Missing columns in: {fp}"
    converted_df = process_test(original_df, encoding_cfg)
    converted_df.to_pickle(fp.parent / f"{fp.stem}_converted.pkl")


def batch_convert(
    tests_dir: Path, encoding_cfg: dict[str, Any], n_jobs: int = 10
) -> None:
    func = partial(convert_test_fp, encoding_cfg=encoding_cfg)
    with Pool(n_jobs) as p:
        p.map(func, tests_dir.glob("[0-9]*/sub_eval/sub_test_[0-9]*.tsv"))


def create_seed_df(
    sub_evals_dir: Path, encoding_cfg: dict[str, Any], n_jobs: int = 10
) -> pd.DataFrame:
    sorted_paths = sorted(
        sub_evals_dir.glob("sub_test_[0-9]*_converted.pkl"), key=get_batch_num
    )

    func = partial(converted_fp_to_row, encoding_cfg=encoding_cfg)
    with Pool(n_jobs) as p:
        rows = p.map(func, sorted_paths)

    seed_df = pd.DataFrame(rows)

    return seed_df


def create_aggregated_df(tests_dir: Path, encoding_cfg: dict[str, Any]) -> pd.DataFrame:
    all_test_dfs = tuple(
        create_seed_df(seed_dir / "sub_eval", encoding_cfg)
        for seed_dir in tests_dir.glob("[0-9]*")
    )
    return pd.concat(all_test_dfs)


def create_single_selection_df(converted_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for key, grouped_df in converted_df.groupby(
        ["non-factive allowed", "factive allowed", "contrafactive allowed"]
    ):
        options = [
            att for att, b in zip(["non-factive", "factive", "contrafactive"], key) if b
        ]
        r = {"options": "-".join(options)}
        r.update(grouped_df.selected.value_counts(normalize=True).to_dict())
        rows.append(r)

    return pd.DataFrame(rows)


def converted_fp_to_selection_df(
    fp: Path
) -> dict[str, Any]:
    with open(fp.parent.parent / "parameters.json") as j_file:
        parameters = json.load(j_file)["train_parameters"]
        batch_size = parameters["batch_size"]

    converted_df = pd.read_pickle(fp)

    selection_df = create_single_selection_df(converted_df)

    selection_df["seed"] = int(fp.parent.parent.stem)

    n_batch = get_batch_num(fp)
    selection_df["batch"] = n_batch
    selection_df["sample size"] = n_batch * batch_size
    return selection_df


def create_aggregated_selection_df(
    tests_dir: pd.DataFrame, n_jobs: int = 10
) -> pd.DataFrame:
    func = partial(converted_fp_to_selection_df)
    with Pool(n_jobs) as p:
        all_converted_dfs = tuple(
            p.map(func, tests_dir.glob("[0-9]*/sub_eval/sub_test_[0-9]*_converted.pkl"))
        )

    return pd.concat(all_converted_dfs)


#  LocalWords:  eval pkl
