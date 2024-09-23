"""Script to convert codes in output to readable format."""


from pathlib import Path
from typing import Any
from multiprocessing import Pool

import click
import pandas as pd

COLS = ((0, True), (1, True), "overall", "seed", "file")


def load_process(fp: Path) -> list[dict[str, Any]]:
    current_df = pd.read_csv(fp, sep="\t")

    assert "matching" in current_df.columns, f"Incorrect columns in: {fp}"

    row = (
        current_df.groupby("matching")
        .overall_correctness.value_counts(normalize=True)
        .to_dict()
    )

    overall_values = current_df.overall_correctness.value_counts(normalize=True)
    if True not in overall_values.index:
        print(overall_values)
        row["overall"] = None
    else:
        row["overall"] = overall_values.loc[True]

    row["seed"] = fp.parent.stem
    row["file"] = fp.stem

    return row


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
def get_statistics(data_dir: str) -> None:
    with Pool(20) as p:
        rows = p.map(load_process, Path(data_dir).glob("**/test_*.tsv"))

    df = pd.DataFrame(rows)
    df = df.loc[:, COLS]

    print(df.to_markdown())
    print(df.iloc[df[(1, True)].argmax()])


if __name__ == "__main__":
    get_statistics()
