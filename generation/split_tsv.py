"""Convenience utility for splitting TSV-files into train and test data."""

from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit


@click.command()
@click.argument("input_fpath", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=True))
@click.option("--create_dev", default=False, type=bool)
@click.option("--cv_splits", default=True, type=bool)
@click.option("--n_splits", default=5, type=int)
@click.option("--test_prop", default=0.1, type=float)
@click.option("--seed", default=1848, type=int)
def split_tsv(
    input_fpath: str,
    output_dir: str,
    create_dev: bool,
    cv_splits: bool,
    n_splits: int,
    test_prop: float,
    seed: int,
) -> None:
    """Main function for splitting TSV"""

    original_df = pd.read_csv(input_fpath, sep="\t", keep_default_na=False)
    traindev_df, test_df = train_test_split(
        original_df, test_size=test_prop, random_state=seed
    )

    if create_dev:
        train_df, dev_df = train_test_split(
            traindev_df, test_size=test_prop, random_state=seed
        )

        dev_path = Path(output_dir) / "dev.tsv"
        dev_df.to_csv(dev_path, sep="\t", index=False)
    else:
        train_df = traindev_df

    if cv_splits:
        splitter = ShuffleSplit(
            n_splits=n_splits, test_size=test_prop, random_state=seed
        )
        for i, (train_index, test_index) in enumerate(splitter.split(train_df)):
            split_train_df = train_df.iloc[train_index]
            split_test_df = train_df.iloc[test_index]

            split_train_path = Path(output_dir) / f"split_train_{i}.tsv"
            split_train_df.to_csv(split_train_path, sep="\t", index=False)
            split_test_path = Path(output_dir) / f"split_test_{i}.tsv"
            split_test_df.to_csv(split_test_path, sep="\t", index=False)

    train_path = Path(output_dir) / "train.tsv"
    train_df.to_csv(train_path, sep="\t", index=False)

    test_path = Path(output_dir) / "test.tsv"
    test_df.to_csv(test_path, sep="\t", index=False)


if __name__ == "__main__":
    split_tsv()  # pylint: disable=no-value-for-parameter
