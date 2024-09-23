"""Script to convert codes in output to readable format."""


import json
from pathlib import Path
from typing import Any
from multiprocessing import Pool

import click
import pandas as pd


def get_setting_result(setting_dir: Path) -> tuple[float, int, dict[str, Any]]:
    overall_performances = []
    for test_fp in setting_dir.glob("test_*.tsv"):
        test_df = pd.read_csv(test_fp, sep="\t")
        performance = test_df.overall_correctness.value_counts(normalize=True).loc[True]
        overall_performances.append(performance)

    seed = int(setting_dir.stem)

    with open(setting_dir / "parameters.json", "r") as j_file:
        parameters = json.load(j_file)
    return sum(overall_performances) / len(overall_performances), seed, parameters


@click.command()
@click.argument("output_dir_str", type=click.Path(exists=True))
def get_statistics(output_dir_str: str) -> None:
    output_dir = Path(output_dir_str)

    with Pool(20) as p:
        collected_results = p.map(get_setting_result, output_dir.glob("explore/*"))

    best_performance = float("-inf")
    best_parameters = {}

    for performance, seed, parameters in collected_results:
        if performance < best_performance:
            continue
        elif performance > best_performance:
            best_performance = performance
            best_parameters = {seed: parameters}
        elif performance == best_performance:
            best_parameters[seed] = parameters

    assert isinstance(best_parameters, dict)
    for seed, parameters in best_parameters.items():
        parameters["seed"] = seed

        selected_dir = output_dir / "selected_parameters"
        selected_dir.mkdir(parents=True, exist_ok=True)

        with open(selected_dir / f"{seed}_parameters.json", "w") as j_file:
            json.dump(parameters, j_file, indent=4)


if __name__ == "__main__":
    get_statistics()
