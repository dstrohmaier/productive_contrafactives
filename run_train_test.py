import json

from pathlib import Path

import click

from main_experiment.setups.train_test import train_test
from main_experiment.utilities.repro import set_seed


@click.command()
@click.argument("settings_fpath", type=click.Path(exists=True))
def run(settings_fpath):
    with open(settings_fpath, encoding="utf-8") as j_file:
        settings = json.load(j_file)

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(settings["seed"])
    set_seed(seed)

    train_test(
        Path(settings["data_dir"]),
        output_dir / str(seed),
        settings["text_parameters"],
        settings["model_parameters"],
        settings["train_parameters"],
        settings["device"],
        seed=seed,
    )


if __name__ == "__main__":
    run()
