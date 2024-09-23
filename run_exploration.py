import json
import random

from pathlib import Path
from multiprocessing import Process

import click

from main_experiment.setups.exploration import cross_search
from main_experiment.utilities.repro import set_seed


@click.command()
@click.argument("settings_fpath", type=click.Path(exists=True))
def run(settings_fpath):
    with open(settings_fpath, encoding="utf-8") as j_file:
        settings = json.load(j_file)

    processes = []

    seen_seeds = set()

    for i in range(settings["n_draws"]):
        seed = random.randint(0, 9999999)
        while seed in seen_seeds:
            seed = random.randint(0, 9999999)
        seen_seeds.add(seed)

        set_seed(seed)
        output_dir = Path(settings["output_dir"]) / str(seed)
        output_dir.mkdir(parents=True, exist_ok=True)

        args = (
            Path(settings["data_dir"]),
            output_dir,
            settings["text_params"],
            settings["model_space"],
            settings["train_space"],
            settings["device"],
            seed,
        )

        p = Process(target=cross_search, args=args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    run()
