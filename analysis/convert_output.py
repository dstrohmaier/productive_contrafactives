"""Script to convert codes in output to readable format."""


import json
from typing import Union

import click
import pandas as pd

from bidict import bidict


def convert_codes(codes: Union[str, int], encoding_map: dict[str, int]):
    b_map = bidict(encoding_map)
    b_map["None"] = len(b_map)

    if isinstance(codes, int):
        return b_map.inv[codes]

    indices = list(map(int, codes.split()))
    return " ".join([b_map.inv[i] for i in indices])


@click.command()
@click.argument("data_fp", type=click.Path(exists=True))
@click.argument("encoding_cfg_fp", type=click.Path(exists=True))
@click.argument("output_fp", type=click.Path())
def convert_output(data_fp: str, encoding_cfg_fp: str, output_fp: str) -> None:
    df = pd.read_csv(data_fp, sep="\t")

    with open(encoding_cfg_fp, encoding="utf-8") as jf:
        encoding_cfg = json.load(jf)

    input_map = encoding_cfg["input_map"]
    df.semantic_value = df.semantic_value.apply(lambda x: convert_codes(x, input_map))
    df.mind_world_relation = df.mind_world_relation.apply(
        lambda x: convert_codes(x, input_map)
    )
    df.mind_codes = df.mind_codes.apply(lambda x: convert_codes(x, input_map))

    output_map = encoding_cfg["output_map"]
    df.attitude_1 = df.attitude_1.apply(lambda x: convert_codes(x, output_map))
    df.attitude_2 = df.attitude_2.apply(lambda x: convert_codes(x, output_map))
    df.compare_codes = df.compare_codes.apply(lambda x: convert_codes(x, output_map))
    df.output = df.output.apply(lambda x: convert_codes(x, output_map))

    df.to_csv(output_fp, sep="\t", index=False)


if __name__ == "__main__":
    convert_output()
