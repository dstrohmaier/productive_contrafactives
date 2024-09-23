import json
from itertools import product

import click

import pandas as pd

from data_generation import SpecialTokens, AttitudeVerbs, SemValues, MindWorldRelation


@click.command()
@click.argument("vocab_fpath", type=click.Path(exists=True))
@click.argument("out_fpath", type=click.Path(exists=False))
def create_encoding(vocab_fpath, out_fpath):
    with open(vocab_fpath, encoding="utf-8") as f:
        vocab = json.load(f)["vocab"]

    sem_tokens = {t.value for t in SemValues}
    mw_tokens = {t.value for t in MindWorldRelation}
    all_vocab_tokens = {t for tokens in vocab.values() for t in tokens}

    input_vocab = set()
    input_vocab = input_vocab.union(sem_tokens)
    input_vocab = input_vocab.union(mw_tokens)
    input_vocab = input_vocab.union(all_vocab_tokens)

    special_tokens = {t.value for t in SpecialTokens}
    attitude_tokens = {t.value for t in AttitudeVerbs}

    output_verbs = set(vocab["verbs"])

    past_verbs = set()
    future_verbs = set()
    for v in output_verbs:
        past_verbs.add(v + "ed")
        future_verbs.add("will-" + v)
    output_verbs = output_verbs.union(past_verbs)
    output_verbs = output_verbs.union(future_verbs)

    output_agents = set(vocab["agent"])
    output_descriptors = set(
        "-".join(pro) for pro in product(vocab["main_ingredient"], vocab["spice"])
    )

    output_for = {
        "for",
    }
    output_dish = set(vocab["dish"])
    output_meal = set(vocab["meal"])
    output_day = set(vocab["day"])

    output_vocab = set()
    output_vocab = output_vocab.union(special_tokens)
    output_vocab = output_vocab.union(attitude_tokens)
    output_vocab = output_vocab.union(output_agents)
    output_vocab = output_vocab.union(output_verbs)
    output_vocab = output_vocab.union(output_descriptors)
    output_vocab = output_vocab.union(output_for)
    output_vocab = output_vocab.union(output_dish)
    output_vocab = output_vocab.union(output_meal)
    output_vocab = output_vocab.union(output_day)

    input_map = {t: i for i, t in enumerate(input_vocab)}
    output_map = {t: i for i, t in enumerate(output_vocab)}

    attitude_map = {t: output_map[t] for t in attitude_tokens}
    special_map = {t: output_map[t] for t in special_tokens}
    verbs_map = {t: output_map[t] for t in output_verbs}
    agent_map = {t: output_map[t] for t in output_agents}
    descriptors_map = {t: output_map[t] for t in output_descriptors}
    for_map = {t: output_map[t] for t in output_for}
    dish_map = {t: output_map[t] for t in output_dish}
    meal_map = {t: output_map[t] for t in output_meal}
    day_map = {t: output_map[t] for t in output_day}

    maps_in_order = [
        "attitude_map",
        "agent_map",
        "verbs_map",
        "descriptors_map",
        "dish_map",
        "for_map",
        "meal_map",
        "day_map",
        "special_map",
    ]

    encoding_cfg = {
        "input_map": input_map,
        "output_map": output_map,
        "attitude_map": attitude_map,
        "special_map": special_map,
        "verbs_map": verbs_map,
        "agent_map": agent_map,
        "descriptors_map": descriptors_map,
        "for_map": for_map,
        "dish_map": dish_map,
        "meal_map": meal_map,
        "day_map": day_map,
        "maps_in_order": maps_in_order,
    }

    with open(out_fpath, "w", encoding="utf-8") as j_file:
        json.dump(encoding_cfg, j_file, indent=4)


if __name__ == "__main__":
    create_encoding()  # pylint: disable=no-value-for-parameter
