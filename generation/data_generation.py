"""Code to generate the data."""

import csv
import json
import click
import random

from enum import Enum
from typing import Generator
from itertools import product


MindRep = tuple[str, str, str, str, str, str, str]  # TODO: improve
PhraseTokens = tuple[str, str, str, str, str, str, str]
Vocab = dict[str, list[str]]


class TimeIndex(Enum):
    PRESENT = "present"
    PAST = "past"
    FUTURE = "future"


class SemValues(Enum):
    TRUE = "true"
    FALSE = "false"
    PFAILURE = "pfailure"
    UNKNOWN = "unknown"


class MindWorldRelation(Enum):
    SAME = "="
    DIFFERENT = "!="
    UNKNOWN = "?"


class SpecialTokens(Enum):
    START = "[START]"
    STOP = "[STOP]"


class AttitudeVerbs(Enum):
    FACTIVE = "factive"
    CONTRAFACTIVE = "contrafactive"
    NONFACTIVE = "non-factive"


class MindPhraseRelation(Enum):
    MATCHING = True
    NONMATCHING = False


SemFunctionTable = (
    (
        SemValues.TRUE,
        SemValues.TRUE,
        MindWorldRelation.SAME,
        (AttitudeVerbs.FACTIVE,),
        MindPhraseRelation.MATCHING,
    ),
    (
        SemValues.FALSE,
        SemValues.TRUE,
        MindWorldRelation.SAME,
        (AttitudeVerbs.FACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.FALSE,
        SemValues.FALSE,
        MindWorldRelation.SAME,
        (AttitudeVerbs.CONTRAFACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.FALSE,
        SemValues.UNKNOWN,
        MindWorldRelation.SAME,
        (AttitudeVerbs.NONFACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.PFAILURE,
        SemValues.TRUE,
        MindWorldRelation.SAME,
        (AttitudeVerbs.CONTRAFACTIVE,),
        MindPhraseRelation.MATCHING,
    ),
    (
        SemValues.PFAILURE,
        SemValues.FALSE,
        MindWorldRelation.SAME,
        (AttitudeVerbs.FACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.PFAILURE,
        SemValues.UNKNOWN,
        MindWorldRelation.SAME,
        (AttitudeVerbs.FACTIVE, AttitudeVerbs.CONTRAFACTIVE),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.TRUE,
        SemValues.FALSE,
        MindWorldRelation.DIFFERENT,
        (AttitudeVerbs.CONTRAFACTIVE,),
        MindPhraseRelation.MATCHING,
    ),
    (
        SemValues.FALSE,
        SemValues.TRUE,
        MindWorldRelation.DIFFERENT,
        (AttitudeVerbs.FACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.FALSE,
        SemValues.FALSE,
        MindWorldRelation.DIFFERENT,
        (AttitudeVerbs.CONTRAFACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.FALSE,
        SemValues.UNKNOWN,
        MindWorldRelation.DIFFERENT,
        (AttitudeVerbs.NONFACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.PFAILURE,
        SemValues.TRUE,
        MindWorldRelation.DIFFERENT,
        (AttitudeVerbs.CONTRAFACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.PFAILURE,
        SemValues.FALSE,
        MindWorldRelation.DIFFERENT,
        (AttitudeVerbs.FACTIVE,),
        MindPhraseRelation.MATCHING,
    ),
    (
        SemValues.PFAILURE,
        SemValues.UNKNOWN,
        MindWorldRelation.DIFFERENT,
        (AttitudeVerbs.FACTIVE, AttitudeVerbs.CONTRAFACTIVE),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.TRUE,
        SemValues.UNKNOWN,
        MindWorldRelation.UNKNOWN,
        (AttitudeVerbs.NONFACTIVE,),
        MindPhraseRelation.MATCHING,
    ),
    (
        SemValues.FALSE,
        SemValues.TRUE,
        MindWorldRelation.UNKNOWN,
        (AttitudeVerbs.FACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.FALSE,
        SemValues.FALSE,
        MindWorldRelation.UNKNOWN,
        (AttitudeVerbs.CONTRAFACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.FALSE,
        SemValues.UNKNOWN,
        MindWorldRelation.UNKNOWN,
        (AttitudeVerbs.NONFACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.PFAILURE,
        SemValues.TRUE,
        MindWorldRelation.UNKNOWN,
        (AttitudeVerbs.CONTRAFACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.PFAILURE,
        SemValues.FALSE,
        MindWorldRelation.UNKNOWN,
        (AttitudeVerbs.FACTIVE,),
        MindPhraseRelation.NONMATCHING,
    ),
    (
        SemValues.PFAILURE,
        SemValues.UNKNOWN,
        MindWorldRelation.UNKNOWN,
        (AttitudeVerbs.FACTIVE, AttitudeVerbs.CONTRAFACTIVE),
        MindPhraseRelation.MATCHING,
    ),
)


def tensify(verb: str, time_index: str) -> str:
    match time_index:
        case TimeIndex.PRESENT.value:
            return verb
        case TimeIndex.PAST.value:
            return verb + "ed"
        case TimeIndex.FUTURE.value:
            return "will-" + verb
        case _:
            raise ValueError(f"Unacceptable time_index: {time_index}")


def translate_mind(mind_rep: MindRep, time_index: str) -> PhraseTokens:
    verb = mind_rep[0]
    agent = mind_rep[1]
    main_ingredient = mind_rep[2]
    spice = mind_rep[3]
    dish = mind_rep[4]
    meal = mind_rep[5]
    day = mind_rep[6]

    phrase_tokens = (
        agent,
        tensify(verb, time_index),
        f"{main_ingredient}-{spice}",
        dish,
        "for",
        meal,
        day,
    )

    return phrase_tokens


def create_random_mind_rep(avoid_rep: MindRep, vocab: Vocab) -> MindRep:
    new_rep = []

    for possible_tokens in vocab.values():
        new_rep.append(random.choice(possible_tokens))

    assert new_rep != avoid_rep

    return tuple(new_rep)


def create_random_phrase(
    avoid_rep: MindRep, vocab: Vocab, day_to_index: dict[str, str]
) -> PhraseTokens:
    new_rep = create_random_mind_rep(avoid_rep, vocab)
    time_index = day_to_index[new_rep[-1]]

    return translate_mind(new_rep, time_index)


def rows_for_phrase(
    mind_rep: MindRep,
    phrase_tokens: PhraseTokens,
    vocab: Vocab,
    day_to_index: dict[str, str],
) -> Generator[dict[str, str | bool | None], None, None]:
    for (
        main_value,
        sub_value,
        mind_world_relation,
        attitude_verbs,
        matching,
    ) in SemFunctionTable:
        # if not ((semantic_value == SemValues.TRUE) and (mind_world_relation == MindWorldRelation.UNKNOWN)):
        #     if random.random() > 1/6:
        #         continue

        match len(attitude_verbs):
            case 1:
                verb_1 = attitude_verbs[0].value
                verb_2 = None
                selected_verb = verb_1
            case 2:
                verb_1 = attitude_verbs[0].value
                verb_2 = attitude_verbs[1].value
                selected_verb = random.choice((verb_1, verb_2))
            case _:
                raise ValueError(
                    f"Incorrect number of attitude verbs: {attitude_verbs}"
                )

        if matching.value:
            train_phrase = " ".join(
                [SpecialTokens.START.value, selected_verb]
                + list(phrase_tokens)
                + [SpecialTokens.STOP.value]
            )
        else:
            train_phrase = " ".join(
                [SpecialTokens.START.value, selected_verb]
                + list(create_random_phrase(mind_rep, vocab, day_to_index))
                + [SpecialTokens.STOP.value]
            )

        yield {
            "main_value": main_value.value,
            "sub_value": sub_value.value,
            "mind_world_relation": mind_world_relation.value,
            "mind_representation": " ".join(mind_rep),
            "attitude_verb_1": verb_1,
            "attitude_verb_2": verb_2,
            "compare_phrase": " ".join(phrase_tokens),
            "matching": matching.value,
            "train_phrase": train_phrase,
        }


@click.command()
@click.argument("vocab_file", type=click.File("r"))
@click.argument("output_fp", type=click.Path(exists=False))
def generate_data(vocab_file, output_fp) -> None:
    random.seed(1848)

    vocab_info = json.load(vocab_file)
    vocab: Vocab = vocab_info["vocab"]
    day_to_index = vocab_info["day_to_index"]

    field_names = (
        "main_value",
        "sub_value",
        "mind_world_relation",
        "mind_representation",
        "attitude_verb_1",
        "attitude_verb_2",
        "compare_phrase",
        "matching",
        "train_phrase",
    )

    with open(output_fp, "w") as tsv_file:
        writer = csv.DictWriter(tsv_file, field_names, delimiter="\t")
        writer.writeheader()
        for mind_rep in product(*vocab.values()):
            time_index = day_to_index[mind_rep[-1]]

            corresponding_phrase = translate_mind(mind_rep, time_index)
            for row in rows_for_phrase(
                mind_rep, corresponding_phrase, vocab, day_to_index
            ):
                writer.writerow(row)


if __name__ == "__main__":
    generate_data()

#  LocalWords:  factive contrafactive
