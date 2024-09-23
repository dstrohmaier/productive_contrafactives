import json
import torch
import pandas as pd

from typing import Any
from pathlib import Path

from torch.utils.data import TensorDataset


def encode(string, encoding):
    converted = [encoding[token] for token in string.split()]
    return torch.LongTensor(converted)


def df_to_dataset(df: pd.DataFrame, encoding_cfg: dict[str, Any]):
    main_sem_codes, sub_sem_codes, mw_relation_codes, mind_codes = (
        [],
        [],
        [],
        [],
    )
    attitude_1_codes, attitude_2_codes = [], []
    compare_codes, matching_codes, train_phrase_codes = [], [], []

    input_map = encoding_cfg["input_map"]
    output_map = encoding_cfg["output_map"]
    matching_map = {True: 1, False: 0}

    for _, row in df.iterrows():
        main_sem_codes.append(input_map[row.main_value])
        sub_sem_codes.append(input_map[row.sub_value])
        mw_relation_codes.append(input_map[row.mind_world_relation])
        mind_codes.append(encode(row.mind_representation, input_map))
        attitude_1_codes.append(output_map[row.attitude_verb_1])
        attitude_2_codes.append(output_map.get(row.attitude_verb_2, len(output_map)))
        compare_codes.append(encode(row.compare_phrase, output_map))
        matching_codes.append(matching_map[row.matching])
        train_phrase_codes.append(encode(row.train_phrase, output_map))

    dataset = TensorDataset(
        torch.LongTensor(main_sem_codes).reshape(-1, 1),
        torch.LongTensor(sub_sem_codes).reshape(-1, 1),
        torch.LongTensor(mw_relation_codes).reshape(-1, 1),
        torch.stack(mind_codes),
        torch.LongTensor(attitude_1_codes).reshape(-1, 1),
        torch.LongTensor(attitude_2_codes).reshape(-1, 1),
        torch.stack(compare_codes),
        torch.LongTensor(matching_codes).reshape(-1, 1),
        torch.stack(train_phrase_codes),
    )
    return dataset


def load_cv_splits(data_dir: Path, n_splits: int = 5):
    with open(data_dir / "encoding_cfg.json", encoding="utf-8") as j_file:
        encoding_cfg = json.load(j_file)

    for i in range(n_splits):
        train_dataset = load_tsv(
            data_dir / "split" / f"split_train_{i}.tsv", encoding_cfg
        )
        test_dataset = load_tsv(
            data_dir / "split" / f"split_test_{i}.tsv", encoding_cfg
        )

        yield train_dataset, test_dataset


def load_train_test(data_dir: Path):
    with open(data_dir / "encoding_cfg.json", encoding="utf-8") as j_file:
        encoding_cfg = json.load(j_file)

    train_dataset = load_tsv(data_dir / "split" / "train.tsv", encoding_cfg)
    test_dataset = load_tsv(data_dir / "split" / "test.tsv", encoding_cfg)

    return train_dataset, test_dataset


def load_tsv(file_path: Path, encoding_cfg: dict[str, Any]):
    df = pd.read_csv(file_path, sep="\t")
    dataset = df_to_dataset(df, encoding_cfg)
    return dataset


def load_encoding_cfg(data_dir: Path):
    with open(data_dir / "encoding_cfg.json", encoding="utf-8") as j_file:
        encoding_cfg = json.load(j_file)
    return encoding_cfg


def load_attitude_special_ids(data_dir: Path):
    with open(data_dir / "encoding_cfg.json", encoding="utf-8") as j_file:
        encoding_cfg = json.load(j_file)

    attitude_map = encoding_cfg["attitude_map"]
    special_map = encoding_cfg["special_map"]

    return set(attitude_map.values()), set(special_map.values())
