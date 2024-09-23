"""Cross-Validation Code, assumes splits existing"""

import json
import random

from typing import Dict, Any, Literal
from pathlib import Path

import torch

from main_experiment.model.training_and_testing import train, test
from main_experiment.model.transformer import RestrictedGenTransformer

# from main_experiment.model.transformer import GenTransformer
from main_experiment.data_processing.loading import load_cv_splits, load_encoding_cfg
from main_experiment.utilities.logging_utils import start_logging


def cross_search(
    data_dir: Path,
    output_dir: Path,
    text_params: Dict[str, int],
    model_space: Dict[str, Any],
    train_space: Dict[str, Any],
    device: Literal["parallel", "cpu", "cuda"],
    seed: int,
):
    logger = start_logging(output_dir)
    logger.info("Seed: %s", seed)
    logger.info("Model hyperparameter space: %s", model_space)
    logger.info("Training hyperparameter space: %s", train_space)

    model_parameters = draw_parameters(model_space)
    model_parameters.update(text_params)
    train_parameters = draw_parameters(train_space)
    logger.info("train_parameters: %s", train_parameters)
    logger.info("model_parameters: %s", model_parameters)

    with open(output_dir / "parameters.json", "w", encoding="utf-8") as jf:
        json.dump(
            {
                "model_parameters": model_parameters,
                "train_parameters": train_parameters,
            },
            jf,
            indent=4,
        )

    encoding_cfg = load_encoding_cfg(data_dir)

    for i, (train_set, test_set) in enumerate(load_cv_splits(data_dir)):
        logger.info("Validation split: %s", i)
        model: RestrictedGenTransformer | torch.nn.DataParallel
        model = RestrictedGenTransformer(encoding_cfg=encoding_cfg, **model_parameters)
        # logger.info('parameters: %s', list(model.named_parameters()))
        model.init_weights()

        if device == "parallel":
            model = torch.nn.DataParallel(model)
            device = "cuda"
        logger.info("device: %s", device)

        logger.info("Initialised model")

        trained_model = train(
            model,
            train_set,
            tsv_file=None,
            encoding_cfg=encoding_cfg,
            device=device,
            **train_parameters,
        )

        with open(output_dir / f"test_{i}.tsv", "w", encoding="utf-8") as test_file:
            test(
                trained_model,
                test_set,
                test_file,
                encoding_cfg=encoding_cfg,
                batch_size=10_000,
                sentence_len=text_params["sentence_len"],
                device=device,
            )


def draw_parameters(space):
    return {key: random.choice(value) for key, value in space.items()}
