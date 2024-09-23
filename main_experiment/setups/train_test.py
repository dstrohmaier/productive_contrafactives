import json
import torch

from pathlib import Path
from typing import Dict, Any, Literal

from main_experiment.model.training_and_testing import train, test
from main_experiment.model.transformer import RestrictedGenTransformer
from main_experiment.data_processing.loading import load_train_test
from main_experiment.utilities.logging_utils import start_logging


def train_test(
    data_dir: Path,
    output_dir: Path,
    text_parameters: Dict[str, int],
    model_parameters: Dict[str, Any],
    train_parameters: Dict[str, Any],
    device: Literal["cpu", "cuda"],
    seed: int,
):
    logger = start_logging(output_dir)
    logger.info("Seed: %s", seed)

    model_parameters.update(text_parameters)

    with open(data_dir / "encoding_cfg.json") as j_file:
        encoding_cfg = json.load(j_file)

    logger.info("trans_parameters: %s", model_parameters)
    logger.info("train_parameters: %s", train_parameters)

    with open(output_dir / "parameters.json", "w", encoding="utf-8") as jf:
        json.dump(
            {
                "model_parameters": model_parameters,
                "train_parameters": train_parameters,
            },
            jf,
            indent=4,
        )

    model: RestrictedGenTransformer | torch.nn.DataParallel
    model = RestrictedGenTransformer(encoding_cfg=encoding_cfg, **model_parameters)
    model.init_weights()

    if device == "parallel":
        model = torch.nn.DataParallel(model)
        device = "cuda"
    logger.info(f"device: {device}")
    logger.info("Initialised model")

    train_set, test_set = load_train_test(data_dir)
    test_out_dir = output_dir / "sub_eval"
    test_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Main training")
    with open(output_dir / "training.tsv", "w") as train_file:
        trained_model = train(
            model,
            train_set,
            tsv_file=train_file,
            encoding_cfg=encoding_cfg,
            device=device,
            test_dataset=test_set,
            test_out_dir=test_out_dir,
            **train_parameters,
        )

    with open(output_dir / "test.tsv", "w") as test_file:
        test(
            trained_model,
            test_set,
            tsv_file=test_file,
            encoding_cfg=encoding_cfg,
            batch_size=10_000,
            sentence_len=text_parameters["sentence_len"],
            device=device,
        )
