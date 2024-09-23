import csv
import time
import logging

from pathlib import Path
from typing import TextIO, Literal, Union, Optional

import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from main_experiment.model.transformer import RestrictedGenTransformer, GenTransformer
from main_experiment.evaluation.correctness import check_correctness


Model = Union[RestrictedGenTransformer, GenTransformer, nn.DataParallel]
Devices = Literal["cpu", "cuda"]


class AppropriateLoss(nn.Module):
    def __init__(self, encoding_cfg: dict[str, dict[str, int]]) -> None:
        super().__init__()
        self.encoding_cfg = encoding_cfg
        self.logger = logging.getLogger("contrafactives")

    @staticmethod
    def get_attitude_target(
        b_attitude_1: Tensor, b_attitude_2: Tensor, n_classes: int
    ) -> Tensor:
        b_attitude_1 = b_attitude_1.squeeze()
        b_attitude_2 = b_attitude_2.squeeze()

        combined_hot = F.one_hot(b_attitude_1, n_classes)

        one_hot_2 = F.one_hot(b_attitude_2, n_classes + 1)  # +1 when there is nothing
        one_hot_2 = one_hot_2[:, :-1]  # excluding when there is nothing

        assert combined_hot.shape == one_hot_2.shape

        combined_hot += one_hot_2

        return combined_hot.unsqueeze(1).float()

    def get_phrase_target(
        self,
        b_compare: Tensor,
        b_matching: Tensor,
        n_classes: int,
        mis_val: float = 0.5,
    ) -> Tensor:
        b_matching = b_matching.squeeze().bool()
        # b_matching: batch x 1 -> batch
        assert len(b_matching.shape) == 1

        b_non_matching = ~b_matching
        # inverting the mask

        match_target = F.one_hot(b_compare[b_matching, :], n_classes).float()
        # when the phrase matches, than we have on correct label, indicated by one_hot encoding

        selected_len = b_non_matching.sum().item()
        assert selected_len < b_non_matching.shape[0]

        seq_len = b_compare.shape[1]
        non_match_target = torch.zeros(selected_len, seq_len, n_classes).to(
            b_compare.device
        )

        selected_maps = self.encoding_cfg["maps_in_order"][1:-1]
        # getting rid of map for attitude and special tokens
        for i, map_name in enumerate(selected_maps):
            values = tuple(self.encoding_cfg[map_name].values())
            non_match_target[:, i, values] = 1.0

        non_match_target = non_match_target.reshape(-1, n_classes)

        # self.logger.info('non_match_target.shape: %s', non_match_target.shape)
        # self.logger.info('selected_len*seq_len: %s', selected_len*seq_len)
        # self.logger.info('b_compare[b_non_matching, :].flatten().shape: %s', b_compare[b_non_matching, :].flatten().shape)

        assert selected_len * seq_len == non_match_target.shape[0]
        non_match_target[
            torch.arange(selected_len * seq_len), b_compare[b_non_matching, :].flatten()
        ] = mis_val

        non_match_target = non_match_target.reshape(selected_len, seq_len, n_classes)

        batch_size = b_compare.shape[0]
        phrase_target = torch.zeros(batch_size, seq_len, n_classes).to(b_compare.device)
        phrase_target[b_matching, :, :] = match_target
        phrase_target[b_non_matching, :, :] = non_match_target
        return phrase_target

    @staticmethod
    def get_special_target(b_train_phrase: Tensor, n_classes: int) -> Tensor:
        b_special_phrase = b_train_phrase[:, -1]
        special_target = F.one_hot(b_special_phrase, n_classes).float()
        return special_target.unsqueeze(1)

    def forward(
        self,
        logits: Tensor,
        b_train_phrase: Tensor,
        b_attitude_1: Tensor,
        b_attitude_2: Tensor,
        b_compare: Tensor,
        b_matching: Tensor,
    ) -> Tensor:
        # b_train_phrase: batch x sequence

        b_matching = b_matching.squeeze()

        n_classes = logits.shape[2]
        assert n_classes == len(self.encoding_cfg["output_map"])

        batch_size = b_matching.shape[0]
        seq_len = logits.shape[1]

        # self.logger.info('batch_size: %s | seq_len: %s |  n_classes: %s',
        #                  batch_size, seq_len, n_classes)

        target = torch.cat(
            (
                self.get_attitude_target(b_attitude_1, b_attitude_2, n_classes),
                self.get_phrase_target(b_compare, b_matching, n_classes),
                self.get_special_target(b_train_phrase, n_classes),
            ),
            dim=1,
        )

        losses = F.binary_cross_entropy_with_logits(
            logits.reshape(-1, n_classes),
            target.reshape(-1, n_classes),
            reduction="none",
        )

        losses = losses.sum(1)
        b_loss = losses.reshape(batch_size, seq_len)

        return b_loss


def tensor_to_str(t: Tensor) -> str:
    return " ".join(map(str, t.detach().cpu().tolist()))


def write_compare_tensors(
    writer: csv.DictWriter, epoch: int, batch_no: int, all_compare_tensors
):
    for (
        s_main_sem,
        s_sub_sem,
        s_mw_relation,
        s_mind,
        s_attitude_1,
        s_attitude_2,
        s_compare,
        s_matching,
        s_train_phrase,
        s_output,
        s_loss,
        s_attitude_c,
        s_phrase_c,
        s_special_c,
        s_overall_c,
    ) in zip(*all_compare_tensors):
        writer.writerow(
            {
                "epoch": epoch,
                "batch_no": batch_no,
                "main_value": s_main_sem.item(),
                "sub_value": s_sub_sem.item(),
                "mind_world_relation": s_mw_relation.item(),
                "mind_codes": tensor_to_str(s_mind),
                "attitude_1": s_attitude_1.item(),
                "attitude_2": s_attitude_2.item(),
                "compare_codes": tensor_to_str(s_compare),
                "matching": s_matching.item(),
                "train_codes": tensor_to_str(s_train_phrase),
                "output": tensor_to_str(s_output),
                "loss": tensor_to_str(s_loss),
                "attitude_correctness": s_attitude_c.item(),
                "phrase_correctness": s_phrase_c.item(),
                "special_correctness": s_special_c.item(),
                "overall_correctness": s_overall_c.item(),
            }
        )


def train(
    model: Model,
    dataset: TensorDataset,
    tsv_file: Optional[TextIO],
    encoding_cfg: dict[str, dict[str, int]],
    batch_size: int,
    epochs: int,
    lr: float,
    max_grad_norm: float = 1.0,
    device: Devices = "cuda",
    test_dataset: Optional[TensorDataset] = None,
    test_out_dir: Optional[Path] = None,
) -> Model:
    logger = logging.getLogger("contrafactives")
    logger.info("training device: %s", device)
    logger.info("training lr: %s", lr)
    logger.info("training max_grad_norm: %s", max_grad_norm)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    fieldnames = [
        "epoch",
        "batch_no",
        "main_value",
        "sub_value",
        "mind_world_relation",
        "mind_codes",
        "attitude_1",
        "attitude_2",
        "compare_codes",
        "matching",
        "train_codes",
        "output",
        "loss",
        "attitude_correctness",
        "phrase_correctness",
        "special_correctness",
        "overall_correctness",
    ]

    if tsv_file:
        writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    criterion = AppropriateLoss(encoding_cfg)

    model.to(device)
    model.train()

    total_batch_counter = 0
    # counter for when to test in between

    for e in range(epochs):
        start_time = time.time()
        logger.info("Starting epoch %s at %s", e, start_time)

        for i, b_data in enumerate(loader):
            (
                b_main_sem,
                b_sub_sem,
                b_mw_relation,
                b_mind,
                b_attitude_1,
                b_attitude_2,
                b_compare,
                b_matching,
                b_train_phrase,
            ) = map(lambda t: t.to(device), b_data)

            logits = model(
                b_main_sem, b_sub_sem, b_mind, b_mw_relation, b_train_phrase[:, :-1]
            )
            # removing [STOP] from b_train_phrase
            # logits: batch x sequence-1 x vocabulary

            # b_compare: batch x sequence-3 x vocabulary
            assert b_train_phrase.shape[0] <= batch_size
            # can be smaller than batch size at the end, when we are running low on samples
            assert b_compare.shape[0] == b_train_phrase.shape[0]
            assert (
                b_compare.shape[1] + 3 == b_train_phrase.shape[1]
            ), f"{b_compare.shape[1]+3} != {b_train_phrase.shape[1]}"
            # +3 for special tokens and attitude
            assert logits.shape[0] == b_train_phrase.shape[0]
            assert logits.shape[1] + 1 == b_train_phrase.shape[1]
            # +1 for [START] token
            assert logits.shape[2] == len(encoding_cfg["output_map"])

            b_loss = criterion(
                logits,
                b_train_phrase,
                b_attitude_1,
                b_attitude_2,
                b_compare,
                b_matching,
            )

            # getting the maximum indices along the sequence dimensions
            # Then decode the sequence greedily all at once

            _, max_output = torch.max(logits, dim=2)
            max_output = torch.cat((b_train_phrase[:, :1], max_output), dim=1)
            # adding [START] token at beginning for consistency
            assert max_output.shape == b_train_phrase.shape

            # To track the model, we are saving all relevant information
            # for every instance, unbinding the batch

            # logger.info('logits.shape: %s', logits.shape)
            # logger.info('b_loss.shape: %s', b_loss.shape)

            b_attitude_c, b_phrase_c, b_special_c, b_overall_c = check_correctness(
                max_output,
                b_train_phrase,
                b_compare,
                b_matching,
                b_attitude_1,
                b_attitude_2,
            )

            # logger.info("b_attitude_c.shape: %s", b_attitude_c.shape)
            # logger.info("b_phrase_c.shape: %s", b_phrase_c.shape)
            # logger.info("b_special_c.shape: %s", b_special_c.shape)
            # logger.info("b_overall_c.shape: %s", b_overall_c.shape)

            all_compare_tensors = (
                b_main_sem,
                b_sub_sem,
                b_mw_relation,
                b_mind,
                b_attitude_1,
                b_attitude_2,
                b_compare,
                b_matching,
                b_train_phrase,
                max_output,
                b_loss,
                b_attitude_c,
                b_phrase_c,
                b_special_c,
                b_overall_c,
            )

            assert all(
                tensor.shape[0] == b_train_phrase.shape[0]
                for tensor in all_compare_tensors
            )

            if tsv_file:
                write_compare_tensors(writer, e, i, all_compare_tensors)

            optimizer.zero_grad()
            b_loss.sum().backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if test_dataset and (total_batch_counter % 20 == 0):
                assert isinstance(test_out_dir, Path)

                with open(
                    test_out_dir / f"sub_test_{total_batch_counter}.tsv", "w"
                ) as test_file:
                    test(
                        model,
                        test_dataset,
                        tsv_file=test_file,
                        encoding_cfg=encoding_cfg,
                        batch_size=10_000,
                        sentence_len=b_train_phrase.shape[1],
                        device=device,
                    )

                model.train()
            total_batch_counter += 1

        logger.info("End of epoch after %s", time.time() - start_time)

    return model


def test(
    model: Model,
    dataset: TensorDataset,
    tsv_file: TextIO,
    encoding_cfg: dict[str, dict[str, int]],
    batch_size: int,
    sentence_len: int,
    device: Devices = "cuda",
) -> None:
    logger = logging.getLogger("contrafactives")
    logger.info("testing device: %s", device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    fieldnames = [
        "batch_no",
        "main_value",
        "sub_value",
        "mind_world_relation",
        "mind_codes",
        "attitude_1",
        "attitude_2",
        "compare_codes",
        "matching",
        "train_codes",
        "output",
        "loss",
        "attitude_correctness",
        "phrase_correctness",
        "special_correctness",
        "overall_correctness",
    ]
    writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    criterion = AppropriateLoss(encoding_cfg)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_i, b_data in enumerate(loader):
            if batch_i % 100 == 0:
                logger.info("In batch: %s", batch_i)

            (
                b_main_sem,
                b_sub_sem,
                b_mw_relation,
                b_mind,
                b_attitude_1,
                b_attitude_2,
                b_compare,
                b_matching,
                b_train_phrase,
            ) = map(lambda t: t.to(device), b_data)

            max_output = b_train_phrase[:, :1]
            # Using only special start token [START]

            for i in range(sentence_len - 1):  # len - [START]
                logits = model(b_main_sem, b_sub_sem, b_mind, b_mw_relation, max_output)
                # assumed logits shape: batch x sequence x vocab

                next_max_output = torch.argmax(logits[:, i : i + 1, :], dim=2)
                # i:i+1 instead of just i to preserve dimension

                max_output = torch.cat([max_output, next_max_output], dim=1)

            assert b_train_phrase.shape[0] <= batch_size
            # batches not complete at the end
            assert b_compare.shape[0] == b_train_phrase.shape[0]
            assert b_compare.shape[1] + 3 == b_train_phrase.shape[1]
            assert logits.shape[0] == b_train_phrase.shape[0]
            assert logits.shape[1] + 1 == b_train_phrase.shape[1]
            assert logits.shape[2] == len(encoding_cfg["output_map"])

            b_loss = criterion(
                logits,
                b_train_phrase,
                b_attitude_1,
                b_attitude_2,
                b_compare,
                b_matching,
            )

            b_attitude_c, b_phrase_c, b_special_c, b_overall_c = check_correctness(
                max_output,
                b_train_phrase,
                b_compare,
                b_matching,
                b_attitude_1,
                b_attitude_2,
            )

            all_compare_tensors = (
                b_main_sem,
                b_sub_sem,
                b_mw_relation,
                b_mind,
                b_attitude_1,
                b_attitude_2,
                b_compare,
                b_matching,
                b_train_phrase,
                max_output,
                b_loss,
                b_attitude_c,
                b_phrase_c,
                b_special_c,
                b_overall_c,
            )
            assert all(
                tensor.shape[0] == b_train_phrase.shape[0]
                for tensor in all_compare_tensors
            )

            for (
                s_main_sem,
                s_sub_sem,
                s_mw_relation,
                s_mind,
                s_attitude_1,
                s_attitude_2,
                s_compare,
                s_matching,
                s_train_phrase,
                s_output,
                s_loss,
                s_attitude_c,
                s_phrase_c,
                s_special_c,
                s_overall_c,
            ) in zip(*all_compare_tensors):
                writer.writerow(
                    {
                        "batch_no": i,
                        "main_value": s_main_sem.item(),
                        "sub_value": s_sub_sem.item(),
                        "mind_world_relation": s_mw_relation.item(),
                        "mind_codes": tensor_to_str(s_mind),
                        "attitude_1": s_attitude_1.item(),
                        "attitude_2": s_attitude_2.item(),
                        "compare_codes": tensor_to_str(s_compare),
                        "matching": s_matching.item(),
                        "train_codes": tensor_to_str(s_train_phrase),
                        "output": tensor_to_str(s_output),
                        "loss": tensor_to_str(s_loss),
                        "attitude_correctness": s_attitude_c.item(),
                        "phrase_correctness": s_phrase_c.item(),
                        "special_correctness": s_special_c.item(),
                        "overall_correctness": s_overall_c.item(),
                    }
                )
