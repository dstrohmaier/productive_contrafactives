import torch
from torch import Tensor


def check_correctness(
        b_output: Tensor,
        b_train_phrase: Tensor,
        b_compare: Tensor,
        b_matching: Tensor,
        b_attitude_1: Tensor,
        b_attitude_2: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:

    b_output_attitude = b_output[:, 1]
    b_attitude_1 = b_attitude_1.squeeze()
    b_attitude_2 = b_attitude_2.squeeze()
    attitude_1_eq: Tensor = torch.eq(b_output_attitude, b_attitude_1)
    attitude_2_eq: Tensor = torch.eq(b_output_attitude, b_attitude_2)

    attitude_correctness = attitude_1_eq.logical_or(attitude_2_eq)

    b_output_phrase = b_output[:, 2:-1]
    # getting rid of [START], [STOP], and attitude verb token

    b_matching = b_matching.squeeze()
    phrase_eq = torch.eq(b_output_phrase, b_compare).all(dim=1)
    phrase_correctness = phrase_eq.logical_xor(~b_matching.bool())

    b_output_special = b_output[:, (0, -1)]
    b_train_special = b_train_phrase[:, (0, -1)]
    # getting only [START], [STOP], and attitude verb token

    special_correctness = torch.eq(b_output_special, b_train_special).all(dim=1)

    overall_correctness = phrase_correctness.logical_and(attitude_correctness)
    overall_correctness = overall_correctness.logical_and(special_correctness)
    return attitude_correctness, phrase_correctness, special_correctness, overall_correctness
