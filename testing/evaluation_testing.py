import unittest
from unittest import TestCase

import torch
from torch import LongTensor, BoolTensor

from evaluation.correctness import check_correctness


class CorrectnessTest(TestCase):
    def setUp(self):
        self.b_output = LongTensor([
            [48, 60, 31, 6, 51, 3, 40, 24, 41, 30],
            [48, 52, 32, 7, 52, 3, 40, 24, 41, 30],
            [52, 60, 31, 6, 51, 3, 40, 24, 41, 48],
            [48, 52, 31, 6, 51, 3, 40, 24, 41, 30]
        ])

        self.b_train_phrase = LongTensor([
            [48, 60, 31, 6, 51, 3, 40, 24, 41, 30],
            [48, 52, 31, 6, 51, 3, 40, 24, 41, 30],
            [48, 60, 31, 6, 51, 3, 40, 24, 41, 30],
            [48, 60, 31, 6, 51, 3, 40, 24, 41, 30]
        ])

        self.b_compare = LongTensor([
            [31, 6, 51, 3, 40, 24, 41],
            [31, 6, 51, 3, 40, 24, 41],
            [31, 6, 51, 3, 40, 24, 41],
            [31, 6, 51, 3, 40, 24, 41]
        ])

        self.b_matching = LongTensor([
            [1],
            [1],
            [1],
            [1]
        ])

        self.b_attitude_1 = LongTensor([
            [60],
            [52],
            [52],
            [60]
        ])

        self.b_attitude_2 = LongTensor([
            [63],
            [63],
            [60],
            [63]
        ])

        self.b_attitude_c = BoolTensor([
            True,
            True,
            True,
            False
        ])
        self.b_phrase_c = BoolTensor([
            True,
            False,
            True,
            True
        ])
        self.b_special_c = BoolTensor([
            True,
            True,
            False,
            True
        ])
        self.b_overall_c = BoolTensor([
            True,
            False,
            False,
            False
        ])


    def test_correctness_check(self):
        b_attitude_c, b_phrase_c, b_special_c, b_overall_c = check_correctness(
                self.b_output,
                self.b_train_phrase,
                self.b_compare,
                self.b_matching,
                self.b_attitude_1,
                self.b_attitude_2
            )

        self.assertTrue(torch.eq(b_attitude_c, self.b_attitude_c).all())
        self.assertTrue(torch.eq(b_phrase_c, self.b_phrase_c).all())
        self.assertTrue(torch.eq(b_special_c, self.b_special_c).all())
        self.assertTrue(torch.eq(b_overall_c, self.b_overall_c).all())
        self.assertFalse(torch.eq(b_attitude_c, b_phrase_c).all())


if __name__ == '__main__':
    unittest.main()
