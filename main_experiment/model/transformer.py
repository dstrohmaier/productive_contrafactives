"""Core Transformer models used for the language generation task"""

import math
import logging

import torch
from torch import nn, Tensor


class PositionEncoder(nn.Module):
    def __init__(self, d_embedding: int, sent_len: int) -> None:
        super().__init__()
        self.logger = logging.getLogger("contrafactives")
        self.logger.info("Setting up PositionEncoder...")

        position = torch.arange(1, sent_len + 1).float().unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_embedding, step=2).float()
            * (-math.log(10000.0) / d_embedding)
        )

        position_encoding = torch.zeros(sent_len, d_embedding).float()
        position_encoding.requires_grad = False

        position_encoding[:, 0::2] = torch.sin(position * div_term)
        if d_embedding % 2 == 0:
            position_encoding[:, 1::2] = torch.cos(position * div_term)
        else:
            position_encoding[:, 1::2] = torch.cos(position * div_term)[
                :, :-1
            ]  # otherwise shape mismatch

        self.register_buffer("position_encoding", position_encoding)

    def forward(self, batch_size: int) -> Tensor:
        return self.position_encoding.repeat(batch_size, 1, 1)
        # assumption that length is always the same


class GenTransformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        input_len: int,
        sentence_len: int,
        d_embedding: int,
        d_hidden: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        n_heads: int,
        p_dropout: float,
        n_segments: int = 4,
        disable_generator: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.logger = logging.getLogger("contrafactives")
        self.logger.info("Setting up GenTransformer...")

        self.target_vocab_size = target_vocab_size

        self.d_embedding = d_embedding
        self.d_hidden = d_hidden
        self.source_embedding = nn.Embedding(source_vocab_size, d_embedding)
        self.target_embedding = nn.Embedding(target_vocab_size, d_embedding)
        self.pos_embedding = PositionEncoder(d_embedding, sentence_len - 1)
        # sentence_len-1 because we ignore start token
        self.seg_embedding = nn.Embedding(n_segments, d_embedding)

        self.input_len = input_len
        self.sentence_len = sentence_len

        self.transformer = nn.Transformer(
            d_model=d_embedding,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_hidden,
            dropout=p_dropout,
        )

        if not disable_generator:
            self.generator = nn.Linear(d_embedding, target_vocab_size)

    def init_weights(self) -> None:
        for p in self.transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        if hasattr(self, "generator"):
            torch.nn.init.zeros_(self.generator.bias.data)
            torch.nn.init.xavier_uniform_(self.generator.weight.data)

    def main_forward(
        self,
        b_main_sem: Tensor,
        b_sub_sem: Tensor,
        b_mind: Tensor,
        b_comparison: Tensor,
        b_lang_in: Tensor,
    ) -> Tensor:
        seg_main_sem = torch.zeros_like(b_main_sem)
        seg_sub_sem = torch.ones_like(b_sub_sem)
        seg_mind = torch.full_like(b_mind, 2)
        seg_comparison = torch.full_like(b_comparison, 3)

        # self.logger.info('Permuted seg_sem.shape %s', seg_sem.shape)
        # self.logger.info('Permuted seg_mind.shape %s', seg_mind.shape)
        # self.logger.info('Permuted seg_comparison.shape %s', seg_comparison.shape)

        seg_cat = torch.cat(
            (seg_main_sem, seg_sub_sem, seg_mind, seg_comparison), dim=1
        )

        b_cat = torch.cat((b_main_sem, b_sub_sem, b_mind, b_comparison), dim=1)
        b_source = self.source_embedding(b_cat) * math.sqrt(self.d_embedding)
        b_source += self.seg_embedding(seg_cat)

        b_target = self.target_embedding(b_lang_in) * math.sqrt(self.d_embedding)
        b_target_pos = self.pos_embedding(b_lang_in.shape[0])

        assert b_target_pos.shape[0] == b_target.shape[0]
        assert b_target_pos.shape[2] == b_target.shape[2]

        b_target += b_target_pos[:, : b_target.shape[1], :]
        # restriction on dim 1/seq_len for testing, when inferring one token at a time

        b_source, b_target = (b_source.permute(1, 0, 2), b_target.permute(1, 0, 2))
        # transformer_encoder takes batch in dim 1
        # batch x embedding x sequence -> embedding x batch x sequence
        # self.logger.info('Permuted b_source.shape: %s', b_source.shape)
        # self.logger.info('Permuted b_source.shape: %s', b_target.shape)

        source_mask, target_mask = self.create_masks(b_source, b_target)
        outputs = self.transformer(
            src=b_source, tgt=b_target, src_mask=source_mask, tgt_mask=target_mask
        )

        outputs = outputs.permute(1, 0, 2)

        # self.logger.info("outputs.shape: %s", outputs.shape)

        assert outputs.shape[0] == b_comparison.shape[0]
        assert outputs.shape[1] <= self.sentence_len - 1
        # -1 because we ignore [START], <= because it can be smaller in autregressive testing
        assert (
            outputs.shape[2] == self.d_embedding
        ), f"{outputs.shape[2]} != {self.d_embedding}"
        # sequence x batch x d_embedding -> batch x sequence-1 x d_embedding

        return outputs

    def forward(self, *args, **kwargs) -> Tensor:
        outputs = self.main_forward(*args, **kwargs)
        outputs = self.generator(outputs)
        # self.logger.info("Permuted output.shape: %s", outputs.shape)
        assert outputs.shape[2] == self.target_vocab_size
        return outputs

    @staticmethod
    def create_masks(source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        src_seq_len = source.shape[0]
        tgt_seq_len = target.shape[0]

        source_mask = (
            torch.zeros((src_seq_len, src_seq_len)).type(torch.bool).to(source.device)
        )

        target_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len))) == 1
        target_mask = target_mask.transpose(0, 1).float()
        target_mask = target_mask.masked_fill(target_mask == 0, float("-inf"))
        target_mask = target_mask.masked_fill(target_mask == 1, float(0.0)).to(
            target.device
        )

        return source_mask, target_mask


class RestrictedGenTransformer(GenTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, disable_generator=True, **kwargs)
        self.encoding_cfg = kwargs["encoding_cfg"]

        self.output_layers = nn.ModuleList()

        for token_map in self.encoding_cfg["maps_in_order"]:
            self.output_layers.append(
                nn.Linear(self.d_embedding, len(self.encoding_cfg[token_map]))
            )

    def init_weights(self) -> None:
        super().init_weights()
        for o_layer in self.output_layers:
            torch.nn.init.zeros_(o_layer.bias.data)
            torch.nn.init.xavier_uniform_(o_layer.weight.data)

    def project(
        self,
        restricted_output: Tensor,
        token_map: dict[str, int],
        disallowed_val: float = -999.0,
    ) -> Tensor:
        project_t = torch.full(
            (restricted_output.shape[0], self.target_vocab_size), disallowed_val
        )
        project_t = project_t.to(restricted_output.device)
        for i, val in enumerate(token_map.values()):
            project_t[:, val] = restricted_output[:, i]

        return project_t

    def forward(self, *args, **kwargs) -> Tensor:
        outputs = self.main_forward(*args, **kwargs)
        # batch x sequence x d_embeddings

        all_outputs = []

        assert outputs.shape[1] <= len(self.output_layers)

        for i, o_layer in enumerate(self.output_layers):
            if i == outputs.shape[1]:
                break
                # should only happen in the case of testing
                # when we feed in one token at a time

            restricted_output = o_layer(outputs[:, i, :])

            token_map = self.encoding_cfg[self.encoding_cfg["maps_in_order"][i]]
            projected_output = self.project(restricted_output, token_map)

            all_outputs.append(projected_output)

        outputs = torch.stack(all_outputs, dim=1)
        assert outputs.shape[2] == self.target_vocab_size
        return outputs
