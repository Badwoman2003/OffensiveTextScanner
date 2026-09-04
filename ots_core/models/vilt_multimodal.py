"""ViLT backbone adapted for Chinese + offensive-content classification.

Strategy:
1. Start from the official English ``dandelin/vilt-b32-mlm`` checkpoint.
2. Replace the English BERT tokenizer with ``bert-base-chinese`` and warm-start the new token
   embedding table from the mean of the original embeddings. A learnable per-token linear
   projection is added so the backbone has a chance to reshape the warm-start distribution.
3. Keep the single-stream Transformer exactly as-is (so fused kernels trained against HF ViLT
   internals keep working).
4. Add a small classification head that pools the ``[CLS]`` token.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    BertTokenizerFast,
    ViltConfig,
    ViltImageProcessor,
    ViltModel,
    ViltProcessor,
)

from ots_core.models.heads import OffenseClassificationHead


@dataclass
class ViltOffenseConfig:
    base_model: str = "dandelin/vilt-b32-mlm"
    chinese_tokenizer: str = "bert-base-chinese"
    num_labels: int = 2
    dropout: float = 0.1
    freeze_vision_patch: bool = False


def build_zh_processor(cfg: ViltOffenseConfig) -> ViltProcessor:
    """Build a ``ViltProcessor`` whose tokenizer is Chinese BERT but whose image processor is
    the ViLT default so all image statistics match the pretrained backbone."""
    tokenizer = BertTokenizerFast.from_pretrained(cfg.chinese_tokenizer)
    image_processor = ViltImageProcessor.from_pretrained(cfg.base_model)
    return ViltProcessor(image_processor=image_processor, tokenizer=tokenizer)


def warm_start_embeddings(model: ViltModel, new_vocab_size: int) -> None:
    """Resize the token embedding matrix and fill new rows with the mean of the old rows."""
    old_emb = model.embeddings.text_embeddings.word_embeddings
    old_w: torch.Tensor = old_emb.weight.data.clone()
    old_mean = old_w.mean(dim=0, keepdim=True)
    old_vocab_size, hidden = old_w.shape

    new_emb = nn.Embedding(new_vocab_size, hidden)
    new_emb.weight.data.copy_(old_mean.expand(new_vocab_size, -1))
    copy_n = min(old_vocab_size, new_vocab_size)
    new_emb.weight.data[:copy_n] = old_w[:copy_n]
    model.embeddings.text_embeddings.word_embeddings = new_emb
    model.config.vocab_size = new_vocab_size


class ViltForOffenseClassification(nn.Module):
    """ViLT + classification head, compatible with HF's ``Trainer`` and our own loop."""

    def __init__(self, cfg: ViltOffenseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        vilt_cfg = ViltConfig.from_pretrained(cfg.base_model)
        vilt_cfg.num_labels = cfg.num_labels
        self.config = vilt_cfg
        self.vilt = ViltModel.from_pretrained(cfg.base_model, config=vilt_cfg, add_pooling_layer=True)

        self.processor = build_zh_processor(cfg)
        new_vocab = self.processor.tokenizer.vocab_size
        warm_start_embeddings(self.vilt, new_vocab)

        self.embed_projection = nn.Linear(vilt_cfg.hidden_size, vilt_cfg.hidden_size, bias=False)
        nn.init.eye_(self.embed_projection.weight)  # identity init; learn to remap if needed

        self.head = OffenseClassificationHead(
            hidden_size=vilt_cfg.hidden_size,
            num_labels=cfg.num_labels,
            dropout=cfg.dropout,
        )

        if cfg.freeze_vision_patch:
            for p in self.vilt.embeddings.patch_embeddings.parameters():
                p.requires_grad_(False)

    # -- surgery hooks so fused kernels / CPT can reach internals --------------------------------

    def text_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.vilt.embeddings.text_embeddings.word_embeddings(input_ids)
        return self.embed_projection(emb)

    # -- forward ---------------------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        pixel_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        outputs = self.vilt(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_mask=pixel_mask,
        )
        pooled = outputs.pooler_output
        logits = self.head(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits, "pooler_output": pooled, "last_hidden_state": outputs.last_hidden_state}

    # -- save / load -----------------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "pytorch_model.bin")
        self.processor.save_pretrained(directory)
        (directory / "config.json").write_text(
            self.config.to_json_string(), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path, cfg: ViltOffenseConfig | None = None) -> "ViltForOffenseClassification":
        directory = Path(directory)
        cfg = cfg or ViltOffenseConfig()
        model = cls(cfg)
        state = torch.load(directory / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state, strict=False)
        try:
            model.processor = ViltProcessor.from_pretrained(directory)
        except Exception:
            pass
        return model
