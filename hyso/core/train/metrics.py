from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import math

import torch


class Metric:
    def reset(self) -> None:
        raise NotImplementedError

    def compute(self) -> Any:
        raise NotImplementedError


class AverageMetric(Metric):
    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def update(self, value: Union[float, torch.Tensor], weight: float = 1.0) -> None:
        if torch.is_tensor(value):
            v = float(value.detach().item())
        else:
            v = float(value)
        w = float(weight)
        self.total += v * w
        self.weight += w

    def compute(self) -> float:
        if self.weight <= 0.0:
            return 0.0
        return float(self.total / self.weight)


class PerplexityMetric(Metric):
    def __init__(self) -> None:
        self.avg_loss = AverageMetric()

    def reset(self) -> None:
        self.avg_loss.reset()

    def update(self, loss: Union[float, torch.Tensor], weight: float = 1.0) -> None:
        self.avg_loss.update(loss, weight=weight)

    def compute(self) -> Dict[str, float]:
        avg = float(self.avg_loss.compute())
        if not math.isfinite(avg):
            return {"perplexity": float("inf")}
        return {"perplexity": float(math.exp(avg))}



class TokenAccuracyMetric(Metric):
    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = int(ignore_index)
        self.correct = 0
        self.total = 0

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels != self.ignore_index
            if mask.numel() == 0:
                return
            matched = (preds == labels) & mask
            self.correct += int(matched.sum().item())
            self.total += int(mask.sum().item())

    def compute(self) -> Dict[str, float]:
        if self.total == 0:
            return {"token_accuracy": 0.0}
        return {"token_accuracy": float(self.correct / self.total)}


class BLEUMetric(Metric):
    def __init__(self, max_order: int = 4, smooth: bool = True) -> None:
        self.max_order = int(max_order)
        self.smooth = bool(smooth)
        self.matches_by_order = [0] * self.max_order
        self.possible_matches_by_order = [0] * self.max_order
        self.ref_length = 0
        self.pred_length = 0

    def reset(self) -> None:
        self.matches_by_order = [0] * self.max_order
        self.possible_matches_by_order = [0] * self.max_order
        self.ref_length = 0
        self.pred_length = 0

    def _count_ngrams(self, tokens: List[str], n: int) -> Dict[tuple, int]:
        counts: Dict[tuple, int] = {}
        if len(tokens) < n:
            return counts
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i : i + n])
            counts[ng] = counts.get(ng, 0) + 1
        return counts

    def update(self, preds: List[str], refs: List[str]) -> None:
        for p, r in zip(preds, refs):
            p_tokens = p.split()
            r_tokens = r.split()
            self.pred_length += len(p_tokens)
            self.ref_length += len(r_tokens)
            for order in range(1, self.max_order + 1):
                pred_ngrams = self._count_ngrams(p_tokens, order)
                ref_ngrams = self._count_ngrams(r_tokens, order)
                for ng, c in pred_ngrams.items():
                    self.possible_matches_by_order[order - 1] += c
                    if ng in ref_ngrams:
                        self.matches_by_order[order - 1] += min(c, ref_ngrams[ng])

    def compute(self) -> Dict[str, float]:
        if self.pred_length == 0 or self.ref_length == 0:
            return {"bleu": 0.0}
        precisions: List[float] = []
        for i in range(self.max_order):
            if self.possible_matches_by_order[i] == 0:
                if self.smooth:
                    precisions.append(1.0)
                else:
                    precisions.append(0.0)
            else:
                precisions.append(
                    self.matches_by_order[i] / max(1, self.possible_matches_by_order[i])
                )
        if min(precisions) <= 0.0:
            geo_mean = 0.0
        else:
            geo_mean = math.exp(
                sum(math.log(p) for p in precisions) / self.max_order
            )
        ratio = float(self.pred_length) / float(self.ref_length)
        if ratio > 1.0:
            bp = 1.0
        else:
            bp = math.exp(1.0 - 1.0 / max(ratio, 1e-8))
        bleu = bp * geo_mean
        return {"bleu": float(bleu)}


class LLMetrics:
    def __init__(
        self,
        bleu: bool = False,
        perplexity: bool = True,
        token_accuracy: bool = True,
        token_ignore_index: int = -100,
    ) -> None:
        self.loss_meter = AverageMetric()
        self.perplexity = PerplexityMetric() if perplexity else None
        self.token_accuracy = (
            TokenAccuracyMetric(ignore_index=token_ignore_index)
            if token_accuracy
            else None
        )
        self.bleu = BLEUMetric() if bleu else None
        self.use_bleu = self.bleu is not None

    def reset(self) -> None:
        self.loss_meter.reset()
        if self.perplexity is not None:
            self.perplexity.reset()
        if self.token_accuracy is not None:
            self.token_accuracy.reset()
        if self.bleu is not None:
            self.bleu.reset()

    def update_loss(self, loss: Union[float, torch.Tensor], weight: float = 1.0) -> None:
        self.loss_meter.update(loss, weight=weight)
        if self.perplexity is not None:
            self.perplexity.update(loss, weight=weight)

    def update_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if self.token_accuracy is not None:
            self.token_accuracy.update(logits, labels)

    def update_texts(self, preds: List[str], refs: List[str]) -> None:
        if self.bleu is not None:
            self.bleu.update(preds, refs)

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        loss_val = float(self.loss_meter.compute())
        out["loss"] = loss_val
        if self.perplexity is not None:
            out.update(self.perplexity.compute())
        if self.token_accuracy is not None:
            out.update(self.token_accuracy.compute())
        if self.bleu is not None:
            out.update(self.bleu.compute())
        return out


__all__ = [
    "Metric",
    "AverageMetric",
    "PerplexityMetric",
    "TokenAccuracyMetric",
    "BLEUMetric",
    "LLMetrics",
]
