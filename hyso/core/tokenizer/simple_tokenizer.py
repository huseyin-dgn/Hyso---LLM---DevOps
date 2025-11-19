from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable, Union
from collections import OrderedDict
import json
import unicodedata
import re
import torch


@dataclass
class SimpleSpecial:
    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    UNK: int = 3
    OFFSET: int = 4


class SimpleTokenizer:
    VERSION = 2

    def __init__(
        self,
        *,
        lowercase: bool = False,
        normalize: Optional[str] = "NFKC",
        cache_size: int = 0,
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
        split_mode: str = "basic",
        keep_punct: bool = True,
        punct_chars: Optional[str] = None,
    ) -> None:
        self.special = SimpleSpecial()
        self.lowercase = lowercase
        self.normalize = normalize
        self.min_freq = max(1, int(min_freq))
        self.max_vocab_size = max_vocab_size if max_vocab_size is None else int(max_vocab_size)
        self.split_mode = split_mode
        self.keep_punct = bool(keep_punct)

        self.tokens: List[str] = []
        self.token_to_id: Dict[str, int] = {}

        self.cache_size = max(0, int(cache_size))
        self._cache: "OrderedDict[str, List[int]]" = OrderedDict()

        self._re_basic = re.compile(r"\w+|[^\w\s]", re.UNICODE)
        self._re_ws = re.compile(r"\S+", re.UNICODE)

        default_punct = ".,!?;:()[]{}<>\"'«»“”‘’…-—–/%"
        self._punct_set = set(punct_chars) if punct_chars is not None else set(default_punct)

    def _normalize_text(self, s: str) -> str:
        if self.lowercase:
            s = s.lower()
        if self.normalize:
            s = unicodedata.normalize(self.normalize, s)
        return s

    def _is_all_punct(self, token: str) -> bool:
        if not token:
            return False
        return all(ch in self._punct_set for ch in token)

    def _tokenize_with_spans(self, s: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        tokens: List[str] = []
        spans: List[Tuple[int, int]] = []

        if self.split_mode == "char":
            for idx, ch in enumerate(s):
                if ch.isspace():
                    continue
                if not self.keep_punct and self._is_all_punct(ch):
                    continue
                tokens.append(ch)
                spans.append((idx, idx + 1))
            return tokens, spans

        if self.split_mode == "whitespace":
            for m in self._re_ws.finditer(s):
                tok = m.group(0)
                if not self.keep_punct and self._is_all_punct(tok):
                    continue
                tokens.append(tok)
                spans.append(m.span())
            return tokens, spans

        for m in self._re_basic.finditer(s):
            tok = m.group(0)
            if not self.keep_punct and self._is_all_punct(tok):
                continue
            tokens.append(tok)
            spans.append(m.span())
        return tokens, spans

    def _detokenize(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        out = tokens[0]
        for t in tokens[1:]:
            if self._is_all_punct(t):
                out += t
            else:
                out += " " + t
        return out

    def _build_vocab_from_counter(self, freq) -> None:
        items = [(tok, c) for tok, c in freq.items() if c >= self.min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))

        if self.max_vocab_size is not None:
            items = items[: self.max_vocab_size]

        self.tokens = [tok for tok, _ in items]
        self.token_to_id.clear()
        for idx, tok in enumerate(self.tokens):
            _id = self.special.OFFSET + idx
            self.token_to_id[tok] = _id

        self._cache_clear()

    def fit(self, texts: List[str]) -> None:
        from collections import Counter

        freq = Counter()
        for t in texts:
            norm = self._normalize_text(t)
            toks, _ = self._tokenize_with_spans(norm)
            freq.update(toks)

        self._build_vocab_from_counter(freq)

    def fit_iter(self, text_iter: Iterable[str]) -> None:
        from collections import Counter

        freq = Counter()
        for line in text_iter:
            norm = self._normalize_text(line)
            toks, _ = self._tokenize_with_spans(norm)
            freq.update(toks)

        self._build_vocab_from_counter(freq)

    def _maybe_cache_get(self, key: str) -> Optional[List[int]]:
        if self.cache_size == 0:
            return None
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    def _maybe_cache_put(self, key: str, value: List[int]) -> None:
        if self.cache_size == 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _cache_clear(self) -> None:
        if self.cache_size:
            self._cache.clear()

    def __call__(self, text: str, **kwargs) -> Union[List[int], Tuple[List[int], List[Tuple[int, int]]]]:
        return self.encode(text, **kwargs)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: Optional[int] = None,
        truncation: bool = True,
        return_offsets: bool = False,
        bpe_dropout: float = 0.0,
    ) -> Union[List[int], Tuple[List[int], List[Tuple[int, int]]]]:
        if not self.token_to_id:
            raise RuntimeError("SimpleTokenizer is not fitted; call fit or fit_iter first.")

        norm = self._normalize_text(text)

        use_cache = (
            self.cache_size > 0
            and bpe_dropout == 0.0
            and not add_bos
            and not add_eos
            and max_len is None
            and not return_offsets
            and truncation
        )
        if use_cache:
            cached = self._maybe_cache_get(norm)
            if cached is not None:
                return cached[:]

        toks, spans = self._tokenize_with_spans(norm)
        ids: List[int] = []
        out_spans: List[Tuple[int, int]] = []

        for tok, sp in zip(toks, spans):
            _id = self.token_to_id.get(tok, self.special.UNK)
            ids.append(_id)
            out_spans.append(sp)

        if add_bos:
            ids = [self.special.BOS] + ids
            out_spans = [(-1, -1)] + out_spans
        if add_eos:
            ids = ids + [self.special.EOS]
            out_spans = out_spans + [(-1, -1)]

        if max_len is not None and len(ids) > max_len:
            if truncation:
                ids = ids[:max_len]
                out_spans = out_spans[:max_len]
            else:
                raise ValueError(f"Sequence length {len(ids)} > max_len={max_len}")

        if use_cache:
            self._maybe_cache_put(norm, ids)

        if return_offsets:
            return ids, out_spans
        return ids

    def batch_encode(
        self,
        texts: List[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: Optional[int] = None,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        bpe_dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.long,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not texts:
            ids = torch.empty((0, 0), dtype=dtype, device=device)
            mask = torch.empty((0, 0), dtype=torch.bool, device=device)
            return ids, mask

        seqs: List[List[int]] = [
            self.encode(
                t,
                add_bos=add_bos,
                add_eos=add_eos,
                max_len=max_len,
                truncation=truncation,
                return_offsets=False,
                bpe_dropout=bpe_dropout,
            )
            for t in texts
        ]

        maxL = max((len(s) for s in seqs), default=1)
        if pad_to_multiple_of and maxL % pad_to_multiple_of != 0:
            maxL = ((maxL + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        B = len(seqs)
        ids = torch.full((B, maxL), self.special.PAD, dtype=dtype)
        mask = torch.zeros((B, maxL), dtype=torch.bool)

        for i, s in enumerate(seqs):
            L = min(len(s), maxL)
            if L > 0:
                ids[i, :L] = torch.tensor(s[:L], dtype=dtype)
                mask[i, :L] = True

        if device is not None:
            ids = ids.to(device)
            mask = mask.to(device)

        return ids, mask

    def encode_for_encoder(
        self,
        texts: List[str],
        *,
        max_len: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        bpe_dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.long,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.batch_encode(
            texts,
            add_bos=True,
            add_eos=False,
            max_len=max_len,
            truncation=True,
            pad_to_multiple_of=pad_to_multiple_of,
            bpe_dropout=bpe_dropout,
            device=device,
            dtype=dtype,
        )

    def encode_for_decoder_lm(
        self,
        texts: List[str],
        *,
        max_len: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        bpe_dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.long,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.batch_encode(
            texts,
            add_bos=True,
            add_eos=True,
            max_len=max_len,
            truncation=True,
            pad_to_multiple_of=pad_to_multiple_of,
            bpe_dropout=bpe_dropout,
            device=device,
            dtype=dtype,
        )

    def encode_src_seq2seq(
        self,
        texts: List[str],
        *,
        max_len: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        bpe_dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.long,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.batch_encode(
            texts,
            add_bos=False,
            add_eos=False,
            max_len=max_len,
            truncation=True,
            pad_to_multiple_of=pad_to_multiple_of,
            bpe_dropout=bpe_dropout,
            device=device,
            dtype=dtype,
        )

    def encode_tgt_seq2seq(
        self,
        texts: List[str],
        *,
        max_len: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        bpe_dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.long,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.batch_encode(
            texts,
            add_bos=True,
            add_eos=True,
            max_len=max_len,
            truncation=True,
            pad_to_multiple_of=pad_to_multiple_of,
            bpe_dropout=bpe_dropout,
            device=device,
            dtype=dtype,
        )

    def _ids_to_tokens(self, ids: List[int], *, skip_special: bool) -> List[str]:
        toks: List[str] = []
        for i in ids:
            if i == self.special.PAD:
                if skip_special:
                    continue
                toks.append("<pad>")
                continue
            if i == self.special.BOS:
                if skip_special:
                    continue
                toks.append("<bos>")
                continue
            if i == self.special.EOS:
                if skip_special:
                    break
                toks.append("<eos>")
                break
            if i == self.special.UNK:
                if skip_special:
                    toks.append("?")
                else:
                    toks.append("<unk>")
                continue

            idx = i - self.special.OFFSET
            if 0 <= idx < len(self.tokens):
                toks.append(self.tokens[idx])
            else:
                toks.append("?")
        return toks

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        toks = self._ids_to_tokens(ids, skip_special=skip_special)
        return self._detokenize(toks)

    def batch_decode(
        self,
        batch_ids: Union[torch.Tensor, List[List[int]]],
        skip_special: bool = True,
    ) -> List[str]:
        if isinstance(batch_ids, torch.Tensor):
            return [self.decode(row, skip_special=skip_special) for row in batch_ids]
        return [self.decode(x, skip_special=skip_special) for x in batch_ids]

    @property
    def pad_id(self) -> int:
        return self.special.PAD

    @property
    def bos_id(self) -> int:
        return self.special.BOS

    @property
    def eos_id(self) -> int:
        return self.special.EOS

    @property
    def unk_id(self) -> int:
        return self.special.UNK

    @property
    def vocab_size(self) -> int:
        return self.special.OFFSET + len(self.tokens)

    def __len__(self) -> int:
        return self.vocab_size

    def save(self, path: str) -> None:
        obj = {
            "version": self.VERSION,
            "lowercase": self.lowercase,
            "normalize": self.normalize,
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
            "split_mode": self.split_mode,
            "keep_punct": self.keep_punct,
            "punct_chars": "".join(sorted(self._punct_set)),
            "special": {
                "PAD": self.special.PAD,
                "BOS": self.special.BOS,
                "EOS": self.special.EOS,
                "UNK": self.special.UNK,
                "OFFSET": self.special.OFFSET,
            },
            "tokens": self.tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        ver = obj.get("version", 1)
        if ver not in {1, 2}:
            raise ValueError(f"Unsupported SimpleTokenizer version: {ver}")

        tok = cls(
            lowercase=obj.get("lowercase", False),
            normalize=obj.get("normalize", "NFKC"),
            cache_size=0,
            min_freq=obj.get("min_freq", 1),
            max_vocab_size=obj.get("max_vocab_size", None),
            split_mode=obj.get("split_mode", "basic"),
            keep_punct=obj.get("keep_punct", True),
            punct_chars=obj.get("punct_chars", None),
        )
        sp = obj.get("special", {})
        tok.special = SimpleSpecial(
            PAD=sp.get("PAD", 0),
            BOS=sp.get("BOS", 1),
            EOS=sp.get("EOS", 2),
            UNK=sp.get("UNK", 3),
            OFFSET=sp.get("OFFSET", 4),
        )

        tok.tokens = obj.get("tokens", [])
        tok.token_to_id.clear()
        for idx, t in enumerate(tok.tokens):
            _id = tok.special.OFFSET + idx
            tok.token_to_id[t] = _id

        return tok

    def export_vocab(self) -> Dict[int, str]:
        vocab: Dict[int, str] = {
            self.special.PAD: "<pad>",
            self.special.BOS: "<bos>",
            self.special.EOS: "<eos>",
            self.special.UNK: "<unk>",
        }
        for idx, t in enumerate(self.tokens):
            _id = self.special.OFFSET + idx
            vocab[_id] = t
        return vocab
    
# from simple_tokenizer import SimpleTokenizer

# tok = SimpleTokenizer(
#     lowercase=True,
#     normalize="NFKC",
#     min_freq=2,
#     max_vocab_size=20000,
#     split_mode="basic",     # "basic" | "whitespace" | "char"
#     keep_punct=True,        # noktalama da token olsun mu
#     cache_size=10000,
# )
# train_texts = [
#     "Merhaba dünya!",
#     "Bugün Hyso LLM ile tokenizer yazıyorum.",
#     "Merhaba Hyso.",
# ]

# tok.fit(train_texts)

# print("Vocab size:", tok.vocab_size)
# print("PAD/BOS/EOS/UNK:", tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id)
