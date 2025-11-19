from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable, Union
from collections import OrderedDict
import json
import unicodedata
import random

import torch


@dataclass
class Special:
    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    UNK: int = 3
    OFFSET: int = 4  # bytes: ids 4..259 -> 0..255


class HysoBPETokenizer:
    """
    Byte-level BPE tokenizer (TR/ENG ayırmadan tüm UTF-8 text için).
    - Özel ID'ler: PAD=0, BOS=1, EOS=2, UNK=3, OFFSET=4
    - Base vocab: 256 byte tokenı (OFFSET + 0..255)
    - BPE merge'leri: fit()/fit_iter() ile öğrenilir
    - batch_encode -> (ids[B,T], mask[B,T])  (mask: 1=token, 0=pad)

    Önerilen kullanım:
    - HysoEncoder:  encode_for_encoder(...)
    - HysoDecoder:  encode_for_decoder_lm(...)
    - HysoLLM src:  encode_src_seq2seq(...)
    - HysoLLM tgt:  encode_tgt_seq2seq(...)
    """

    VERSION = 3

    def __init__(
        self,
        *,
        lowercase: bool = False,
        normalize: Optional[str] = "NFKC",
        use_space_sentinel: bool = True,
        sentinel_char: str = "▁",
        use_newline_sentinel: bool = True,
        newline_sentinel: str = "⏎",
        cache_size: int = 0,
    ) -> None:
        # config
        self.special = Special()
        self.lowercase = lowercase
        self.normalize = normalize
        self.use_space_sentinel = use_space_sentinel
        self.sentinel_char = sentinel_char
        self.use_newline_sentinel = use_newline_sentinel
        self.newline_sentinel = newline_sentinel

        # BPE tabloları
        self.pair2id: Dict[Tuple[int, int], int] = {}
        self.rank: Dict[Tuple[int, int], int] = {}
        self.merges_seq: List[Tuple[int, int, int]] = []  # (left_id, right_id, new_id)

        # id <-> bytes mapping
        self.id_to_bytes: Dict[int, List[int]] = {}
        self.bytes_to_id: Dict[Tuple[int, ...], int] = {}

        # base 256 byte tokenları: OFFSET + b
        for b in range(256):
            _id = self.special.OFFSET + b
            self.id_to_bytes[_id] = [b]
            self.bytes_to_id[(b,)] = _id
        self.next_id = self.special.OFFSET + 256

        # encode cache (opsiyonel)
        self.cache_size = max(0, int(cache_size))
        self._cache: "OrderedDict[str, List[int]]" = OrderedDict()

    # ============================================================
    # Yardımcı: normalize / denormalize
    # ============================================================

    def _normalize_text(self, s: str) -> str:
        if self.lowercase:
            s = s.lower()
        if self.normalize:
            s = unicodedata.normalize(self.normalize, s)
        if self.use_space_sentinel:
            s = s.replace(" ", self.sentinel_char)
        if self.use_newline_sentinel:
            s = s.replace("\n", self.newline_sentinel)
        return s

    def _denormalize_text(self, s: str) -> str:
        if self.use_newline_sentinel:
            s = s.replace(self.newline_sentinel, "\n")
        if self.use_space_sentinel:
            s = s.replace(self.sentinel_char, " ")
        return s

    def _text_to_byte_ids(self, s: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Text -> UTF-8 byte dizisi -> base token ID'leri
        - ids: [OFFSET + byte]
        - spans: her byte için (start,end) offset
        """
        norm = self._normalize_text(s)
        b = norm.encode("utf-8", errors="strict")
        ids = [self.special.OFFSET + x for x in b]
        spans = [(i, i + 1) for i in range(len(b))]
        return ids, spans

    def _id_to_bytes(self, i: int) -> List[int]:
        seq = self.id_to_bytes.get(i)
        if seq is None:
            raise KeyError(f"id {i} has no byte mapping")
        return seq

    # ============================================================
    # BPE eğitim (pair frekansı, merge, vb.)
    # ============================================================

    def _count_pairs(self, sequences: List[List[int]]) -> Dict[Tuple[int, int], int]:
        pf: Dict[Tuple[int, int], int] = {}
        for seq in sequences:
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                # özel ID’leri merge etme (PAD/BOS/EOS/UNK)
                if a < self.special.OFFSET or b < self.special.OFFSET:
                    continue
                pf[(a, b)] = pf.get((a, b), 0) + 1
        return pf

    def _best_pair(
        self,
        pair_freq: Dict[Tuple[int, int], int],
        min_pair_freq: int,
    ) -> Optional[Tuple[int, int]]:
        cand = [p for p, c in pair_freq.items() if c >= min_pair_freq]
        if not cand:
            return None
        # deterministik seçim: (count, sonra sözlük sırası)
        return max(cand, key=lambda p: (pair_freq[p], p))

    def _fit_on_sequences(
        self,
        sequences: List[List[int]],
        min_pair_freq: int,
        max_merges: int,
    ) -> None:
        merges_done = 0
        while merges_done < max_merges:
            pf = self._count_pairs(sequences)
            best = self._best_pair(pf, min_pair_freq)
            if best is None:
                break
            a, b = best
            assert a >= self.special.OFFSET and b >= self.special.OFFSET, "Merge özel ID’lere taşamaz."
            new_id = self.next_id
            self.next_id += 1

            # korpusu (a,b) -> new_id yap
            for i in range(len(sequences)):
                seq = sequences[i]
                if len(seq) < 2:
                    continue
                out: List[int] = []
                j = 0
                while j < len(seq):
                    if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                        out.append(new_id)
                        j += 2
                    else:
                        out.append(seq[j])
                        j += 1
                sequences[i] = out

            # BPE tablolarını güncelle
            self.pair2id[(a, b)] = new_id
            self.rank[(a, b)] = merges_done
            self.merges_seq.append((a, b, new_id))

            # byte açılımı
            self.id_to_bytes[new_id] = self._id_to_bytes(a) + self._id_to_bytes(b)
            self.bytes_to_id[tuple(self.id_to_bytes[new_id])] = new_id

            merges_done += 1

    def fit(
        self,
        texts: List[str],
        *,
        target_vocab_size: int = 16384,
        min_pair_freq: int = 3,
        max_merges: Optional[int] = None,
    ) -> None:
        """
        Küçük/orta boyutlu corpus için BPE eğitimi.
        - texts: ham string listesi
        - target_vocab_size: toplam vocab hedefi (special + 256 byte + merges)
        - min_pair_freq: bir pair'in en az kaç kez geçmesi gerektiği
        - max_merges: override; None ise target_vocab'ten türetilir
        """
        if max_merges is None:
            max_merges = max(0, target_vocab_size - (self.special.OFFSET + 256))
        sequences: List[List[int]] = []
        for t in texts:
            ids, _ = self._text_to_byte_ids(t)
            if ids:
                sequences.append(ids)
        if not sequences:
            self._cache_clear()
            return
        self._fit_on_sequences(sequences, min_pair_freq, max_merges)
        self._cache_clear()

    def fit_iter(
        self,
        text_iter: Iterable[str],
        *,
        target_vocab_size: int = 16384,
        min_pair_freq: int = 3,
        max_merges: Optional[int] = None,
        buffer_size: int = 10000,
    ) -> None:
        """
        Büyük corpus için streaming BPE eğitimi.
        - text_iter: satır satır / dosya dosya iterator
        - buffer_size: her turda belleğe alınacak satır sayısı
        """
        if max_merges is None:
            max_merges = max(0, target_vocab_size - (self.special.OFFSET + 256))

        buf: List[List[int]] = []
        sequences: List[List[int]] = []
        for line in text_iter:
            ids, _ = self._text_to_byte_ids(line)
            if ids:
                buf.append(ids)
            if len(buf) >= buffer_size:
                sequences.extend(buf)
                buf = []
        if buf:
            sequences.extend(buf)

        if not sequences:
            self._cache_clear()
            return

        self._fit_on_sequences(sequences, min_pair_freq, max_merges)
        self._cache_clear()

    # ============================================================
    # Encode tarafı: merge pass + cache
    # ============================================================

    def _merge_pass_ranked(
        self,
        seq: List[int],
        spans: List[Tuple[int, int]],
        bpe_dropout: float,
    ) -> Tuple[List[int], List[Tuple[int, int]], bool]:
        """
        Left-to-right; mevcut (a,b) varsa ve dropout değilse merge eder.
        Aynı anda (i,i+1) ve (i+1,i+2) seçenekleri varsa rank'ı küçük olana öncelik verir.
        """
        if not self.pair2id:
            return seq, spans, False

        changed = False
        out_ids: List[int] = []
        out_sp: List[Tuple[int, int]] = []
        i = 0

        while i < len(seq):
            if i < len(seq) - 1:
                pair = (seq[i], seq[i + 1])
                nid = self.pair2id.get(pair)
                if nid is not None and not (bpe_dropout > 0.0 and random.random() < bpe_dropout):
                    r_cur = self.rank[pair]
                    # lookahead: sağdaki çift daha iyi rank ise şimdilik kaydır
                    prefer_right = False
                    if i < len(seq) - 2:
                        right_pair = (seq[i + 1], seq[i + 2])
                        nid_r = self.pair2id.get(right_pair)
                        if (
                            nid_r is not None
                            and self.rank[right_pair] < r_cur
                            and not (bpe_dropout > 0.0 and random.random() < bpe_dropout)
                        ):
                            prefer_right = True
                    if not prefer_right:
                        # merge current
                        out_ids.append(nid)
                        out_sp.append((spans[i][0], spans[i + 1][1]))
                        i += 2
                        changed = True
                        continue
            # no merge
            out_ids.append(seq[i])
            out_sp.append(spans[i])
            i += 1

        return out_ids, out_sp, changed

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

    # ============================================================
    # Public encode / batch_encode
    # ============================================================

    def __call__(self, text: str, **kwargs) -> Union[List[int], Tuple[List[int], List[Tuple[int, int]]]]:
        # tok("merhaba") -> encode("merhaba")
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
        """
        Tek cümle encode.
        - add_bos/add_eos: BOS/EOS ekle
        - max_len: kesme limiti
        - truncation=False ise over-length durumda hata atar
        - return_offsets=True ise (ids, spans) döner, spans: (byte_start, byte_end)
        - bpe_dropout > 0.0 -> data augmentation için random merge skip
        """
        # Cache sadece saf, deterministik, offsetsiz durum için devrede
        use_cache = (
            self.cache_size > 0
            and bpe_dropout == 0.0
            and not add_bos
            and not add_eos
            and max_len is None
            and return_offsets is False
            and truncation
        )
        if use_cache:
            norm_key = self._normalize_text(text)
            cached = self._maybe_cache_get(norm_key)
            if cached is not None:
                return cached[:]  # kopya

        base_ids, base_spans = self._text_to_byte_ids(text)
        seq, spans = base_ids, base_spans

        changed = True
        while changed and self.pair2id:
            seq, spans, changed = self._merge_pass_ranked(seq, spans, bpe_dropout)

        if add_bos:
            seq = [self.special.BOS] + seq
            spans = [(-1, -1)] + spans
        if add_eos:
            seq = seq + [self.special.EOS]
            spans = spans + [(-1, -1)]

        if max_len is not None and len(seq) > max_len:
            if truncation:
                seq = seq[:max_len]
                spans = spans[:max_len]
            else:
                raise ValueError(f"Sequence length {len(seq)} > max_len={max_len}")

        if use_cache:
            norm_key = self._normalize_text(text)
            self._maybe_cache_put(norm_key, seq)

        if return_offsets:
            return seq, spans
        return seq

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
        """
        Çoklu cümle encode.
        Dönüş:
        - ids:  [B, T_max]
        - mask: [B, T_max] (1=token, 0=pad)
        """
        if not texts:
            # Boş batch için düzgün shape dön
            ids = torch.empty((0, 0), dtype=dtype, device=device)
            mask = torch.empty((0, 0), dtype=dtype, device=device)
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
        mask = torch.zeros((B, maxL), dtype=dtype)

        for i, s in enumerate(seqs):
            L = min(len(s), maxL)
            if L > 0:
                ids[i, :L] = torch.tensor(s[:L], dtype=dtype)
                mask[i, :L] = 1

        if device is not None:
            ids = ids.to(device)
            mask = mask.to(device)

        return ids, mask

    # ============================================================
    # Model uyumlu helper'lar
    # ============================================================

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
        """
        HysoEncoder için:
        - BOS = pseudo [CLS]
        - EOS yok
        """
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
        """
        HysoDecoder (causal LM) için:
        - BOS + text + EOS
        """
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
        """
        HysoLLM encoder src için:
        - BOS/EOS yok, raw kaynak cümle
        """
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
        """
        HysoLLM decoder tgt için (teacher forcing):
        - BOS + text + EOS
        """
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

    # ============================================================
    # Decode
    # ============================================================

    def _ids_to_text_via_bytes(
        self,
        ids: List[int],
        *,
        skip_special: bool,
    ) -> str:
        out_bytes: List[int] = []
        prefix: List[str] = []

        for i in ids:
            if i == self.special.PAD:
                if skip_special:
                    continue
                prefix.append("<pad>")
                continue
            if i == self.special.BOS:
                if skip_special:
                    continue
                prefix.append("<bos>")
                continue
            if i == self.special.EOS:
                if skip_special:
                    break
                prefix.append("<eos>")
                # EOS gördükten sonrasını da byte'a çevirmiyoruz
                break
            if i == self.special.UNK:
                if skip_special:
                    out_bytes.append(ord("?"))
                else:
                    prefix.append("<unk>")
                continue
            out_bytes.extend(self._id_to_bytes(i))

        text = bytes(out_bytes).decode("utf-8", errors="replace")
        text = self._denormalize_text(text)
        if prefix:
            # debug modunda özel tokenları başa string olarak koy
            return " ".join(prefix) + (" " if text else "") + text
        return text

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._ids_to_text_via_bytes(ids, skip_special=skip_special)

    def batch_decode(
        self,
        batch_ids: Union[torch.Tensor, List[List[int]]],
        skip_special: bool = True,
    ) -> List[str]:
        if isinstance(batch_ids, torch.Tensor):
            return [self.decode(row, skip_special=skip_special) for row in batch_ids]
        return [self.decode(x, skip_special=skip_special) for x in batch_ids]

    # ============================================================
    # Meta özellikler
    # ============================================================

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
        # special + 256 byte + merge sayısı
        return self.special.OFFSET + 256 + len(self.merges_seq)

    def __len__(self) -> int:
        return self.vocab_size

    # ============================================================
    # save / load
    # ============================================================

    def save(self, path: str) -> None:
        """
        Tokenizer'ı JSON olarak kaydet.
        - Sadece BPE tablolarını ve config'i kaydeder.
        """
        obj = {
            "version": self.VERSION,
            "lowercase": self.lowercase,
            "normalize": self.normalize,
            "use_space_sentinel": self.use_space_sentinel,
            "sentinel_char": self.sentinel_char,
            "use_newline_sentinel": self.use_newline_sentinel,
            "newline_sentinel": self.newline_sentinel,
            "special": {
                "PAD": self.special.PAD,
                "BOS": self.special.BOS,
                "EOS": self.special.EOS,
                "UNK": self.special.UNK,
                "OFFSET": self.special.OFFSET,
            },
            "next_id": self.next_id,
            "merges": self.merges_seq,  # (l, r, new_id)
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "HysoBPETokenizer":
        """
        Kaydedilmiş tokenizer'ı JSON'dan yükle.
        """
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        SUPPORTED = {2, 3}
        ver = obj.get("version", 1)
        if ver not in SUPPORTED:
            raise ValueError(f"Unsupported tokenizer version: {ver}")

        tok = cls(
            lowercase=obj.get("lowercase", False),
            normalize=obj.get("normalize", "NFKC"),
            use_space_sentinel=obj.get("use_space_sentinel", True),
            sentinel_char=obj.get("sentinel_char", "▁"),
            use_newline_sentinel=obj.get(
                "use_newline_sentinel",
                True if ver >= 3 else False,
            ),
            newline_sentinel=obj.get("newline_sentinel", "⏎"),
            cache_size=0,
        )
        sp = obj.get("special", {})
        tok.special = Special(
            PAD=sp.get("PAD", 0),
            BOS=sp.get("BOS", 1),
            EOS=sp.get("EOS", 2),
            UNK=sp.get("UNK", 3),
            OFFSET=sp.get("OFFSET", 4),
        )
        tok.next_id = obj.get("next_id", tok.special.OFFSET + 256)

        # base tabloları resetle
        tok.id_to_bytes.clear()
        tok.bytes_to_id.clear()
        for b in range(256):
            _id = tok.special.OFFSET + b
            tok.id_to_bytes[_id] = [b]
            tok.bytes_to_id[(b,)] = _id

        tok.merges_seq = []
        tok.pair2id.clear()
        tok.rank.clear()
        for rank, (l, r, new_id) in enumerate(obj.get("merges", [])):
            assert l >= tok.special.OFFSET and r >= tok.special.OFFSET
            tok.merges_seq.append((l, r, new_id))
            tok.pair2id[(l, r)] = new_id
            tok.rank[(l, r)] = rank
            tok.id_to_bytes[new_id] = tok._id_to_bytes(l) + tok._id_to_bytes(r)
            tok.bytes_to_id[tuple(tok.id_to_bytes[new_id])] = new_id

        return tok
    def export_vocab(self) -> Dict[int, str]:
        def escape(bs: List[int]) -> str:
            return "".join(
                chr(b) if 32 <= b < 127 else f"\\x{b:02x}"
                for b in bs
            )

        vocab = {
            self.special.PAD: "<pad>",
            self.special.BOS: "<bos>",
            self.special.EOS: "<eos>",
            self.special.UNK: "<unk>",
        }
        for i in range(256):
            _id = self.special.OFFSET + i
            vocab[_id] = escape([i])
        for _, _, new_id in self.merges_seq:
            vocab[new_id] = escape(self.id_to_bytes[new_id])
        return vocab

    
# from bpe_tokenizer import HysoBPETokenizer

# texts = [
#     "Merhaba dünya",
#     "Hello world",
#     "Ben Hyso LLM eğitiyorum",
#     # ... büyük bir liste
# ]

# tok = HysoBPETokenizer(
#     lowercase=False,      # istersen True yap
#     normalize="NFKC",     # Unicode normalizasyon
#     cache_size=10000,     # encode cache (opsiyonel)
# )

# tok.fit(
#     texts,
#     target_vocab_size=32000,  # toplam vocab hedefi
#     min_pair_freq=2,          # çok nadir olan pair'leri merge etme
# )

# print("Vocab size:", tok.vocab_size)       # model için vereceğin değer
# print("PAD:", tok.pad_id, "BOS:", tok.bos_id, "EOS:", tok.eos_id)