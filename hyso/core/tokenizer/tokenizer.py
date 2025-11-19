from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Iterable, Union
from dataclasses import dataclass
from collections import OrderedDict
import json, unicodedata, random
import torch


@dataclass
class Special:
    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    UNK: int = 3
    OFFSET: int = 4 


class HysoTokenizer:


    VERSION = 3

    def __init__(self,
                 lowercase: bool = False,
                 normalize: Optional[str] = "NFKC",
                 use_space_sentinel: bool = True,
                 sentinel_char: str = "▁",
                 use_newline_sentinel: bool = True,
                 newline_sentinel: str = "⏎",
                 cache_size: int = 0):
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
        self.merges_seq: List[Tuple[int, int, int]] = []  # (l, r, new_id)

        # id <-> bytes
        self.id_to_bytes: Dict[int, List[int]] = {}
        self.bytes_to_id: Dict[Tuple[int, ...], int] = {}

        # base 256 byte tokenları
        for b in range(256):
            _id = self.special.OFFSET + b
            self.id_to_bytes[_id] = [b]
            self.bytes_to_id[(b,)] = _id
        self.next_id = self.special.OFFSET + 256

        # opsiyonel encode cache
        self.cache_size = max(0, int(cache_size))
        self._cache: "OrderedDict[str, List[int]]" = OrderedDict()

    # ---------- yardımcı ----------
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
    
    def bad_ids_unicode(self):
        from ...llm.generate import build_bad_ids_unicode
        return build_bad_ids_unicode(self)


    def _text_to_byte_ids(self, s: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        norm = self._normalize_text(s)
        b = norm.encode("utf-8", errors="strict")   # <= ÖNEMLİ
        ids = [self.special.OFFSET + x for x in b]
        spans = [(i, i + 1) for i in range(len(b))]
        return ids, spans


    def _id_to_bytes(self, i: int) -> List[int]:
        seq = self.id_to_bytes.get(i)
        if seq is None:
            raise KeyError(f"id {i} has no byte mapping")
        return seq

    # ---------- eğitim ----------
    def _count_pairs(self, sequences: List[List[int]]) -> Dict[Tuple[int, int], int]:
        pf: Dict[Tuple[int, int], int] = {}
        for seq in sequences:
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                if a < self.special.OFFSET or b < self.special.OFFSET:
                    continue
                pf[(a, b)] = pf.get((a, b), 0) + 1
        return pf

    def _best_pair(self, pair_freq: Dict[Tuple[int, int], int], min_pair_freq: int) -> Optional[Tuple[int, int]]:
        # deterministik bağ: (count, sonra leksik pair)
        cand = [p for p, c in pair_freq.items() if c >= min_pair_freq]
        if not cand: return None
        return max(cand, key=lambda p: (pair_freq[p], p))

    def _fit_on_sequences(self, sequences: List[List[int]], min_pair_freq: int, max_merges: int) -> None:
        merges_done = 0
        while merges_done < max_merges:
            pf = self._count_pairs(sequences)
            best = self._best_pair(pf, min_pair_freq)
            if best is None:
                break
            a, b = best
            assert a >= self.special.OFFSET and b >= self.special.OFFSET, "Merge özel ID’lere taşamaz."
            new_id = self.next_id; self.next_id += 1

            # korpusu (a,b)->new_id yap
            for i in range(len(sequences)):
                seq = sequences[i]
                if len(seq) < 2: continue
                out = []
                j = 0
                while j < len(seq):
                    if j < len(seq) - 1 and seq[j] == a and seq[j+1] == b:
                        out.append(new_id); j += 2
                    else:
                        out.append(seq[j]); j += 1
                sequences[i] = out

            # sözlükleri güncelle
            self.pair2id[(a, b)] = new_id
            self.rank[(a, b)] = merges_done
            self.merges_seq.append((a, b, new_id))

            # byte açılımı
            self.id_to_bytes[new_id] = self._id_to_bytes(a) + self._id_to_bytes(b)
            self.bytes_to_id[tuple(self.id_to_bytes[new_id])] = new_id

            merges_done += 1

    def fit(self,
            texts: List[str],
            target_vocab_size: int = 16384,
            min_pair_freq: int = 3,
            max_merges: Optional[int] = None) -> None:
        if max_merges is None:
            max_merges = max(0, target_vocab_size - (self.special.OFFSET + 256))
        sequences: List[List[int]] = []
        for t in texts:
            ids, _ = self._text_to_byte_ids(t)
            if ids: sequences.append(ids)
        self._fit_on_sequences(sequences, min_pair_freq, max_merges)
        self._cache_clear()

    def fit_iter(self,
                 text_iter: Iterable[str],
                 target_vocab_size: int = 16384,
                 min_pair_freq: int = 3,
                 max_merges: Optional[int] = None,
                 buffer_size: int = 10000) -> None:
        if max_merges is None:
            max_merges = max(0, target_vocab_size - (self.special.OFFSET + 256))
        buf: List[List[int]] = []
        sequences: List[List[int]] = []
        for line in text_iter:
            ids, _ = self._text_to_byte_ids(line)
            if ids: buf.append(ids)
            if len(buf) >= buffer_size:
                sequences.extend(buf); buf = []
        if buf: sequences.extend(buf)
        self._fit_on_sequences(sequences, min_pair_freq, max_merges)
        self._cache_clear()

    # ---------- encode/merge (rank-öncelikli lookahead + dropout) ----------
    def _merge_pass_ranked(self,
                           seq: List[int],
                           spans: List[Tuple[int, int]],
                           bpe_dropout: float) -> Tuple[List[int], List[Tuple[int, int]], bool]:
        """
        Left-to-right; mevcut (a,b) varsa ve dropout değilse merge eder.
        Aynı anda (i,i+1) ve (i+1,i+2) seçenekleri varsa rank'ı küçük olana öncelik verir (lookahead).
        """
        if not self.pair2id: return seq, spans, False
        changed = False
        out_ids: List[int] = []
        out_sp: List[Tuple[int, int]] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1:
                pair = (seq[i], seq[i+1])
                nid = self.pair2id.get(pair)
                if nid is not None and not (bpe_dropout > 0.0 and random.random() < bpe_dropout):
                    r_cur = self.rank[pair]
                    # lookahead: sağdaki çift daha iyi rank ise şimdilik kaydır
                    prefer_right = False
                    if i < len(seq) - 2:
                        right_pair = (seq[i+1], seq[i+2])
                        nid_r = self.pair2id.get(right_pair)
                        if nid_r is not None and self.rank[right_pair] < r_cur and not (bpe_dropout > 0.0 and random.random() < bpe_dropout):
                            prefer_right = True
                    if not prefer_right:
                        # merge current
                        out_ids.append(nid)
                        out_sp.append((spans[i][0], spans[i+1][1]))
                        i += 2; changed = True; continue
            # no merge
            out_ids.append(seq[i]); out_sp.append(spans[i]); i += 1
        return out_ids, out_sp, changed

    def _maybe_cache_get(self, key: str) -> Optional[List[int]]:
        if self.cache_size == 0: return None
        val = self._cache.get(key)
        if val is not None:
            # LRU bump
            self._cache.move_to_end(key)
        return val

    def _maybe_cache_put(self, key: str, value: List[int]) -> None:
        if self.cache_size == 0: return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _cache_clear(self):
        if self.cache_size:
            self._cache.clear()

    def encode(self,
               text: str,
               add_bos: bool = False,
               add_eos: bool = False,
               max_len: Optional[int] = None,
               truncation: bool = True,
               return_offsets: bool = False,
               bpe_dropout: float = 0.0) -> Union[List[int], Tuple[List[int], List[Tuple[int, int]]]]:
        # Cache sadece saf, deterministik, offsetsiz durum için devrede
        use_cache = (self.cache_size > 0 and bpe_dropout == 0.0 and not add_bos and not add_eos
                     and max_len is None and return_offsets is False and truncation)
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
                seq = seq[:max_len]; spans = spans[:max_len]
            else:
                raise ValueError(f"Sequence length {len(seq)} > max_len={max_len}")

        if use_cache:
            self._maybe_cache_put(norm_key, seq)

        if return_offsets:
            return seq, spans
        return seq

    def batch_encode(self,
                     texts: List[str],
                     add_bos: bool = False,
                     add_eos: bool = False,
                     max_len: Optional[int] = None,
                     truncation: bool = True,
                     pad_to_multiple_of: Optional[int] = None,
                     bpe_dropout: float = 0.0
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.encode(t, add_bos=add_bos, add_eos=add_eos,
                            max_len=max_len, truncation=truncation,
                            return_offsets=False, bpe_dropout=bpe_dropout)
                for t in texts]
        maxL = max((len(s) for s in seqs), default=1)
        if pad_to_multiple_of and maxL % pad_to_multiple_of != 0:
            maxL = ((maxL + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        B = len(seqs)
        ids = torch.full((B, maxL), self.special.PAD, dtype=torch.long)
        mask = torch.zeros((B, maxL), dtype=torch.long)
        for i, s in enumerate(seqs):
            L = min(len(s), maxL)
            if L > 0:
                ids[i, :L] = torch.tensor(s[:L], dtype=torch.long)
                mask[i, :L] = 1
        return ids, mask

    def _ids_to_text_via_bytes(self, ids: list[int]) -> str:
        out_bytes = []
        for i in ids:
            if i == self.special.PAD:   continue
            if i == self.special.BOS:   continue
            if i == self.special.EOS:   break
            if i == self.special.UNK:   # '?' koy
                out_bytes.append(ord('?')); continue
            out_bytes.extend(self._id_to_bytes(i))
    # HATA KAYNAĞI BUYDU: strict -> replace
        text = bytes(out_bytes).decode("utf-8", errors="replace")
        return self._denormalize_text(text)

    def decode(self, ids, skip_special: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._ids_to_text_via_bytes(ids)

    def batch_decode(self, batch_ids, skip_special: bool = True):
        return [self.decode(x, skip_special=skip_special) for x in batch_ids]



    # ---------- meta ----------
    @property
    def pad_id(self): return self.special.PAD
    @property
    def bos_id(self): return self.special.BOS
    @property
    def eos_id(self): return self.special.EOS
    @property
    def unk_id(self): return self.special.UNK
    @property
    def vocab_size(self) -> int:
        return self.special.OFFSET + 256 + len(self.merges_seq)

    # ---------- save/load ----------
    def save(self, path: str) -> None:
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
            "merges": self.merges_seq,   # (l, r, new_id)
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "HysoTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        SUPPORTED = {2, 3}
        ver = obj.get("version", 1)
        if ver not in SUPPORTED:
            raise ValueError(f"Unsupported tokenizer version: {ver}")

        # v2->v3 göç varsayılanları
        tok = cls(
            lowercase=obj.get("lowercase", False),
            normalize=obj.get("normalize", "NFKC"),
            use_space_sentinel=obj.get("use_space_sentinel", True),
            sentinel_char=obj.get("sentinel_char", "▁"),
            use_newline_sentinel=obj.get("use_newline_sentinel", True if ver >= 3 else False),
            newline_sentinel=obj.get("newline_sentinel", "⏎" if ver >= 3 else "⏎"),
            cache_size=0
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

        # base tablolar
        tok.id_to_bytes.clear(); tok.bytes_to_id.clear()
        for b in range(256):
            _id = tok.special.OFFSET + b
            tok.id_to_bytes[_id] = [b]
            tok.bytes_to_id[(b,)] = _id

        tok.merges_seq = []
        tok.pair2id.clear(); tok.rank.clear()
        for rank, (l, r, new_id) in enumerate(obj.get("merges", [])):
            assert l >= tok.special.OFFSET and r >= tok.special.OFFSET
            tok.merges_seq.append((l, r, new_id))
            tok.pair2id[(l, r)] = new_id
            tok.rank[(l, r)] = rank
            tok.id_to_bytes[new_id] = tok._id_to_bytes(l) + tok._id_to_bytes(r)
            tok.bytes_to_id[tuple(tok.id_to_bytes[new_id])] = new_id

        return tok

    # ---------- debug/export ----------
    def export_vocab(self) -> Dict[int, str]:
        def escape(bs: List[int]) -> str:
            return "".join(chr(b) if 32 <= b < 127 else f"\\x{b:02x}" for b in bs)
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

# --- GPT-2 style reversible bytes<->unicode mapping ---
def _bytes_to_unicode():
    # Taken from OpenAI GPT-2; every byte gets a printable unicode.
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    bs = [bytes([n]) for n in bs]
    return dict(zip(bs, cs))

def _unicode_to_bytes():
    b2u = _bytes_to_unicode()
    return {u: b for b, u in b2u.items()}
