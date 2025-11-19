from .bpe_tokenizer import HysoBPETokenizer
from .simple_tokenizer import SimpleTokenizer


def build_tokenizer(cfg):
    t = str(cfg.get("type", "bpe")).lower()
    lowercase = cfg.get("lowercase", False)
    normalize = cfg.get("normalize", "NFKC")
    cache_size = cfg.get("cache_size", 0)

    if t == "bpe":
        return HysoBPETokenizer(
            lowercase=lowercase,
            normalize=normalize,
            use_space_sentinel=cfg.get("use_space_sentinel", True),
            sentinel_char=cfg.get("sentinel_char", "▁"),
            use_newline_sentinel=cfg.get("use_newline_sentinel", True),
            newline_sentinel=cfg.get("newline_sentinel", "⏎"),
            cache_size=cache_size,
        )
    elif t in ("simple", "word", "wordlevel"):
        return SimpleTokenizer(
            lowercase=lowercase,
            normalize=normalize,
            cache_size=cache_size,
            min_freq=cfg.get("min_freq", 1),
            max_vocab_size=cfg.get("max_vocab_size", None),
            split_mode=cfg.get("split_mode", "basic"),
            keep_punct=cfg.get("keep_punct", True),
            punct_chars=cfg.get("punct_chars", None),
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {t}")


__all__ = [
    "HysoBPETokenizer",
    "SimpleTokenizer",
    "build_tokenizer",
]
