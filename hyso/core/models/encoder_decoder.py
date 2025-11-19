# model.py
import math
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        rnd = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rnd.floor_()
        return x.div(keep) * rnd

def build_rope_cache(head_dim: int, max_len: int, base: float = 10000.0,
                     device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cpu") if device is None else device
    dtype = torch.float32 if dtype is None else dtype
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_len, device=device, dtype=dtype)
    freqs = torch.einsum("t,f->tf", t, inv)   # [T, Hd/2]
    cos = torch.cos(freqs)                    # [T, Hd/2]
    sin = torch.sin(freqs)                    # [T, Hd/2]
    return cos, sin

def apply_rope_slice(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B,H,T,D], cos/sin: [T, D/2] -> apply rotary to this slice."""
    B, H, T, D = x.shape
    assert D % 2 == 0, "RoPE head_dim must be even."
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)  # [1,1,T,D/2]
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
    xr = torch.stack([x_even * cos_t - x_odd * sin_t,
                      x_even * sin_t + x_odd * cos_t], dim=-1).flatten(-2)
    return xr

def key_padding_to_bool(attn_mask_bt: torch.Tensor) -> torch.Tensor:
    # [B,T] (1=token,0=pad) → [B,T] (True=PAD=gizle)
    if attn_mask_bt.dtype == torch.bool:
        return attn_mask_bt
    return (attn_mask_bt == 0)

def _merge_masks(attn_mask: Optional[torch.Tensor],
                 key_pad_bool_bt: Optional[torch.Tensor],
                 Tq: int, Tk: int, device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns:
      - bool_mask: [B,1,Tq,Tk] (True=masked) or None
      - additive:  float mask/bias (e.g. ALiBi) or None
    """
    bool_mask, additive = None, None
    if key_pad_bool_bt is not None:
        B = key_pad_bool_bt.size(0)
        bool_mask = key_pad_bool_bt[:, None, None, :].expand(B, 1, Tq, Tk).to(device=device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            m = attn_mask
            if m.dim() == 2:      # [Tq,Tk]
                m = m[None, None, :, :]
            elif m.dim() == 4:    # [B,1,Tq,Tk]
                pass
            else:
                raise ValueError(f"attn_mask bool expected; got {attn_mask.shape}")
            m = m.to(device=device)
            bool_mask = m if bool_mask is None else (bool_mask | m)
        else:
            additive = attn_mask.to(device=device)
    return bool_mask, additive

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight

def make_norm(d_model: int, use_rmsnorm: bool, eps: float) -> nn.Module:
    return RMSNorm(d_model, eps) if use_rmsnorm else nn.LayerNorm(d_model, eps=eps)

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, mlp_dropout: float = 0.1, bias: bool = True):
        super().__init__()
        hidden = expansion * d_model
        self.wi = nn.Linear(d_model, 2 * hidden, bias=bias)  # -> [U, V]
        self.wo = nn.Linear(hidden, d_model, bias=bias)
        self.drop_hidden = nn.Dropout(mlp_dropout)
        self.drop_out = nn.Dropout(mlp_dropout)

        nn.init.xavier_uniform_(self.wi.weight)
        nn.init.xavier_uniform_(self.wo.weight)
        if self.wi.bias is not None: nn.init.zeros_(self.wi.bias)
        if self.wo.bias is not None: nn.init.zeros_(self.wo.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.wi(x).chunk(2, dim=-1)      # [B,T,hidden] x2
        x = F.silu(u) * v                        # SwiGLU
        x = self.drop_hidden(x)
        x = self.wo(x)
        x = self.drop_out(x)
        if not torch.jit.is_scripting():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return x


class Step3Attention(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 kv_heads: Optional[int] = None,
                 attn_dropout: float = 0.1,
                 proj_dropout: float = 0.1,
                 use_fused_qkv_for_self: bool = True,
                 use_bias_qkv: bool = False,
                 use_bias_out: bool = True,
                 use_sdpa: bool = True,
                 ln_eps: float = 1e-5,
                 use_qk_norm: bool = False,
                 qk_norm_eps: float = 1e-6,
                 learnable_q_scale: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model, n_heads'e tam bölünmeli."
        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_heads = n_heads if kv_heads is None else kv_heads
        assert self.n_heads % self.kv_heads == 0, "n_heads, kv_heads'e tam bölünmeli."
        self.group_size = self.n_heads // self.kv_heads

        self.head_dim = d_model // n_heads               # Hd
        self.q_dim = self.n_heads * self.head_dim        # = d_model
        self.kv_dim = self.kv_heads * self.head_dim      # ≤ d_model

        self.use_sdpa = use_sdpa
        self.fused_qkv_for_self = use_fused_qkv_for_self

        self.use_qk_norm = use_qk_norm
        self.qk_norm_eps = float(qk_norm_eps)
        self.learnable_q_scale = learnable_q_scale
        self.q_scale = nn.Parameter(torch.ones(1)) if learnable_q_scale else None

        # Fused QKV: q_dim + 2*kv_dim
        self.qkv   = nn.Linear(d_model, self.q_dim + 2*self.kv_dim, bias=use_bias_qkv) if self.fused_qkv_for_self else None
        # Ayrı projeksiyonlar
        self.q_proj = nn.Linear(d_model, self.q_dim, bias=use_bias_qkv)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=use_bias_qkv)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=use_bias_qkv)

        self.out    = nn.Linear(d_model, d_model, bias=use_bias_out)
        self.attn_dropout = float(attn_dropout)
        self.out_drop = nn.Dropout(proj_dropout)
        self.ln_eps = ln_eps

        for layer in [self.qkv, self.q_proj, self.k_proj, self.v_proj, self.out]:
            if layer is not None:
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.q_dim, f"Q dim beklenen {self.q_dim}, gelen {D}"
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()  # [B,h,T,Hd]

    def _shape_kv(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.kv_dim, f"KV dim beklenen {self.kv_dim}, gelen {D}"
        kv_h = self.kv_heads
        kv_hd = self.head_dim
        x = x.view(B, T, kv_h, kv_hd).transpose(1, 2).contiguous()  # [B,kv_h,T,Hd]
        if self.group_size > 1:
            x = x.repeat_interleave(self.group_size, dim=1)         # [B,h,T,Hd]
        return x

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H*Hd)

    @staticmethod
    def kv_proj_safe(proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        y = proj(x)
        if not torch.jit.is_scripting():
            y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        return y

    def _qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q / (q.norm(dim=-1, keepdim=True) + self.qk_norm_eps)
        k = k / (k.norm(dim=-1, keepdim=True) + self.qk_norm_eps)
        return q, k

    def forward(self,
                x_q: torch.Tensor,                 # [B,Tq,D]
                x_kv: Optional[torch.Tensor],      # None→self; yoksa cross [B,Tk,D]
                *,
                key_padding_mask: Optional[torch.Tensor] = None,  # [B,Tk] 1/0 veya bool (True=PAD)
                attn_mask: Optional[torch.Tensor] = None,         # bool [Tq,Tk]/[B,1,Tq,Tk] veya additive float
                attn_bias: Optional[torch.Tensor] = None,         # additive bias [1/B,1/h,Tq,Tk]
                is_causal: bool = False,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (cos,sin)
                pos_offset: int = 0,
                q_ln: Optional[nn.Module] = None,
                kv_ln: Optional[nn.Module] = None,
                kv_cache: Optional[Dict[str, torch.Tensor]] = None    # self-attn: {k,v}
                ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        # LN ayrımı
        q_in  = q_ln(x_q) if q_ln is not None else x_q
        src   = x_q if x_kv is None else x_kv
        kv_in = kv_ln(src) if kv_ln is not None else src

        # Q/K/V projeksiyonları
        if x_kv is None and self.fused_qkv_for_self:
            qkv = self.qkv(q_in)
            q_lin, k_lin, v_lin = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            q_lin = self.q_proj(q_in)
            k_lin = self.kv_proj_safe(self.k_proj, kv_in)
            v_lin = self.kv_proj_safe(self.v_proj, kv_in)

        # optional learnable q-scale
        if self.q_scale is not None:
            q_lin = q_lin * self.q_scale

        q = self._shape_q(q_lin)     # [B,h,Tq,Hd]
        k_new = self._shape_kv(k_lin)# [B,h,Tk_new,Hd]
        v_new = self._shape_kv(v_lin)

        # KV-cache uzunluğu
        cache_len = 0
        if kv_cache is not None and x_kv is None and "k" in kv_cache:
            cache_len = kv_cache["k"].size(2)

        # RoPE
        if rope is not None:
            cos_full, sin_full = rope
            cos_full = cos_full.to(q.device, q.dtype); sin_full = sin_full.to(q.device, q.dtype)

            Tq = q.size(2)
            cos_q = cos_full[pos_offset:pos_offset + Tq]
            sin_q = sin_full[pos_offset:pos_offset + Tq]
            q = apply_rope_slice(q, cos_q, sin_q)

            Tk_new = k_new.size(2)
            cos_k = cos_full[cache_len:cache_len + Tk_new]
            sin_k = sin_full[cache_len:cache_len + Tk_new]
            k_new = apply_rope_slice(k_new, cos_k, sin_k)

        # QK-Norm (RoPE sonrası uygulanır)
        if self.use_qk_norm:
            q, k_new = self._qk_norm(q, k_new)

        # KV-cache birleştir
        if kv_cache is not None and x_kv is None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k_new], dim=2)
                v = torch.cat([kv_cache["v"], v_new], dim=2)
            else:
                k, v = k_new, v_new
            kv_cache = {"k": k, "v": v}
        else:
            k, v = k_new, v_new

        Tk = k.size(2)
        device = q.device

        # Mask standardizasyonu
        key_pad_bool_bt = key_padding_to_bool(key_padding_mask) if key_padding_mask is not None else None
        bool_mask, additive = _merge_masks(attn_mask, key_pad_bool_bt, q.size(2), Tk, device)
        if attn_bias is not None:
            additive = attn_bias.to(device=device) if additive is None else (additive + attn_bias.to(device=device))

        if additive is not None and bool_mask is not None:
            neg_inf = torch.finfo(q.dtype).min
            sdpa_mask = additive.masked_fill(bool_mask, neg_inf)
        else:
            sdpa_mask = additive if additive is not None else bool_mask

        # SDPA / fallback
        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            # PyTorch kuralı: is_causal=True iken explicit attn_mask verilmemeli.
            # Eğer causal + başka maske varsa, causal'ı maskeye gömüp is_causal=False yap.
            use_is_causal = is_causal
            if is_causal and sdpa_mask is not None:
                Tq, Tk_ = q.size(2), k.size(2)
                causal = torch.ones(Tq, Tk_, dtype=torch.bool, device=device).triu(1)[None, None, :, :]
                if sdpa_mask.dtype == torch.bool:
                    sdpa_mask = sdpa_mask | causal
                else:
                    neg_inf = torch.finfo(q.dtype).min
                    sdpa_mask = sdpa_mask.masked_fill(causal, neg_inf)
                use_is_causal = False  # causal artık maskeye gömülü

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,                         # None/bool/additive (causal gömülü olabilir)
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=use_is_causal
            )
        else:
            # Fallback: logits = QK^T / sqrt(Hd) + causal/pad/additive
            scale = 1.0 / math.sqrt(self.head_dim)
            logits = (q @ k.transpose(-2, -1)) * scale

            if is_causal:
                Tq, Tk_ = q.size(2), k.size(2)
                causal = torch.ones(Tq, Tk_, dtype=torch.bool, device=device).triu(1)[None, None, :, :]
                bool_mask = causal if bool_mask is None else (bool_mask | causal)

            if bool_mask is not None:
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(bool_mask, neg_inf)
            if additive is not None:
                logits = logits + additive

            attn = torch.softmax(logits, dim=-1)
            attn = F.dropout(attn, p=self.attn_dropout if self.training else 0.0, training=self.training)
            out = attn @ v

        out = self._merge(out)               # [B,Tq,D]
        out = self.out_drop(self.out(out))   # proj dropout
        if not torch.jit.is_scripting():
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out, kv_cache


# ---------------------------
# 7) Encoder / Decoder Layers (+ ALiBi opsiyonu)
# ---------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 *, kv_heads: Optional[int] = None,
                 attn_dropout: float = 0.1, proj_dropout: float = 0.1,
                 resid_dropout: float = 0.1,
                 ffn_expansion: int = 4, mlp_dropout: float = 0.1,
                 droppath: float = 0.0, layerscale_init: float = 1e-2,
                 ln_eps: float = 1e-5, use_rmsnorm: bool = False,
                 use_qk_norm: bool = False, qk_norm_eps: float = 1e-6,
                 learnable_q_scale: bool = False):
        super().__init__()
        self.self_ln = make_norm(d_model, use_rmsnorm, ln_eps)
        self.self_attn = Step3Attention(
            d_model, n_heads, kv_heads=kv_heads,
            attn_dropout=attn_dropout, proj_dropout=proj_dropout,
            use_fused_qkv_for_self=True,
            use_bias_qkv=False, use_bias_out=True, use_sdpa=True,
            ln_eps=ln_eps,
            use_qk_norm=use_qk_norm, qk_norm_eps=qk_norm_eps,
            learnable_q_scale=learnable_q_scale
        )
        self.ls_attn = LayerScale(d_model, init_value=layerscale_init)
        self.resid_drop_attn = nn.Dropout(resid_dropout)
        self.drop_path_attn = DropPath(droppath) if droppath > 0 else nn.Identity()

        self.ffn_ln = make_norm(d_model, use_rmsnorm, ln_eps)
        self.ffn = SwiGLUFeedForward(d_model, expansion=ffn_expansion,
                                     mlp_dropout=mlp_dropout, bias=True)
        self.ls_ffn = LayerScale(d_model, init_value=layerscale_init)
        self.resid_drop_ffn = nn.Dropout(resid_dropout)
        self.drop_path_ffn = DropPath(droppath) if droppath > 0 else nn.Identity()

    def forward(self, x: torch.Tensor,
                *, rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y, _ = self.self_attn(self.self_ln(x), x_kv=None,
                              key_padding_mask=pad_mask,
                              is_causal=False, rope=rope)
        y = self.ls_attn(y)
        y = self.resid_drop_attn(y)
        x = x + self.drop_path_attn(y)

        z = self.ffn(self.ffn_ln(x))
        z = self.ls_ffn(z)
        z = self.resid_drop_ffn(z)
        x = x + self.drop_path_ffn(z)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 *, kv_heads: Optional[int] = None,
                 attn_dropout: float = 0.1, proj_dropout: float = 0.1,
                 resid_dropout: float = 0.1,
                 ffn_expansion: int = 4, mlp_dropout: float = 0.1,
                 droppath: float = 0.0, layerscale_init: float = 1e-2,
                 ln_eps: float = 1e-5, use_rmsnorm: bool = False,
                 use_qk_norm: bool = False, qk_norm_eps: float = 1e-6,
                 learnable_q_scale: bool = False,
                 use_alibi: bool = False,
                 alibi_slopes: Optional[torch.Tensor] = None):
        super().__init__()
        self.use_alibi = use_alibi
        self.alibi_slopes = alibi_slopes  # [h] (buffer dışarıdan verilecek)

        # Self-attn
        self.self_ln = make_norm(d_model, use_rmsnorm, ln_eps)
        self.self_attn = Step3Attention(
            d_model, n_heads, kv_heads=kv_heads,
            attn_dropout=attn_dropout, proj_dropout=proj_dropout,
            use_fused_qkv_for_self=True,
            use_bias_qkv=False, use_bias_out=True, use_sdpa=True,
            ln_eps=ln_eps,
            use_qk_norm=use_qk_norm, qk_norm_eps=qk_norm_eps,
            learnable_q_scale=learnable_q_scale
        )
        self.ls_self = LayerScale(d_model, init_value=layerscale_init)
        self.resid_drop_self = nn.Dropout(resid_dropout)
        self.drop_path_self = DropPath(droppath) if droppath > 0 else nn.Identity()

        # Cross-attn
        self.cross_q_ln = make_norm(d_model, use_rmsnorm, ln_eps)
        self.cross_kv_ln = make_norm(d_model, use_rmsnorm, ln_eps)
        self.cross_attn = Step3Attention(
            d_model, n_heads, kv_heads=kv_heads,
            attn_dropout=attn_dropout, proj_dropout=proj_dropout,
            use_fused_qkv_for_self=False,
            use_bias_qkv=False, use_bias_out=True, use_sdpa=True,
            ln_eps=ln_eps,
            use_qk_norm=False, learnable_q_scale=False  # genelde cross'a uygulamıyoruz
        )
        self.ls_cross = LayerScale(d_model, init_value=layerscale_init)
        self.resid_drop_cross = nn.Dropout(resid_dropout)
        self.drop_path_cross = DropPath(droppath) if droppath > 0 else nn.Identity()

        # FFN
        self.ffn_ln = make_norm(d_model, use_rmsnorm, ln_eps)
        self.ffn = SwiGLUFeedForward(d_model, expansion=ffn_expansion,
                                     mlp_dropout=mlp_dropout, bias=True)
        self.ls_ffn = LayerScale(d_model, init_value=layerscale_init)
        self.resid_drop_ffn = nn.Dropout(resid_dropout)
        self.drop_path_ffn = DropPath(droppath) if droppath > 0 else nn.Identity()

    def _alibi_bias(self, Tq: int, Tk: int, device, dtype, pos_offset: int) -> Optional[torch.Tensor]:
        if not self.use_alibi or self.alibi_slopes is None:
            return None
        # bias[h, Tq, Tk] = -slope[h] * ( (pos_offset + i) - j ), i in [0..Tq-1], j in [0..Tk-1]
        i = torch.arange(Tq, device=device, dtype=dtype).unsqueeze(-1) + pos_offset
        j = torch.arange(Tk, device=device, dtype=dtype).unsqueeze(0)
        dist = i - j  # [Tq,Tk]
        slopes = self.alibi_slopes.to(device=device, dtype=dtype)[:, None, None]  # [h,1,1]
        bias = -slopes * dist  # [h,Tq,Tk]
        return bias.unsqueeze(0)  # [1,h,Tq,Tk]

    def forward(self,
                x_dec: torch.Tensor, x_enc: torch.Tensor,
                *, enc_attention_mask: Optional[torch.Tensor] = None,
                tgt_attention_mask: Optional[torch.Tensor] = None,
                self_attn_causal: bool = True,
                rope_dec: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                rope_enc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                pos_offset: int = 0,
                self_kv_cache: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        # Self-attn (+ ALiBi bias ops.)
        alibi = None
        if self.use_alibi:
            Tq = x_dec.size(1)
            Tk = x_dec.size(1) if self_kv_cache is None else (self_kv_cache["k"].size(2) + 1)
            alibi = self._alibi_bias(Tq, Tk, x_dec.device, x_dec.dtype, pos_offset)

        y, self_kv_cache = self.self_attn(
            self.self_ln(x_dec), x_kv=None,
            key_padding_mask=tgt_attention_mask,
            attn_mask=None, attn_bias=alibi,
            is_causal=self_attn_causal,
            rope=rope_dec, pos_offset=pos_offset,
            kv_cache=self.self_kv_cache_passthrough(self_kv_cache)
        )
        y = self.ls_self(y)
        y = self.resid_drop_self(y)
        x_dec = x_dec + self.drop_path_self(y)

        # Cross-attn
        z, _ = self.cross_attn(x_dec, x_kv=x_enc,
                               key_padding_mask=enc_attention_mask,
                               attn_mask=None, attn_bias=None, is_causal=False,
                               rope=rope_enc, pos_offset=0,
                               q_ln=self.cross_q_ln, kv_ln=self.cross_kv_ln)
        z = self.ls_cross(z)
        z = self.resid_drop_cross(z)
        x_dec = x_dec + self.drop_path_cross(z)

        # FFN
        u = self.ffn(self.ffn_ln(x_dec))
        u = self.ls_ffn(u)
        u = self.resid_drop_ffn(u)
        x_dec = x_dec + self.drop_path_ffn(u)
        return x_dec, self_kv_cache

    @staticmethod
    def self_kv_cache_passthrough(cache: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
        return cache


# ---------------------------
# 8) Full Encoder–Decoder Transformer
# ---------------------------
class HysoLLM(nn.Module):
    def __init__(self,
                 vocab_src: int, vocab_tgt: int,
                 *, d_model: int = 512, n_heads: int = 8,
                 kv_heads: Optional[int] = None,        # GQA/MQA
                 num_layers_enc: int = 6, num_layers_dec: int = 6,
                 attn_dropout: float = 0.1, proj_dropout: float = 0.1,
                 resid_dropout: float = 0.1, embed_dropout: float = 0.0,
                 ffn_expansion: int = 4, mlp_dropout: float = 0.1,
                 droppath: float = 0.0, layerscale_init: float = 1e-2,
                 ln_eps: float = 1e-5,
                 max_len: int = 2048, use_rope: bool = False, rope_base: float = 10000.0,
                 tie_embed: bool = False, pad_id: int = 0,
                 use_rmsnorm: bool = False,
                 # Ek: QK-Norm/learnable scale/ALiBi/grad ckpt
                 use_qk_norm: bool = False, qk_norm_eps: float = 1e-6,
                 learnable_q_scale: bool = False,
                 use_alibi: bool = False,
                 grad_ckpt_enc: bool = False,
                 bos_id: int = 1, eos_id: int = 2):
        super().__init__()
        self.d_model = d_model
        self.use_rope = use_rope
        self.rope_base = float(rope_base)
        self.pad_id = pad_id
        self.max_len_rope = max_len
        self.n_heads = n_heads
        self.kv_heads = n_heads if kv_heads is None else kv_heads
        self.grad_ckpt_enc = bool(grad_ckpt_enc)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)

        self.src_embed = nn.Embedding(vocab_src, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_tgt, d_model, padding_idx=pad_id)
        self.embed_drop = nn.Dropout(embed_dropout)

        # RoPE
        if use_rope:
            cos, sin = build_rope_cache(d_model // n_heads, max_len, base=self.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

        # ALiBi slopes (sadece decoder self-attn'de kullanılıyor)
        self.use_alibi = bool(use_alibi)
        if self.use_alibi:
            slopes = self._alibi_slopes(self.n_heads)  # [h]
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        else:
            self.alibi_slopes = None

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, kv_heads=self.kv_heads,
                         attn_dropout=attn_dropout, proj_dropout=proj_dropout,
                         resid_dropout=resid_dropout,
                         ffn_expansion=ffn_expansion, mlp_dropout=mlp_dropout,
                         droppath=droppath, layerscale_init=layerscale_init,
                         ln_eps=ln_eps, use_rmsnorm=use_rmsnorm,
                         use_qk_norm=use_qk_norm, qk_norm_eps=qk_norm_eps,
                         learnable_q_scale=learnable_q_scale)
            for _ in range(num_layers_enc)
        ])
        self.enc_ln = make_norm(d_model, use_rmsnorm, ln_eps)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, kv_heads=self.kv_heads,
                         attn_dropout=attn_dropout, proj_dropout=proj_dropout,
                         resid_dropout=resid_dropout,
                         ffn_expansion=ffn_expansion, mlp_dropout=mlp_dropout,
                         droppath=droppath, layerscale_init=layerscale_init,
                         ln_eps=ln_eps, use_rmsnorm=use_rmsnorm,
                         use_qk_norm=use_qk_norm, qk_norm_eps=qk_norm_eps,
                         learnable_q_scale=learnable_q_scale,
                         use_alibi=self.use_alibi, alibi_slopes=self.alibi_slopes)
            for _ in range(num_layers_dec)
        ])
        self.dec_ln = make_norm(d_model, use_rmsnorm, ln_eps)

        # LM Head
        self.lm_head = nn.Linear(d_model, vocab_tgt, bias=False)
        if tie_embed:
            self.tie_embeddings(True)
        else:
            self._tied = False

        # Init
        self._reset_parameters()

    # ---------- Init / Utils ----------
    def _reset_parameters(self):
        std = self.d_model ** -0.5
        nn.init.normal_(self.src_embed.weight, mean=0.0, std=std)
        nn.init.normal_(self.tgt_embed.weight, mean=0.0, std=std)

    def tie_embeddings(self, enable: bool = True):
        self._tied = enable
        if enable:
            self.lm_head.weight = self.tgt_embed.weight
        else:
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.detach().clone())

    def greedy(self, tok_src, tok_tgt, src_texts, **kwargs):
        from .generate import run_greedy
        return run_greedy(self, tok_src, tok_tgt, src_texts, **kwargs)

    def sample(self, tok_src, tok_tgt, src_texts, **kwargs):
        from .generate import run_sampling
        return run_sampling(self, tok_src, tok_tgt, src_texts, **kwargs)

    def generate(self, tok_src, tok_tgt, src_texts, **kwargs):
        from .generate import run_generate
        return run_generate(self, tok_src, tok_tgt, src_texts, **kwargs)


    def resize_token_embeddings(self, new_size: int, which: str = "tgt"):
        assert which in {"src", "tgt"}
        emb = self.src_embed if which == "src" else self.tgt_embed
        old_weight = emb.weight.data
        d = old_weight.size(1)
        new_weight = old_weight.new_empty(new_size, d)
        new_weight.normal_(mean=0.0, std=(self.d_model ** -0.5))
        num = min(old_weight.size(0), new_size)
        new_weight[:num] = old_weight[:num]
        emb.weight = nn.Parameter(new_weight)
        if which == "tgt" and getattr(self, "_tied", False):
            self.lm_head.weight = emb.weight

    def ensure_rope_len(self, need_len: int):
        if not self.use_rope:
            return
        if need_len <= self.max_len_rope:
            return
        # cache'i uzat
        hd = self.d_model // self.n_heads
        cos_new, sin_new = build_rope_cache(hd, need_len, base=self.rope_base,
                                            device=self.rope_cos.device, dtype=self.rope_cos.dtype)
        self.rope_cos = cos_new
        self.rope_sin = sin_new
        self.max_len_rope = need_len

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _alibi_slopes(n_heads: int) -> torch.Tensor:

        def get_slopes(n):
            def power_of_2(n): 
                return 2 ** math.ceil(math.log2(n))
            m = power_of_2(n)
            m_list = torch.pow(2, -torch.arange(1, m+1)).tolist()
            slopes = torch.tensor(m_list[:n], dtype=torch.float32)
            return slopes
        return get_slopes(n_heads)

    def _maybe_ckpt(self, layer: EncoderLayer, x: torch.Tensor,
                    rope, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.grad_ckpt_enc or not self.training:
            return layer(x, rope=rope, pad_mask=pad_mask)

        def fn(inp):
            return layer(inp, rope=rope, pad_mask=pad_mask)
        return ckpt(fn, x)

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        # src_ids: [B,Ts]
        x = self.src_embed(src_ids) * (self.d_model ** 0.5)
        x = self.embed_drop(x)
        rope = (self.rope_cos, self.rope_sin) if self.use_rope else None
        if rope is not None:
            self.ensure_rope_len(src_ids.size(1))

        src_pad_mask = (src_ids != self.pad_id).long()  # [B,T]
        for layer in self.encoder_layers:
            x = self._maybe_ckpt(layer, x, rope=rope, pad_mask=src_pad_mask)
        return self.enc_ln(x)

    def decode(self,
               tgt_ids_in: torch.Tensor, enc_out: torch.Tensor,
               *, src_attention_mask: Optional[torch.Tensor] = None,
               pos_offset: int = 0) -> torch.Tensor:
        # tgt_ids_in: [B,Tt_in] (BOS + ... + y[t-1])
        y = self.tgt_embed(tgt_ids_in) * (self.d_model ** 0.5)
        y = self.embed_drop(y)
        rope_dec = (self.rope_cos, self.rope_sin) if self.use_rope else None
        rope_enc = (self.rope_cos, self.rope_sin) if self.use_rope else None
        if rope_dec is not None:
            self.ensure_rope_len(max(tgt_ids_in.size(1) + pos_offset, enc_out.size(1)))

        tgt_pad_mask = (tgt_ids_in != self.pad_id).long()  # [B,Tt_in]

        for layer in self.decoder_layers:
            y, _ = layer(y, enc_out,
                         enc_attention_mask=src_attention_mask,
                         tgt_attention_mask=tgt_pad_mask,
                         self_attn_causal=True,
                         rope_dec=rope_dec, rope_enc=rope_enc,
                         pos_offset=pos_offset,
                         self_kv_cache=None)
        return self.dec_ln(y)

    def forward(self,
                src_ids: torch.Tensor, tgt_ids: torch.Tensor,
                *, src_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        enc_out = self.encode(src_ids)  # [B,Ts,D]
        tgt_in  = tgt_ids[:, :-1]
        dec_hid = self.decode(tgt_in, enc_out, src_attention_mask=src_attention_mask)  # [B,Tt-1,D]
        logits  = self.lm_head(dec_hid) # [B,Tt-1,Vt]
        return logits

    def compute_loss(self, logits: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        target = tgt_ids[:, 1:].contiguous()
        B, T, V = logits.shape
        assert target.shape[:2] == (B, T), f"Loss alignment broken: logits {logits.shape}, target {target.shape}"
        loss = F.cross_entropy(
            logits.reshape(B*T, V), target.reshape(B*T),
            ignore_index=self.pad_id
        )
        return loss

    @torch.no_grad()
    def generate_step(self,
                      prev_token: torch.Tensor,   # [B]
                      enc_out: torch.Tensor,      # [B,Ts,D]
                      *, layer_caches: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
                      src_attention_mask: Optional[torch.Tensor] = None,
                      step_idx: int = 0
                      ) -> Tuple[torch.Tensor, List[Optional[Dict[str, torch.Tensor]]]]:
        device = enc_out.device
        prev_token = prev_token.to(device)
        if layer_caches is None:
            layer_caches = [None] * len(self.decoder_layers)

        y = self.tgt_embed(prev_token).unsqueeze(1) * (self.d_model ** 0.5)
        y = self.embed_drop(y)
        rope_dec = (self.rope_cos, self.rope_sin) if self.use_rope else None
        rope_enc = (self.rope_cos, self.rope_sin) if self.use_rope else None
        if rope_dec is not None:
            self.ensure_rope_len(max(step_idx + 1, enc_out.size(1)))

        new_caches = []
        x = y
        for i, layer in enumerate(self.decoder_layers):
            x, cache = layer(x, enc_out,
                             enc_attention_mask=src_attention_mask,
                             tgt_attention_mask=None,
                             self_attn_causal=True,
                             rope_dec=rope_dec, rope_enc=rope_enc,
                             pos_offset=step_idx,
                             self_kv_cache=layer_caches[i])
            new_caches.append(cache)

        x = self.dec_ln(x)
        logits = self.lm_head(x[:, -1, :])  # [B,V]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return logits, new_caches

    @torch.no_grad()
    def generate_greedy(self,
                        src_ids: torch.Tensor,
                        *, src_attention_mask: Optional[torch.Tensor] = None,
                        bos_id: Optional[int] = None, eos_id: Optional[int] = None,
                        max_new_tokens: int = 32) -> torch.Tensor:
        enc_out = self.encode(src_ids)
        B = src_ids.size(0)
        caches: List[Optional[Dict[str, torch.Tensor]]] = [None]*len(self.decoder_layers)
        if bos_id is None: bos_id = self.bos_id
        if eos_id is None: eos_id = self.eos_id
        prev = torch.full((B,), bos_id, dtype=torch.long, device=enc_out.device)
        out_tokens = [prev.clone()]

        for step in range(max_new_tokens):
            logits, caches = self.generate_step(prev, enc_out,
                                                layer_caches=caches,
                                                src_attention_mask=src_attention_mask,
                                                step_idx=step)
            prev = logits.argmax(-1)
            out_tokens.append(prev)
            if eos_id is not None and (prev == eos_id).all():
                break
        return torch.stack(out_tokens, dim=1)  # [B,1+T_new]

    @torch.no_grad()
    def generate_sampling(self,
                          src_ids: torch.Tensor,
                          *, src_attention_mask: Optional[torch.Tensor] = None,
                          bos_id: Optional[int] = None, eos_id: Optional[int] = None,
                          max_new_tokens: int = 32,
                          temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        assert temperature > 0, "temperature > 0 olmalı"
        enc_out = self.encode(src_ids)
        B = src_ids.size(0)
        caches: List[Optional[Dict[str, torch.Tensor]]] = [None]*len(self.decoder_layers)
        if bos_id is None: bos_id = self.bos_id
        if eos_id is None: eos_id = self.eos_id
        prev = torch.full((B,), bos_id, dtype=torch.long, device=enc_out.device)
        out_tokens = [prev.clone()]

        for step in range(max_new_tokens):
            logits, caches = self.generate_step(prev, enc_out,
                                                layer_caches=caches,
                                                src_attention_mask=src_attention_mask,
                                                step_idx=step)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            if top_k > 0:
                k = min(top_k, probs.size(-1))
                topk_vals, topk_idx = torch.topk(probs, k, dim=-1)
                mask = probs.new_zeros(probs.shape).scatter_(1, topk_idx, 1.0)
                probs = probs * mask
                probs = probs / probs.sum(dim=-1, keepdim=True)

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumsum > top_p).float()
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = 0.0
                sorted_probs = sorted_probs * (1.0 - cutoff)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                probs = probs.new_zeros(probs.shape).scatter(1, sorted_idx, sorted_probs)

            prev = torch.multinomial(probs, num_samples=1).squeeze(1)
            out_tokens.append(prev)

            if eos_id is not None and (prev == eos_id).all():
                break

        return torch.stack(out_tokens, dim=1)  # [B,1+T_new]

TransformerED = HysoLLM