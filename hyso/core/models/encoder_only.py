import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt


# ---------------------------
# 0) DropPath
# ---------------------------
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


# ---------------------------
# 1) RoPE helpers
# ---------------------------
def build_rope_cache(head_dim: int,
                     max_len: int,
                     base: float = 10000.0,
                     device=None,
                     dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cpu") if device is None else device
    dtype = torch.float32 if dtype is None else dtype
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_len, device=device, dtype=dtype)
    freqs = torch.einsum("t,f->tf", t, inv)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope_slice(x: torch.Tensor,
                     cos: torch.Tensor,
                     sin: torch.Tensor) -> torch.Tensor:
    # x: [B,H,T,D], cos/sin: [T,D/2]
    B, H, T, D = x.shape
    assert D % 2 == 0
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)  # [1,1,T,D/2]
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
    xr = torch.stack(
        [
            x_even * cos_t - x_odd * sin_t,
            x_even * sin_t + x_odd * cos_t,
        ],
        dim=-1,
    ).flatten(-2)
    return xr


# ---------------------------
# 2) Mask helpers
# ---------------------------
def key_padding_to_bool(attn_mask_bt: torch.Tensor) -> torch.Tensor:
    # [B,T] (1=token,0=pad) -> [B,T] (True=PAD)
    if attn_mask_bt.dtype == torch.bool:
        return attn_mask_bt
    return (attn_mask_bt == 0)


def _merge_masks(attn_mask: Optional[torch.Tensor],
                 key_pad_bool_bt: Optional[torch.Tensor],
                 Tq: int,
                 Tk: int,
                 device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    bool_mask, additive = None, None

    if key_pad_bool_bt is not None:
        B = key_pad_bool_bt.size(0)
        bool_mask = key_pad_bool_bt[:, None, None, :].expand(B, 1, Tq, Tk).to(device=device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            m = attn_mask
            if m.dim() == 2:          # [Tq,Tk]
                m = m[None, None, :, :]
            elif m.dim() == 4:        # [B,1,Tq,Tk]
                pass
            else:
                raise ValueError(f"attn_mask bool shape: {attn_mask.shape}")
            m = m.to(device=device)
            bool_mask = m if bool_mask is None else (bool_mask | m)
        else:
            additive = attn_mask.to(device=device)

    return bool_mask, additive


# ---------------------------
# 3) LayerScale
# ---------------------------
class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# ---------------------------
# 4) Norms
# ---------------------------
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


# ---------------------------
# 5) SwiGLU MLP
# ---------------------------
class SwiGLUFeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 expansion: int = 4,
                 mlp_dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        hidden = expansion * d_model
        self.wi = nn.Linear(d_model, 2 * hidden, bias=bias)
        self.wo = nn.Linear(hidden, d_model, bias=bias)
        self.drop_hidden = nn.Dropout(mlp_dropout)
        self.drop_out = nn.Dropout(mlp_dropout)

        nn.init.xavier_uniform_(self.wi.weight)
        nn.init.xavier_uniform_(self.wo.weight)
        if self.wi.bias is not None:
            nn.init.zeros_(self.wi.bias)
        if self.wo.bias is not None:
            nn.init.zeros_(self.wo.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.wi(x).chunk(2, dim=-1)
        x = F.silu(u) * v
        x = self.drop_hidden(x)
        x = self.wo(x)
        x = self.drop_out(x)
        if not torch.jit.is_scripting():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return x


# ---------------------------
# 6) Attention (SDPA + GQA/MQA + RoPE + KV-cache)
# ---------------------------
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

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_heads = n_heads if kv_heads is None else kv_heads
        assert self.n_heads % self.kv_heads == 0
        self.group_size = self.n_heads // self.kv_heads

        self.head_dim = d_model // n_heads
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim

        self.use_sdpa = use_sdpa
        self.fused_qkv_for_self = use_fused_qkv_for_self

        self.use_qk_norm = use_qk_norm
        self.qk_norm_eps = float(qk_norm_eps)
        self.learnable_q_scale = learnable_q_scale
        self.q_scale = nn.Parameter(torch.ones(1)) if learnable_q_scale else None

        if self.fused_qkv_for_self:
            self.qkv = nn.Linear(d_model, self.q_dim + 2 * self.kv_dim, bias=use_bias_qkv)
        else:
            self.qkv = None
        self.q_proj = nn.Linear(d_model, self.q_dim, bias=use_bias_qkv)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=use_bias_qkv)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=use_bias_qkv)

        self.out = nn.Linear(d_model, d_model, bias=use_bias_out)
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
        assert D == self.q_dim
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

    def _shape_kv(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.kv_dim
        kv_h = self.kv_heads
        kv_hd = self.head_dim
        x = x.view(B, T, kv_h, kv_hd).transpose(1, 2).contiguous()
        if self.group_size > 1:
            x = x.repeat_interleave(self.group_size, dim=1)
        return x

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Hd)

    @staticmethod
    def kv_proj_safe(proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        y = proj(x)
        if not torch.jit.is_scripting():
            y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        return y

    def _qk_norm(self,
                 q: torch.Tensor,
                 k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q / (q.norm(dim=-1, keepdim=True) + self.qk_norm_eps)
        k = k / (k.norm(dim=-1, keepdim=True) + self.qk_norm_eps)
        return q, k

    def forward(self,
                x_q: torch.Tensor,
                x_kv: Optional[torch.Tensor],
                *,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None,
                is_causal: bool = False,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                pos_offset: int = 0,
                q_ln: Optional[nn.Module] = None,
                kv_ln: Optional[nn.Module] = None,
                kv_cache: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        q_in = q_ln(x_q) if q_ln is not None else x_q
        src = x_q if x_kv is None else x_kv
        kv_in = kv_ln(src) if kv_ln is not None else src

        if x_kv is None and self.fused_qkv_for_self:
            qkv = self.qkv(q_in)
            q_lin, k_lin, v_lin = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            q_lin = self.q_proj(q_in)
            k_lin = self.kv_proj_safe(self.k_proj, kv_in)
            v_lin = self.kv_proj_safe(self.v_proj, kv_in)

        if self.q_scale is not None:
            q_lin = q_lin * self.q_scale

        q = self._shape_q(q_lin)
        k_new = self._shape_kv(k_lin)
        v_new = self._shape_kv(v_lin)

        cache_len = 0
        if kv_cache is not None and x_kv is None and "k" in kv_cache:
            cache_len = kv_cache["k"].size(2)

        if rope is not None:
            cos_full, sin_full = rope
            cos_full = cos_full.to(q.device, q.dtype)
            sin_full = sin_full.to(q.device, q.dtype)

            Tq = q.size(2)
            cos_q = cos_full[pos_offset:pos_offset + Tq]
            sin_q = sin_full[pos_offset:pos_offset + Tq]
            q = apply_rope_slice(q, cos_q, sin_q)

            Tk_new = k_new.size(2)
            cos_k = cos_full[cache_len:cache_len + Tk_new]
            sin_k = sin_full[cache_len:cache_len + Tk_new]
            k_new = apply_rope_slice(k_new, cos_k, sin_k)

        if self.use_qk_norm:
            q, k_new = self._qk_norm(q, k_new)

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

        key_pad_bool_bt = key_padding_to_bool(key_padding_mask) if key_padding_mask is not None else None
        bool_mask, additive = _merge_masks(attn_mask, key_pad_bool_bt, q.size(2), Tk, device)
        if attn_bias is not None:
            additive = attn_bias.to(device=device) if additive is None else (additive + attn_bias.to(device=device))

        if additive is not None and bool_mask is not None:
            neg_inf = torch.finfo(q.dtype).min
            sdpa_mask = additive.masked_fill(bool_mask, neg_inf)
        else:
            sdpa_mask = additive if additive is not None else bool_mask

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            use_is_causal = is_causal
            if is_causal and sdpa_mask is not None:
                Tq_, Tk_ = q.size(2), k.size(2)
                causal = torch.ones(Tq_, Tk_, dtype=torch.bool, device=device).triu(1)[None, None, :, :]
                if sdpa_mask.dtype == torch.bool:
                    sdpa_mask = sdpa_mask | causal
                else:
                    neg_inf = torch.finfo(q.dtype).min
                    sdpa_mask = sdpa_mask.masked_fill(causal, neg_inf)
                use_is_causal = False

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=use_is_causal
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            logits = (q @ k.transpose(-2, -1)) * scale

            if is_causal:
                Tq_, Tk_ = q.size(2), k.size(2)
                causal = torch.ones(Tq_, Tk_, dtype=torch.bool, device=device).triu(1)[None, None, :, :]
                if bool_mask is None:
                    bool_mask = causal
                else:
                    bool_mask = bool_mask | causal

            if bool_mask is not None:
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(bool_mask, neg_inf)
            if additive is not None:
                logits = logits + additive

            attn = torch.softmax(logits, dim=-1)
            attn = F.dropout(attn, p=self.attn_dropout if self.training else 0.0, training=self.training)
            out = attn @ v

        out = self._merge(out)
        out = self.out_drop(self.out(out))
        if not torch.jit.is_scripting():
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out, kv_cache


# ---------------------------
# 7) EncoderLayer
# ---------------------------
class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 *,
                 kv_heads: Optional[int] = None,
                 attn_dropout: float = 0.1,
                 proj_dropout: float = 0.1,
                 resid_dropout: float = 0.1,
                 ffn_expansion: int = 4,
                 mlp_dropout: float = 0.1,
                 droppath: float = 0.0,
                 layerscale_init: float = 1e-2,
                 ln_eps: float = 1e-5,
                 use_rmsnorm: bool = False,
                 use_qk_norm: bool = False,
                 qk_norm_eps: float = 1e-6,
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
        self.ffn = SwiGLUFeedForward(d_model, expansion=ffn_expansion, mlp_dropout=mlp_dropout, bias=True)
        self.ls_ffn = LayerScale(d_model, init_value=layerscale_init)
        self.resid_drop_ffn = nn.Dropout(resid_dropout)
        self.drop_path_ffn = DropPath(droppath) if droppath > 0 else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                *,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y, _ = self.self_attn(
            self.self_ln(x),
            x_kv=None,
            key_padding_mask=pad_mask,
            attn_mask=None,
            attn_bias=None,
            is_causal=False,
            rope=rope,
            pos_offset=0,
            q_ln=None,
            kv_ln=None,
            kv_cache=None,
        )
        y = self.ls_attn(y)
        y = self.resid_drop_attn(y)
        x = x + self.drop_path_attn(y)

        z = self.ffn(self.ffn_ln(x))
        z = self.ls_ffn(z)
        z = self.resid_drop_ffn(z)
        x = x + self.drop_path_ffn(z)
        return x


# ---------------------------
# 8) Encoder-only Transformer
# ---------------------------
class HysoEncoderOnly(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 *,
                 d_model: int = 512,
                 n_heads: int = 8,
                 kv_heads: Optional[int] = None,
                 num_layers: int = 12,
                 attn_dropout: float = 0.1,
                 proj_dropout: float = 0.1,
                 resid_dropout: float = 0.1,
                 embed_dropout: float = 0.0,
                 ffn_expansion: int = 4,
                 mlp_dropout: float = 0.1,
                 droppath: float = 0.0,
                 layerscale_init: float = 1e-2,
                 ln_eps: float = 1e-5,
                 max_len: int = 2048,
                 use_rope: bool = False,
                 rope_base: float = 10000.0,
                 max_position_embeddings: Optional[int] = None,
                 pos_embedding_type: str = "learned",  # "learned" | "sinusoidal" | "none"
                 tie_embed: bool = False,
                 pad_id: int = 0,
                 use_rmsnorm: bool = False,
                 use_qk_norm: bool = False,
                 qk_norm_eps: float = 1e-6,
                 learnable_q_scale: bool = False,
                 grad_ckpt_enc: bool = False,
                 num_labels: int = 0,
                 pool_type: str = "cls",
                 cls_id: Optional[int] = None,
                 cls_dropout: float = 0.1,
                 cls_use_norm: bool = True,
                 cls_require_presence: bool = False):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_heads = n_heads if kv_heads is None else kv_heads
        self.pad_id = pad_id

        self.use_rope = use_rope
        self.rope_base = float(rope_base)
        self.max_len_rope = max_len

        if max_position_embeddings is None:
            max_position_embeddings = max_len
        self.max_position_embeddings = max_position_embeddings
        self.pos_embedding_type = pos_embedding_type

        self.grad_ckpt_enc = bool(grad_ckpt_enc)
        self.num_labels = int(num_labels)
        self.pool_type = pool_type
        self.cls_id = cls_id
        self.cls_require_presence = bool(cls_require_presence)

        assert d_model % n_heads == 0
        assert self.n_heads % self.kv_heads == 0
        assert self.pos_embedding_type in {"learned", "sinusoidal", "none"}

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embed_drop = nn.Dropout(embed_dropout)

        std = self.d_model ** -0.5

        if self.pos_embedding_type == "learned":
            self.pos_embed = nn.Embedding(self.max_position_embeddings, d_model)
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=std)
            self.register_buffer("pos_sinus", None, persistent=False)
        elif self.pos_embedding_type == "sinusoidal":
            pos = torch.arange(self.max_position_embeddings).unsqueeze(1)
            i = torch.arange(0, d_model, 2)
            angle_rates = torch.exp(-math.log(10000.0) * i / d_model)
            angles = pos * angle_rates
            pe = torch.zeros(self.max_position_embeddings, d_model)
            pe[:, 0::2] = torch.sin(angles)
            pe[:, 1::2] = torch.cos(angles)
            self.register_buffer("pos_sinus", pe, persistent=False)
            self.pos_embed = None
        else:
            self.pos_embed = None
            self.register_buffer("pos_sinus", None, persistent=False)

        if use_rope:
            cos, sin = build_rope_cache(d_model // n_heads, max_len, base=self.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, kv_heads=self.kv_heads,
                         attn_dropout=attn_dropout, proj_dropout=proj_dropout,
                         resid_dropout=resid_dropout,
                         ffn_expansion=ffn_expansion, mlp_dropout=mlp_dropout,
                         droppath=droppath, layerscale_init=layerscale_init,
                         ln_eps=ln_eps, use_rmsnorm=use_rmsnorm,
                         use_qk_norm=use_qk_norm, qk_norm_eps=qk_norm_eps,
                         learnable_q_scale=learnable_q_scale)
            for _ in range(num_layers)
        ])
        self.enc_ln = make_norm(d_model, use_rmsnorm, ln_eps)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if self.num_labels > 0:
            self.cls_norm = make_norm(d_model, use_rmsnorm, ln_eps) if cls_use_norm else nn.Identity()
            self.cls_dropout = nn.Dropout(cls_dropout) if cls_dropout > 0 else nn.Identity()
            self.classifier = nn.Linear(d_model, self.num_labels)
        else:
            self.cls_norm = None
            self.cls_dropout = None
            self.classifier = None

        self._tied = False
        if tie_embed:
            self.tie_embeddings(True)

        self._reset_parameters()

    # ---- utils ----
    def _reset_parameters(self):
        std = self.d_model ** -0.5
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=std)
        if not self._tied:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    def tie_embeddings(self, enable: bool = True):
        self._tied = enable
        if enable:
            self.lm_head.weight = self.token_embed.weight
        else:
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.detach().clone())

    def resize_token_embeddings(self, new_size: int):
        old_weight = self.token_embed.weight.data
        d = old_weight.size(1)
        new_weight = old_weight.new_empty(new_size, d)
        std = self.d_model ** -0.5
        new_weight.normal_(mean=0.0, std=std)
        num = min(old_weight.size(0), new_size)
        new_weight[:num] = old_weight[:num]
        self.token_embed.weight = nn.Parameter(new_weight)
        self.vocab_size = new_size
        if self._tied:
            self.lm_head.weight = self.token_embed.weight
        else:
            old_lm = self.lm_head.weight.data
            new_lm = old_lm.new_empty(new_size, old_lm.size(1))
            new_lm.normal_(mean=0.0, std=std)
            num_lm = min(old_lm.size(0), new_size)
            new_lm[:num_lm] = old_lm[:num_lm]
            self.lm_head.weight = nn.Parameter(new_lm)

    def ensure_rope_len(self, need_len: int):
        if not self.use_rope or self.rope_cos is None:
            return
        if need_len <= self.max_len_rope:
            return
        hd = self.d_model // self.n_heads
        cos_new, sin_new = build_rope_cache(
            hd, need_len, base=self.rope_base,
            device=self.rope_cos.device, dtype=self.rope_cos.dtype
        )
        self.rope_cos = cos_new
        self.rope_sin = sin_new
        self.max_len_rope = need_len

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def _maybe_ckpt(self,
                    layer: EncoderLayer,
                    x: torch.Tensor,
                    rope,
                    pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.grad_ckpt_enc or not self.training:
            return layer(x, rope=rope, pad_mask=pad_mask)

        def fn(inp):
            return layer(inp, rope=rope, pad_mask=pad_mask)
        return ckpt(fn, x)

    # ---- encode / lm ----
    def encode(self,
               input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape

        x = self.token_embed(input_ids) * (self.d_model ** 0.5)

        if self.pos_embedding_type == "learned":
            if T > self.max_position_embeddings:
                raise ValueError(f"seq_len {T} > max_position_embeddings {self.max_position_embeddings}")
            pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
            x = x + self.pos_embed(pos_ids)
        elif self.pos_embedding_type == "sinusoidal":
            if T > self.max_position_embeddings:
                raise ValueError(f"seq_len {T} > max_position_embeddings {self.max_position_embeddings}")
            pos = self.pos_sinus[:T].unsqueeze(0)
            x = x + pos

        x = self.embed_drop(x)

        rope = (self.rope_cos, self.rope_sin) if self.use_rope else None
        if rope is not None:
            self.ensure_rope_len(T)

        if attention_mask is None:
            token_mask = (input_ids != self.pad_id).long()
        else:
            if attention_mask.dtype == torch.bool:
                token_mask = attention_mask.long()
            else:
                token_mask = (attention_mask != 0).long()

        pad_mask = token_mask

        for layer in self.encoder_layers:
            x = self._maybe_ckpt(layer, x, rope=rope, pad_mask=pad_mask)

        return self.enc_ln(x)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        enc = self.encode(input_ids, attention_mask)
        return self.lm_head(enc)

    def compute_loss(self,
                     logits: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
        B, T, V = logits.shape
        assert labels.shape == (B, T)
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            labels.reshape(B * T),
            ignore_index=self.pad_id
        )
        return loss

    # ---- pooling / classification ----
    def pooled(self,
               enc_out: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               pool_type: Optional[str] = None,
               input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pool_type is None:
            pool_type = self.pool_type

        B, T, D = enc_out.shape

        if pool_type == "cls":
            if self.cls_id is not None and input_ids is not None:
                cls_mask = (input_ids == self.cls_id)
                has_cls = cls_mask.any(dim=1)

                if self.cls_require_presence and not has_cls.all():
                    missing = (~has_cls).nonzero(as_tuple=False).squeeze(-1)
                    raise ValueError(
                        f"Some sequences have no CLS token (cls_id={self.cls_id}), "
                        f"missing batch indices: {missing.tolist()}"
                    )

                idx = torch.zeros(B, dtype=torch.long, device=enc_out.device)
                if has_cls.any():
                    first_idx = cls_mask.float().argmax(dim=1)
                    idx = torch.where(has_cls, first_idx, idx)
                return enc_out[torch.arange(B, device=enc_out.device), idx, :]

            return enc_out[:, 0, :]

        elif pool_type == "mean":
            if attention_mask is None:
                return enc_out.mean(dim=1)
            if attention_mask.dtype == torch.bool:
                token_mask = attention_mask.to(enc_out.dtype)
            else:
                token_mask = (attention_mask != 0).to(enc_out.dtype)
            token_mask = token_mask.unsqueeze(-1)
            summed = (enc_out * token_mask).sum(dim=1)
            counts = token_mask.sum(dim=1).clamp_min(1e-6)
            return summed / counts

        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

    def forward_cls(self,
                    input_ids: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None):
        assert self.classifier is not None, "num_labels=0 iken forward_cls kullanılamaz."
        enc = self.encode(input_ids, attention_mask)
        pooled = self.pooled(enc, attention_mask, input_ids=input_ids)
        if self.cls_norm is not None:
            pooled = self.cls_norm(pooled)
        if self.cls_dropout is not None:
            pooled = self.cls_dropout(pooled)
        logits = self.classifier(pooled)
        return logits, pooled

    def compute_cls_loss(self,
                         logits: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


TransformerEnc = HysoEncoderOnly