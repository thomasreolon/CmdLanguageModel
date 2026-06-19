#!/usr/bin/env python3
"""Export an hh_llm checkpoint to a llama.cpp GGUF.

Phase 1 target: the NanoGPT bash model -> llama.cpp's existing `gpt2` architecture
(learned abs-pos, LayerNorm, fused QKV, 4x GELU MLP, tied head). No llama.cpp changes.

This is the ONLY place qq_terminal touches hh_llm: it reads the checkpoint, remaps
tensor names to the GGUF/gpt2 scheme, and writes a self-contained .gguf. Our model is
bias-free but gpt2's loader requires biases, so we synthesize zero biases (exact). The
tokenizer is NOT embedded (vocab type "none"); the runner tokenizes from
tokenizer_spec.json and feeds token ids directly.

Usage:
    python convert_to_gguf.py \
        --ckpt /home/thomas/hh_llm/checkpoints/nanogpt/best.pt \
        --spec tokenizer_spec.json --out ../models/nanogpt-bash.gguf
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import gguf

# nanogpt state_dict key -> GGUF tensor name (per-block uses {i}).
GLOBAL = {
    "wte.weight":  "token_embd.weight",
    "wpe.weight":  "position_embd.weight",
    "ln_f.weight": "output_norm.weight",
}
BLOCK = {
    "norm1.weight":        "blk.{i}.attn_norm.weight",
    "mixer.c_attn.weight": "blk.{i}.attn_qkv.weight",
    "mixer.c_proj.weight": "blk.{i}.attn_output.weight",
    "norm2.weight":        "blk.{i}.ffn_norm.weight",
    "mlp.c_fc.weight":     "blk.{i}.ffn_up.weight",
    "mlp.c_proj.weight":   "blk.{i}.ffn_down.weight",
}
# GGUF weight name -> the zero-bias gpt2 also requires (length = that weight's rows).
ZERO_BIAS = {
    "output_norm.weight":       "output_norm.bias",
    "blk.{i}.attn_norm.weight": "blk.{i}.attn_norm.bias",
    "blk.{i}.attn_qkv.weight":  "blk.{i}.attn_qkv.bias",
    "blk.{i}.attn_output.weight":"blk.{i}.attn_output.bias",
    "blk.{i}.ffn_norm.weight":  "blk.{i}.ffn_norm.bias",
    "blk.{i}.ffn_up.weight":    "blk.{i}.ffn_up.bias",
    "blk.{i}.ffn_down.weight":  "blk.{i}.ffn_down.bias",
}


def f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(torch.float32).numpy())


def write_denselogic(sd, spec, a) -> int:
    """Phase 2: custom arch. Attention/logicsim alternate; dense skip connections.

    Everything exotic is STATIC at inference, so we bake it into plain tensors here:
      - skip_gates[l]  -> softmax weights  blk.l.skip_w        (len l+1)
      - decay_logit    -> EMA conv kernel  blk.l.logic_ema_k   (shape [K, C], causal)
    The loader then assembles a graph from standard ops (rope attn, rms_norm, swiglu,
    cumsum, depthwise conv). RMSNorm has no bias, so no bias synthesis is needed.
    """
    K = a.block_size
    w = gguf.GGUFWriter(a.out, "denselogic")
    w.add_name("denselogic-bash")
    w.add_context_length(a.block_size)
    w.add_embedding_length(a.n_embd)
    w.add_block_count(a.n_layer)
    w.add_head_count(a.n_head)
    w.add_head_count_kv(a.n_head)
    w.add_feed_forward_length(sd["blocks.0.mlp.w_gate.weight"].shape[0])  # swiglu hidden
    w.add_rope_dimension_count(a.n_embd // a.n_head)  # rope rotates full head dim
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(spec["vocab_size"])
    w.add_tokenizer_model("none")

    T = {}
    T["token_embd.weight"] = f32(sd["wte.weight"])
    T["output_norm.weight"] = f32(sd["ln_f.weight"])
    for i in range(a.n_layer):
        p = f"blocks.{i}."
        T[f"blk.{i}.attn_norm.weight"] = f32(sd[p + "norm1.weight"])
        # dense-skip convex weights over H_0..H_i  (softmax of the learned gates)
        g = sd[f"skip_gates.{i}"].float()
        T[f"blk.{i}.skip_w.weight"] = f32(torch.softmax(g, dim=0))
        if p + "mixer.c_attn.weight" in sd:                 # attention + swiglu block
            T[f"blk.{i}.attn_qkv.weight"] = f32(sd[p + "mixer.c_attn.weight"])
            T[f"blk.{i}.attn_output.weight"] = f32(sd[p + "mixer.c_proj.weight"])
            T[f"blk.{i}.ffn_norm.weight"] = f32(sd[p + "norm2.weight"])
            T[f"blk.{i}.ffn_gate.weight"] = f32(sd[p + "mlp.w_gate.weight"])
            T[f"blk.{i}.ffn_up.weight"] = f32(sd[p + "mlp.w_up.weight"])
            T[f"blk.{i}.ffn_down.weight"] = f32(sd[p + "mlp.c_proj.weight"])
        else:                                               # logicsim block (mlp:none)
            T[f"blk.{i}.logic_in.weight"] = f32(sd[p + "mixer.in_proj.weight"])
            T[f"blk.{i}.logic_alu.weight"] = f32(sd[p + "mixer.alu.weight"])
            T[f"blk.{i}.logic_out.weight"] = f32(sd[p + "mixer.c_proj.weight"])
            # bake EMA kernel: ker[c,0,k] = (1-lam_c)*lam_c^(K-1-k), lam=sigmoid(decay_logit).
            # Stored as numpy (C,1,K) so it lands as ggml ne [K,1,C] = what conv_1d_dw wants.
            lam = torch.sigmoid(sd[p + "mixer.decay_logit"].float())      # [C]
            m = torch.arange(K - 1, -1, -1).float()                       # K-1..0
            ker = (1 - lam)[:, None] * lam[:, None] ** m[None, :]         # [C, K]
            T[f"blk.{i}.logic_ema_k.weight"] = f32(ker[:, None, :])      # [C, 1, K]

    for name, arr in T.items():
        w.add_tensor(name, arr)
    print(f"[convert] denselogic: writing {len(T)} tensors -> {a.out}")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("[convert] done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spec", default="tokenizer_spec.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["gpt2", "denselogic"], default="gpt2")
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--block-size", type=int, default=256)
    a = ap.parse_args()

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
    spec = json.loads(Path(a.spec).read_text())
    # write the model's vocab next to the .gguf (one token per line; the runner reads it).
    Path(a.out).with_suffix(".vocab.txt").write_text("".join(t + "\n" for t in spec["itos"]))
    if a.arch == "denselogic":
        return write_denselogic(sd, spec, a)
    n_ff = sd["blocks.0.mlp.c_fc.weight"].shape[0]  # 4*n_embd, read from weights

    w = gguf.GGUFWriter(a.out, "gpt2")
    w.add_name("nanogpt-bash")
    w.add_context_length(a.block_size)
    w.add_embedding_length(a.n_embd)
    w.add_block_count(a.n_layer)
    w.add_head_count(a.n_head)
    w.add_head_count_kv(a.n_head)        # no GQA
    w.add_feed_forward_length(n_ff)
    w.add_layer_norm_eps(1e-5)           # nn.LayerNorm default
    w.add_vocab_size(spec["vocab_size"])
    w.add_tokenizer_model("none")        # runner tokenizes; no embedded vocab

    # real weights
    tensors: dict[str, np.ndarray] = {}
    for src, dst in GLOBAL.items():
        tensors[dst] = f32(sd[src])
    for i in range(a.n_layer):
        for src, dst in BLOCK.items():
            tensors[dst.format(i=i)] = f32(sd[f"blocks.{i}.{src}"])
    # synthesized zero biases (our model is bias-free; gpt2 loader requires them)
    for wname, bname in ZERO_BIAS.items():
        for i in range(a.n_layer) if "{i}" in wname else [None]:
            wn = wname.format(i=i) if i is not None else wname
            bn = bname.format(i=i) if i is not None else bname
            rows = tensors[wn].shape[0]          # bias length = weight's output dim
            tensors[bn] = np.zeros(rows, dtype=np.float32)

    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    print(f"[convert] writing {len(tensors)} tensors -> {a.out}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("[convert] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
