#!/usr/bin/env python3
"""Export an hh_llm `logicsim_v2` checkpoint to a llama.cpp GGUF.

This is the ONLY place ShellWhisper touches hh_llm: it reads the checkpoint, bakes the
inference-static pieces into plain tensors, and writes a self-contained .gguf that the
patched llama.cpp (`logicsim_v2` arch) can load. The tokenizer is NOT embedded
(vocab type "none"); the runner tokenizes from the per-model `<model>.vocab.txt`
(written next to the .gguf here) and feeds token ids directly.

The model is a plain prenorm residual stack where every block's mixer fuses, on an
asymmetric channel split, a DifferentialAttention branch (first c1 channels) and a
LogicSim branch (last c2 channels), concatenated through one outer projection, then a
SwiGLU MLP. RMSNorm + RoPE (rope only on the DiffAttn branch).
See hh_llm models/components/mixers/logicsim_v2.py.

Usage:
    python convert_to_gguf.py \
        --ckpt /home/thomas/hh_llm/checkpoints/bash_logicsim_v2_25m/best.pt \
        --spec tokenizer_spec.json \
        --n-embd 512 --n-layer 8 --block-size 256 \
        --out ../models/logicsim_v2-bash.gguf
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import gguf

LAMBDA_INIT = 0.8


def f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(torch.float32).numpy())


def write_logicsim_v2(sd, spec, a) -> int:
    """Inference-static pieces baked here (no new ggml kernels needed):
      - DiffAttn `lambda` = exp(q1·k1) - exp(q2·k2) + lambda_init  -> ONE scalar per layer.
      - LogicSim `decay_logit` -> the [c2,1,K] causal EMA conv kernel.
    Geometry is derived from weight shapes: c1 (attn) = c_attn.in; c2 (logic) = in_proj.in;
    head_dim = subln.len/2; n_head1 = c1/(2*head_dim).
    """
    K = a.block_size
    # derive split geometry from block 0
    c1 = sd["blocks.0.mixer.attn_branch.c_attn.weight"].shape[1]     # attn branch width
    c2 = sd["blocks.0.mixer.logic_branch.in_proj.weight"].shape[1]   # logic branch width
    head_dim = sd["blocks.0.mixer.attn_branch.subln.weight"].shape[0] // 2
    n_head1 = c1 // (2 * head_dim)
    assert c1 + c2 == a.n_embd, f"c1{c1}+c2{c2} != n_embd{a.n_embd}"
    assert n_head1 * 2 * head_dim == c1, "head geometry mismatch"
    n_ff = sd["blocks.0.mlp.w_gate.weight"].shape[0]

    w = gguf.GGUFWriter(a.out, "logicsim_v2")
    w.add_name("logicsim_v2-bash")
    w.add_context_length(a.block_size)
    w.add_embedding_length(a.n_embd)
    w.add_block_count(a.n_layer)
    w.add_head_count(n_head1)                  # DiffAttn branch head count (v-heads)
    w.add_head_count_kv(n_head1)
    w.add_key_length(head_dim)                  # q/k head width (32); c1 = 2*n_head*key_len
    w.add_value_length(2 * head_dim)            # v head width (64)
    w.add_feed_forward_length(n_ff)
    w.add_rope_dimension_count(head_dim)        # rope rotates the full (32-wide) q/k head
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(spec["vocab_size"])
    w.add_tokenizer_model("none")

    T = {}
    T["token_embd.weight"] = f32(sd["wte.weight"])     # head is tied -> reused as output
    T["output_norm.weight"] = f32(sd["ln_f.weight"])
    for i in range(a.n_layer):
        p = f"blocks.{i}."
        m = p + "mixer."
        T[f"blk.{i}.attn_norm.weight"] = f32(sd[p + "norm1.weight"])
        # --- DiffAttn branch (c1) ---
        T[f"blk.{i}.da_qkv.weight"]   = f32(sd[m + "attn_branch.c_attn.weight"])
        T[f"blk.{i}.da_out.weight"]   = f32(sd[m + "attn_branch.out_proj.weight"])
        T[f"blk.{i}.da_subln.weight"] = f32(sd[m + "attn_branch.subln.weight"])
        q1, k1 = sd[m + "attn_branch.lambda_q1"], sd[m + "attn_branch.lambda_k1"]
        q2, k2 = sd[m + "attn_branch.lambda_q2"], sd[m + "attn_branch.lambda_k2"]
        lam = (torch.exp(torch.dot(q1, k1)) - torch.exp(torch.dot(q2, k2)) + LAMBDA_INIT)
        T[f"blk.{i}.da_lambda.weight"] = np.asarray([float(lam)], dtype=np.float32)
        # --- LogicSim branch (c2) ---
        T[f"blk.{i}.logic_in.weight"]  = f32(sd[m + "logic_branch.in_proj.weight"])
        T[f"blk.{i}.logic_alu.weight"] = f32(sd[m + "logic_branch.alu.weight"])
        T[f"blk.{i}.logic_out.weight"] = f32(sd[m + "logic_branch.out_proj.weight"])
        lamc = torch.sigmoid(sd[m + "logic_branch.decay_logit"].float())   # [c2]
        mm = torch.arange(K - 1, -1, -1).float()
        ker = (1 - lamc)[:, None] * lamc[:, None] ** mm[None, :]            # [c2, K]
        T[f"blk.{i}.logic_ema_k.weight"] = f32(ker[:, None, :])            # [c2,1,K]
        # --- outer post-concat projection + SwiGLU MLP ---
        T[f"blk.{i}.mix_out.weight"]  = f32(sd[m + "c_proj.weight"])
        T[f"blk.{i}.ffn_norm.weight"] = f32(sd[p + "norm2.weight"])
        T[f"blk.{i}.ffn_gate.weight"] = f32(sd[p + "mlp.w_gate.weight"])
        T[f"blk.{i}.ffn_up.weight"]   = f32(sd[p + "mlp.w_up.weight"])
        T[f"blk.{i}.ffn_down.weight"] = f32(sd[p + "mlp.c_proj.weight"])

    for name, arr in T.items():
        w.add_tensor(name, arr)
    print(f"[convert] logicsim_v2: c1={c1} c2={c2} n_head1={n_head1} head_dim={head_dim} "
          f"n_ff={n_ff}; writing {len(T)} tensors -> {a.out}")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("[convert] done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spec", default="tokenizer_spec.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    a = ap.parse_args()

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
    sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    spec = json.loads(Path(a.spec).read_text())
    # write the model's vocab next to the .gguf (one token per line; the runner reads it).
    Path(a.out).with_suffix(".vocab.txt").write_text("".join(t + "\n" for t in spec["itos"]))
    return write_logicsim_v2(sd, spec, a)


if __name__ == "__main__":
    raise SystemExit(main())
