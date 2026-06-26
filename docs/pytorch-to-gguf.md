# From PyTorch (hh_llm) to a runnable GGUF

How an `hh_llm` `logicsim_v2` checkpoint becomes a file llama.cpp can run — a custom
architecture exported **without writing a single new ggml/CUDA kernel**.

## The three things every conversion must carry

1. **Weights** — remap the PyTorch `state_dict` keys to llama.cpp's tensor-name scheme.
2. **Hyperparameters** — `hh_llm` checkpoints store *only* `{model, step, val}`; the
   config lives in the YAML. So the converter takes hparams as flags.
3. **Tokenizer** — our `StructuralTokenizer` (202 tokens, single-token `<...>` tags,
   greedy multi-char symbols) is not BPE/SPM. We do **not** push it through llama.cpp's
   tokenizer: `tokenizer_spec.json` is generated from `hh_llm`, the converter writes a
   per-model `<model>.vocab.txt` next to the `.gguf`, and **the runner tokenizes itself**,
   feeding token ids straight to `llama_decode`. The GGUF's vocab type is `none`.

## The model

`logicsim_v2` is a plain prenorm residual stack (no dense skips). Every block's mixer
fuses, on an asymmetric channel split, two branches that are concatenated through one
outer projection, followed by a SwiGLU MLP. RMSNorm throughout; RoPE on the attention
branch only.

- **DifferentialAttention** branch — first `c1` channels (75%). Two interleaved-head
  causal attentions `a1`, `a2`; output `y = a1 − λ·a2` with a per-layer scalar `λ`,
  then a sub-RMSNorm.
- **LogicSim** branch — last `c2` channels (25%). A gated cumulative running-mean
  (cumsum) plus a multi-scale EMA (depthwise causal conv), an ALU fusion, and SiLU×sigmoid
  gating.

## Why no new kernels are needed

The only "exotic" pieces are **static at inference**, so the converter bakes them into
plain tensors:

| Piece | At inference | Becomes |
|---|---|---|
| DiffAttn `λ` = `exp(q1·k1) − exp(q2·k2) + λ_init` | a single scalar per layer | a `[1]` tensor `blk.l.da_lambda` |
| LogicSim `decay_logit` (per-channel) | builds the EMA kernel | a precomputed `[c2,1,K]` causal conv kernel `blk.l.logic_ema_k` |
| counter port | gated cumulative mean over time | `ggml_cumsum` (lower-triangular running mean) |
| EMA port | depthwise causal conv, K=block_size | `ggml_conv_1d_dw` |
| ALU / gates / attn / norm / mlp | — | existing ops (`silu`, `sigmoid`, `mul`, `mul_mat`, rope, rms_norm, soft_max) |

So the whole conversion is: (a) `convert_to_gguf.py` emits the baked constants plus the
branch tensors under the `logicsim_v2` arch; (b) the patched llama.cpp assembles the graph
from the ops above. No new kernels.

## The llama.cpp touch points (`llm/logicsim_v2-arch.patch`)

- `src/llama-arch.{h,cpp}` — `LLM_ARCH_LOGICSIM_V2`, the custom tensor enums
  (`da_qkv/da_out/da_subln/da_lambda`, `logic_in/alu/out/ema_k`, `mix_out`), their names
  and op-infos.
- `src/llama-model.h` — layer fields for the branch tensors.
- `src/llama-model.cpp` — model factory + rope type (`NEOX` — rotate-half RoPE, no weight
  permutation).
- `src/llama-context.cpp` — a larger `graph_max_nodes` budget (the per-block expansion
  emits ~100 ggml nodes/layer, far above the generic estimate).
- `src/models/logicsim_v2.cpp` — `load_arch_tensors` and the graph: the channel split, the
  DifferentialAttention branch (cache-free causal attention via `ggml_diag_mask_inf` +
  `ggml_soft_max`), the LogicSim branch, the outer projection, and the SwiGLU MLP.

Gotchas worth knowing: `ggml_cumsum` runs on `ne0`, so transpose time into `ne0` and back;
`ggml_conv_1d_dw` wants the kernel in **F16** with left-pad `K−1` for causality; a baked
tensor's op-info hint must not be `GGML_OP_SSM_CONV` (its load-time probe asserts a shape
ours doesn't match — use `GGML_OP_MUL`). Because LogicSim is recurrent, the runner decodes
**cache-free** (`llama_memory_clear` + re-decode the full sequence each step — microseconds
for a 25M model at ≤256 tokens).

## Running the conversion

```bash
python llm/export/convert_to_gguf.py \
    --ckpt /path/to/hh_llm/checkpoints/bash_logicsim_v2_25m/best.pt \
    --spec llm/export/tokenizer_spec.json \
    --n-embd 512 --n-layer 8 --block-size 256 \
    --out llm/models/logicsim_v2-bash.gguf
```

**Verified token-for-token against PyTorch (12/12 prompts, fp32 greedy decode).**

## Reproducing the tokenizer spec

`tokenizer_spec.json` is generated from `hh_llm`, never hand-edited:

```python
from datasets.tokenizers.structural import StructuralTokenizer
t = StructuralTokenizer(add_bash=True)   # 202 tokens (word/symbol-level, bash vocab)
```

The runner mirrors `encode`: on `<` consume to `>` as one token; else greedily match the
longest multi-char symbol; else one character.
