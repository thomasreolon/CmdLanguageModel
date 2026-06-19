# From PyTorch (hh_llm) to a runnable GGUF

How an `hh_llm` checkpoint becomes a file llama.cpp can run. Two cases: the easy one
(Phase 1, NanoGPT) and the custom-architecture one (Phase 2, DenseLogic).

## The three things every conversion must carry

1. **Weights** — remap PyTorch `state_dict` keys to llama.cpp's tensor-name scheme.
2. **Hyperparameters** — `hh_llm` checkpoints store *only* `{model, step, val}`; the
   config lives in the YAML. So the converter takes hparams as flags/dict.
3. **Tokenizer** — our `StructuralTokenizer` (124 tokens, single-token `<...>` tags,
   greedy `->`) is not BPE/SPM. We do **not** push it through llama.cpp's tokenizer:
   `tokenizer_spec.json` is generated from `hh_llm` and the **runner tokenizes itself**,
   feeding token ids to `llama_decode`. The GGUF carries the vocab only for metadata/detok.

## Phase 1 — NanoGPT → `gpt2` (no llama.cpp changes)

The NanoGPT bash model is structurally stock GPT-2: learned absolute positions,
LayerNorm, fused QKV (`c_attn`), 4× GELU MLP, tied head. It maps onto llama.cpp's
existing `gpt2` architecture. Tensor map (see `convert_to_gguf.py`):

| hh_llm key | GGUF (gpt2) name |
|---|---|
| `wte.weight` | `token_embd.weight` |
| `wpe.weight` | `position_embd.weight` |
| `ln_f.weight` | `output_norm.weight` |
| `lm_head.weight` | `output.weight` (tied to wte) |
| `blocks.{i}.norm1.weight` | `blk.{i}.attn_norm.weight` |
| `blocks.{i}.mixer.c_attn.weight` | `blk.{i}.attn_qkv.weight` |
| `blocks.{i}.mixer.c_proj.weight` | `blk.{i}.attn_output.weight` |
| `blocks.{i}.norm2.weight` | `blk.{i}.ffn_norm.weight` |
| `blocks.{i}.mlp.c_fc.weight` | `blk.{i}.ffn_up.weight` |
| `blocks.{i}.mlp.c_proj.weight` | `blk.{i}.ffn_down.weight` |

40 tensors total — verified all 40 map. **Open item:** our model is bias-free, but
the stock `gpt2` loader expects LayerNorm/QKV/proj biases. Plan: synthesize zero-filled
bias tensors at export. Confirm whether that's actually required on first load.

## Phase 2 — DenseLogic → new `denselogic` arch

DenseLogic is a custom top-level assembler: 6 blocks alternating attention and the
invented `logicsim` mixer, with **dense skip connections** (block *l* reads a learned
softmax-weighted combo of all earlier states). RoPE, RMSNorm, SwiGLU.

The key realization that makes this tractable **without writing new ggml/CUDA kernels**:
everything exotic is *static at inference* and is baked into the GGUF at export.

| Piece | At inference | Becomes |
|---|---|---|
| `skip_gates.{l}` (sizes 1..6) | `softmax()` of a parameter | precomputed constant weight vectors → weighted `add` |
| `decay_logit` (per-channel) | builds the EMA kernel | precompute the `[C,1,K]` exp kernel tensor |
| counter port | gated cumulative mean over time | cumsum = lower-triangular ones-matmul (constant) |
| EMA port | depthwise causal conv, K=256 | `ggml_conv_1d` depthwise (or reuse Mamba's `ssm_conv`) |
| ALU / gates / attn / norm / mlp | — | existing ops (`silu`, `sigmoid`, `mul`, `mul_mat`, rope, rms_norm) |

So Phase 2 work is: (a) extend `convert_to_gguf.py` to emit the baked constants and the
logicsim/attention/skip tensors under a `denselogic` arch; (b) register `denselogic` in
llama.cpp (`llama-arch`, model load, and a `build_denselogic` graph function) assembled
from the ops above. No new kernels.

**Implemented and verified (12/12 prompts match PyTorch).** The llama.cpp touch points:

- `src/llama-arch.{h,cpp}` — `LLM_ARCH_DENSELOGIC`, the 5 custom tensor enums
  (`skip_w`, `logic_in/alu/out/ema_k`), their names and op-infos.
- `src/llama-model.h` — layer fields for the logicsim tensors.
- `src/llama-model.cpp` — model factory, rope type (`NEOX` — the lab uses rotate-half
  RoPE and we don't permute weights, so NEOX matches).
- `src/models/denselogic.cpp` — `load_arch_tensors` (attention vs logicsim per layer) and
  the graph: dense-skip convex combine, RoPE attention + SwiGLU, the logicsim mixer.

Gotchas hit along the way: `ggml_cumsum` runs on `ne0`, so transpose time into `ne0`
and back; `ggml_conv_1d_dw` wants the kernel in **F16** (cast it) with data `[L,C]` and
left-pad `K-1` for causality; the tensor's op-info hint must not be `GGML_OP_SSM_CONV`
(its load-time op-support probe asserts a shape ours doesn't match — use `GGML_OP_MUL`).
Because logicsim is recurrent, the runner decodes cache-free (`llama_memory_clear` +
re-decode the full sequence each step).

## Reproducing the tokenizer spec

`tokenizer_spec.json` is generated, never hand-edited:

```python
from datasets.tokenizers.structural import StructuralTokenizer
t = StructuralTokenizer(add_bash=True)   # 124 tokens; <eos>=1, <bash>=2, <in>=28, <out>=30
```

The runner mirrors `encode`: on `<` consume to `>` as one token; else greedily match a
`multi` symbol (`->`); else one character.
