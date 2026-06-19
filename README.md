# qq_terminal

A normal shell with one tweak: type a line beginning with **`!!`** and, instead of
running it, your **local** language model rewrites the line into a shell command and
drops it back into the prompt — **editable**. Review it, tweak it, press Enter to run.

```
!! list png files modified today
        ↓  (local model, no network)
find . -name '*.png' -mtime 0      ← editable; press Enter to run, Ctrl-C to discard
```

The model is one you trained yourself (in [`hh_llm`](../hh_llm)) on a bash
description→command task, exported to GGUF and run via llama.cpp. Everything is local.

## Layout

```
terminal/   the "!!" tweak: a zsh ZLE widget (qq.zsh) + installer
llm/
  export/   convert_to_gguf.py + tokenizer_spec.json  (dev-time bridge to hh_llm)
  runner/   qq-llm: a small C++ binary linking libllama (tokenize + greedy decode)
  llama.cpp/  git submodule (Phase 2 adds the `denselogic` arch here)
  models/   exported .gguf files
docs/       pytorch-to-gguf.md — how to turn an hh_llm checkpoint into a runnable GGUF
```

The only coupling to `hh_llm` is the one-time **export** step. After you run it,
qq_terminal is self-contained.

## How it works

1. **terminal/qq.zsh** binds Enter. If the buffer starts with `!!`, it calls
   `llm/runner/qq-llm "<rest of line>"` and replaces the buffer with the output
   (without executing). Otherwise the line runs as usual — it's still your real shell,
   so pipes, `cd`, history, vim, etc. all work.
2. **llm/runner/qq-llm** is the backend contract: request on argv → one command on
   stdout. It tokenizes `<bash><in>{request}<out>` with our 124-token structural
   tokenizer, feeds the ids to llama.cpp, and greedy-decodes until `<eos>`.

## Two phases

| Phase | Model | llama.cpp work | Bash acc. | Status |
|---|---|---|---|---|
| **1** | NanoGPT (stock GPT-2 shape) | **~none** — uses the existing `gpt2` arch | 38.50% | ✅ done |
| **2** | DenseLogic (custom `logicsim` mixer + dense skips) | new `denselogic` arch | 38.86% | ✅ done |

Both phases are **verified token-for-token against PyTorch** (Phase 2: 12/12 prompts).
DenseLogic is the default model. All of its exotic pieces (skip-gate softmax, EMA decay
kernel) are *static at inference* and baked into the GGUF, so **no new ggml kernels** were
needed — the new graph is assembled from existing ops (`ggml_cumsum` for the gated running
mean, `ggml_conv_1d_dw` for the multi-scale EMA, RoPE/RMSNorm/SwiGLU for the rest). The
`logicsim` mixer is recurrent, so the runner decodes **cache-free** (re-runs the full
sequence each step — microseconds for this model).

> Note: hh_llm's live tokenizer drifted (124→202 tokens) mid-development. qq_terminal is
> immune because it pins a frozen `tokenizer_spec.json` / `vocab.txt` matching the
> checkpoint — exactly why the export step is decoupled.

## Setup (Phase 1)

```bash
# 1. get llama.cpp + apply the denselogic arch patch
git submodule update --init llm/llama.cpp        # (or `git submodule add <url> llm/llama.cpp` first time)
git -C llm/llama.cpp apply ../denselogic-arch.patch   # adds the LLM_ARCH_DENSELOGIC arch

# 2. export the model  (uses hh_llm's venv to read the checkpoint)
cd llm/export
/home/thomas/hh_llm/.venv/bin/python convert_to_gguf.py \
    --ckpt /home/thomas/hh_llm/checkpoints/nanogpt/best.pt \
    --spec tokenizer_spec.json --out ../models/nanogpt-bash.gguf

# 3. build the runner
cmake -S llm/runner -B llm/runner/build && cmake --build llm/runner/build

# 4. install the shell tweak
terminal/install.sh   # appends `source .../qq.zsh` to ~/.zshrc
```

See `docs/pytorch-to-gguf.md` for the conversion details.

> Status: both phases implemented and verified end-to-end. `terminal/qq.zsh` (the `!!`
> tweak), `llm/export/convert_to_gguf.py` (gpt2 + denselogic), the libllama runner
> `llm/runner/qq-llm.cpp`, and the custom `denselogic` arch in `llm/llama.cpp` are all
> working. Build with the steps above and `terminal/install.sh`.
