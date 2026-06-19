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

The **default model is `bash-v2`** — a larger NanoGPT (n_embd 512, 8 heads/layers) trained
on a refreshed bash dataset with a richer 202-token word/symbol tokenizer. It's `gpt2`-shape
(no llama.cpp work) and markedly better than the v1 models (e.g. `commit message "add test"`
→ `git commit -m "add test"`). Each model carries its own vocab (`<model>.vocab.txt`), and
the runner's tokenizer is vocab-driven, so the char-level (124) and word-level (202)
tokenizers coexist. Select a model with `QQ_MODEL=…/denselogic-bash.gguf`.

Both phases are **verified token-for-token against PyTorch** (Phase 2: 12/12; bash-v2: 12/12).
All of DenseLogic's exotic pieces (skip-gate softmax, EMA decay
kernel) are *static at inference* and baked into the GGUF, so **no new ggml kernels** were
needed — the new graph is assembled from existing ops (`ggml_cumsum` for the gated running
mean, `ggml_conv_1d_dw` for the multi-scale EMA, RoPE/RMSNorm/SwiGLU for the rest). The
`logicsim` mixer is recurrent, so the runner decodes **cache-free** (re-runs the full
sequence each step — microseconds for this model).

> Note: hh_llm's live tokenizer drifted (124→202 tokens) mid-development. qq_terminal is
> immune because it pins a frozen `tokenizer_spec.json` / `vocab.txt` matching the
> checkpoint — exactly why the export step is decoupled.

## Install

Prereqs: `git`, `cmake`, a C++17 compiler, `curl`.

```bash
git clone --recursive https://github.com/thomasreolon/CmdLanguageModel.git && cd CmdLanguageModel
./setup.sh                 # patch llama.cpp, build qq-llm, download the models
./terminal/install.sh      # add the !! widget to ~/.zshrc
# open a new zsh, then:  !! list png files modified today
```

`setup.sh` is idempotent — re-run it any time. If you forgot `--recursive`, run
`git submodule update --init` first. Models are fetched from a GitHub Release; point
`setup.sh` at yours with `QQ_RELEASE_URL=...` (or just run `convert_to_gguf.py` if you
have the `hh_llm` checkpoints).

### Publishing models (maintainer)

The `.gguf` files (~172 MB) are gitignored — distribute them via a GitHub Release:

```bash
# build them from hh_llm checkpoints (see docs/pytorch-to-gguf.md), then:
gh release create v1 llm/models/*.gguf
# set QQ_RELEASE_URL in setup.sh to .../releases/latest/download
```

The small per-model `*.vocab.txt` files *are* committed, so `setup.sh` knows which
models to fetch. See `docs/pytorch-to-gguf.md` for the conversion details.

> Status: both phases implemented and verified end-to-end (token-for-token vs PyTorch).
> Default model is **bash-v2**; `denselogic` and `nanogpt` are selectable via `QQ_MODEL`.
