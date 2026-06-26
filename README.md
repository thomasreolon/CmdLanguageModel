<div align="center">

```
       ██╗██╗    ████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗
      ██╔╝██║    ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║
     ██╔╝ ██║       ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║
    ██╔╝  ██║       ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║
   ██╔╝   ██║       ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗
   ╚═╝    ╚═╝       ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
```

### your shell, plus one key — describe the command, get the command

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![100% local](https://img.shields.io/badge/inference-100%25%20local-2ea44f)](#how-it-works)
[![runtime: llama.cpp](https://img.shields.io/badge/runtime-llama.cpp-blue)](https://github.com/ggml-org/llama.cpp)
[![model: logicsim__v2](https://img.shields.io/badge/model-logicsim__v2%20·%2025M-orange)](docs/pytorch-to-gguf.md)
[![verified: 12/12 vs PyTorch](https://img.shields.io/badge/verified-12%2F12%20vs%20PyTorch-success)](#the-model)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](llm/runner/qq-llm.cpp)

</div>

---

A normal shell with one tweak: type a line beginning with **`!!`** and, instead of
running it, your **local** language model rewrites the line into a shell command and
drops it back into the prompt — **editable**. Review it, tweak it, press Enter to run.

```console
$ !! list png files modified today
        ↓  (local model — no network, no API key)
$ find . -name '*.png' -mtime 0      ← editable; press Enter to run, Ctrl-C to discard
```

It's still your real shell, so `cd`, pipes, history, vim — everything works. The only
thing that changed is what `!!` does to the current line.

```console
$ !! show running docker containers          →  docker ps
$ !! find files larger than 100MB             →  find . -type f -size +100M
$ !! disk usage of current directory          →  du -sh *
$ !! list all files including hidden ones      →  ls -la
```

## Quick start

Prereqs: `git`, `cmake`, a C++17 compiler, `curl`, and `zsh`.

```bash
git clone --recursive https://github.com/thomasreolon/CmdLanguageModel.git
cd CmdLanguageModel
./setup.sh                 # patch llama.cpp, build the runner, download the model
./terminal/install.sh      # add the !! widget to ~/.zshrc
exec zsh                   # then try:  !! list png files modified today
```

`setup.sh` is idempotent — re-run it any time. Forgot `--recursive`? Run
`git submodule update --init` first.

## How it works

Three small pieces, no network anywhere:

```
  ~/.zshrc                  llm/runner/qq-llm            llm/llama.cpp (patched)
  ┌─────────────┐  "!! …"   ┌───────────────────┐  ids  ┌────────────────────┐
  │ qq.zsh ZLE  │ ────────► │ structural tokenizer ───► │ logicsim_v2 arch   │
  │ binds Enter │ ◄──────── │ + greedy decode     ◄──── │ (cache-free decode)│
  └─────────────┘  command  └───────────────────┘ logits└────────────────────┘
```

1. **`terminal/qq.zsh`** binds Enter. If the buffer starts with `!!`, it calls
   `qq-llm "<rest of line>"` and replaces the buffer with the output (without executing).
   Any other line runs as usual.
2. **`llm/runner/qq-llm`** is a small C++ binary linking `libllama`. It frames the request
   as `<bash><in>{request}<out>`, tokenizes it with our 202-token **structural tokenizer**
   (the model's vocab drives encoding — no BPE), feeds the ids to llama.cpp, and
   greedy-decodes until `<eos>`.
3. **`llm/llama.cpp`** runs the model. The custom architecture is added as a small,
   self-contained patch over upstream — see below.

## The model

The default model is **`logicsim_v2-bash`** (~25M params), trained from scratch in
[`hh_llm`](https://github.com/thomasreolon) on a natural-language→bash task with a custom
202-token tokenizer, then exported to GGUF. It is **not** a stock Transformer — each block
splits its channels between two parallel branches:

- a **DifferentialAttention** branch (`y = a1 − λ·a2`, attention-noise-cancelling), and
- a **LogicSim** branch (a gated cumulative running-mean + multi-scale EMA recurrence),

concatenated through one outer projection, with RMSNorm + SwiGLU + RoPE. All of its exotic
pieces are *static at inference* and **baked into the GGUF**, so the new graph is assembled
from existing ggml ops — **no new ggml/CUDA kernels were written**. Because the LogicSim
branch is recurrent, the runner decodes **cache-free** (re-runs the full sequence each
step — microseconds at this size).

The export is **verified token-for-token against PyTorch: 12/12 prompts, exact** (fp32
greedy). The whole conversion + llama.cpp integration is documented in
[`docs/pytorch-to-gguf.md`](docs/pytorch-to-gguf.md).

> The custom architecture lives in **`llm/logicsim_v2-arch.patch`**, applied to a pinned
> upstream `llama.cpp` submodule by `setup.sh` — so this repo never forks llama.cpp.

## Layout

```
terminal/    the "!!" tweak: a zsh ZLE widget (qq.zsh) + idempotent installer
llm/
  logicsim_v2-arch.patch   the custom arch, as a patch over upstream llama.cpp
  llama.cpp/               git submodule (pinned upstream; patched at setup time)
  runner/                  qq-llm: C++ binary linking libllama (tokenize + greedy decode)
  export/                  convert_to_gguf.py + tokenizer_spec.json (hh_llm → GGUF bridge)
  models/                  the exported .gguf (+ its vocab.txt); .gguf fetched from Releases
docs/        pytorch-to-gguf.md — how an hh_llm checkpoint becomes a runnable GGUF
setup.sh     one-command build; install.sh adds the widget to ~/.zshrc
```

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `QQ_MODEL` | `llm/models/logicsim_v2-bash.gguf` | path to the GGUF to run |
| `QQ_VOCAB` | `<model>.vocab.txt` next to the GGUF | structural tokenizer vocab |
| `QQ_LLM` | resolved next to `qq.zsh` | path to the `qq-llm` backend |
| `QQ_RELEASE_URL` | this repo's latest Release | where `setup.sh` fetches the `.gguf` |

## Maintainer notes

The `.gguf` (~99 MB) is **gitignored** and distributed via a GitHub Release; the small
`*.vocab.txt` is committed so `setup.sh` knows what to fetch. To (re)publish:

```bash
# build the GGUF from an hh_llm checkpoint (see docs/pytorch-to-gguf.md), then:
gh release create v2 llm/models/logicsim_v2-bash.gguf --title "logicsim_v2-bash" \
  --notes "25M logicsim_v2 NL→bash model (GGUF). Run with setup.sh."
```

`setup.sh` points at `releases/latest/download`, so a normal `gh release create` is picked
up automatically.

## License

[MIT](LICENSE) © Thomas Reolon. Bundles [llama.cpp](https://github.com/ggml-org/llama.cpp)
(MIT) as a submodule.
