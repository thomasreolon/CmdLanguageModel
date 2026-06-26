<div align="center">

```text
███████╗██╗  ██╗███████╗██╗     ██╗     ██╗    ██╗██╗  ██╗██╗███████╗██████╗ ███████╗██████╗ 
██╔════╝██║  ██║██╔════╝██║     ██║     ██║    ██║██║  ██║██║██╔════╝██╔══██╗██╔════╝██╔══██╗
███████╗███████║█████╗  ██║     ██║     ██║ █╗ ██║███████║██║███████╗██████╔╝█████╗  ██████╔╝
╚════██║██╔══██║██╔══╝  ██║     ██║     ██║███╗██║██╔══██║██║╚════██║██╔═══╝ ██╔══╝  ██╔══██╗
███████║██║  ██║███████╗███████╗███████╗╚███╔███╔╝██║  ██║██║███████║██║     ███████╗██║  ██║
╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝
```

### 🔮 Your shell, plus one key — describe the command, get the command. 100% offline & instantaneous.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Inference: 100% Local](https://img.shields.io/badge/inference-100%25%20local-success.svg)](#how-it-works)
[![Runtime: llama.cpp](https://img.shields.io/badge/runtime-llama.cpp-blue.svg)](https://github.com/ggml-org/llama.cpp)
[![Model: logicsim_v2](https://img.shields.io/badge/model-logicsim__v2%20·%2025M-orange.svg)](docs/pytorch-to-gguf.md)
[![Verified: PyTorch](https://img.shields.io/badge/verified-12%2F12%20vs%20PyTorch-success)](#the-model)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](llm/runner/qq-llm.cpp)
[![Latency: <10ms](https://img.shields.io/badge/latency-%3C10ms-brightgreen.svg)](#the-model)

</div>

---

**ShellWhisper** is a normal shell with one powerful tweak: type a line beginning with **`!!`** and, instead of running it, your **local** language model rewrites the line into a shell command and drops it back into your prompt — **completely editable**. Review it, tweak it, and press Enter to run.

```console
$ !! list png files modified today
        ↓  (local model — no network, no API key, microsecond latency)
$ find . -name '*.png' -mtime 0      ← editable; press Enter to run, Ctrl-C to discard
```

It remains your real shell: `cd`, pipes, history, vim — everything works exactly as before. The only change is what `!!` does to the current line.

```console
$ !! show running docker containers          →  docker ps
$ !! find files larger than 100MB             →  find . -type f -size +100M
$ !! disk usage of current directory          →  du -sh *
$ !! list all files including hidden ones      →  ls -la
```

---

## ✨ Features

*   **100% Offline & Private:** Zero network requests, zero telemetry, and zero API keys. Your commands and queries never leave your local terminal.
*   **Insanely Fast (<10ms):** Employs an ultra-lightweight 25M parameter model (`logicsim_v2-bash`) optimized in C++ to decode sequences in milliseconds.
*   **Smart Zsh ZLE Widget:** Binds to Enter (`^M`). It checks your input buffer, intercepts `!!` queries, fetches the prediction, and loads it back into your prompt while preserving full line-editing capabilities.
*   **Exotic Recurrent Architecture:** Implements a hybrid transformer model utilizing a recurrent LogicSim branch + Differential Attention for state-of-the-art efficiency on terminal translation tasks.
*   **Exact Output Verification:** Token-for-token parity with PyTorch weight exports.

---

## 🚀 Quick Start

### Prerequisites
*   `git`, `cmake`, a C++17 compiler (`clang` or `gcc`), `curl`, and `zsh`.

### Installation
Clone the repository and run the setup and installation scripts:

```bash
# Clone the repository with its submodules
git clone --recursive https://github.com/thomasreolon/CmdLanguageModel.git
cd CmdLanguageModel

# Patch llama.cpp, build the C++ runner, and download the model
./setup.sh

# Install the !! widget to your ~/.zshrc
./terminal/install.sh

# Reload your shell
exec zsh
```

Now try typing a query in your terminal:
```bash
!! list all files modified in the last 24 hours
```

> [!NOTE]
> `setup.sh` is completely idempotent. You can re-run it at any time. If you forgot the `--recursive` flag during cloning, run `git submodule update --init` before executing `setup.sh`.

---

## ⚙️ How it Works

The system operates across three lightweight components, completely decoupled from any external network:

```text
  ~/.zshrc                  llm/runner/qq-llm            llm/llama.cpp (patched)
  ┌─────────────┐  "!! …"   ┌───────────────────┐  ids  ┌────────────────────┐
  │ qq.zsh ZLE  │ ────────► │ structural tokenizer ───► │ logicsim_v2 arch   │
  │ binds Enter │ ◄──────── │ + greedy decode     ◄──── │ (cache-free decode)│
  └─────────────┘  command  └───────────────────┘ logits└────────────────────┘
```

1.  **`terminal/qq.zsh` (ZLE Widget):** Intercepts Enter. If the line begins with `!!`, it queries `qq-llm "<request>"` and updates the line editor buffer with the command output without executing it. Standard commands flow through untouched.
2.  **`llm/runner/qq-llm` (C++ Backend):** A highly-optimized C++ binary linking `libllama`. It wraps the query in `<bash><in>{request}<out>`, tokenizes it using a custom 202-token structural tokenizer, runs local inference, and greedy-decodes until `<eos>`.
3.  **`llm/llama.cpp` (Model Runtime):** Coordinates model weights and tensor operations. Features a custom architecture added as a lightweight patch over upstream `llama.cpp` to prevent repository bloating.

---

## 🧠 The Neural Model

ShellWhisper uses **`logicsim_v2-bash`** (~25M parameters), trained from scratch in [`hh_llm`](https://github.com/thomasreolon) on natural-language-to-bash translation. 

Rather than using a generic Transformer block, each layer splits its channels between two parallel pathways:
*   **Differential Attention (`y = a1 - λ·a2`):** Cancels high-frequency noise and stabilizes token correlations.
*   **LogicSim Recurrence:** A gated cumulative running-mean + multi-scale EMA recurrence.

Both branches are concatenated via an outer projection followed by RMSNorm, SwiGLU, and RoPE. 

### Recurrent Inference
Because the LogicSim branch is recurrent, the C++ runner decodes **cache-free** (it simply re-evaluates the entire sequence at each step, taking under a few microseconds at this parameters size). All custom operators are compiled down to standard GGML operations inside `llm/llama.cpp` via a localized patch: **no new CUDA/ggml kernels were written**.

The model conversion has been verified token-for-token against the original PyTorch weights. For full conversion details, see [`docs/pytorch-to-gguf.md`](docs/pytorch-to-gguf.md).

---

## 🗺️ Repository Layout

```text
terminal/               Zsh ZLE widget (qq.zsh) and installation scripts
llm/
  logicsim_v2-arch.patch  Custom LogicSim v2 architecture patch for llama.cpp
  llama.cpp/              Pinned upstream llama.cpp submodule
  runner/                 C++ runner (qq-llm) linking libllama (tokenizes + decodes)
  export/                 convert_to_gguf.py & tokenizer_spec.json (checkpoint converter)
  models/                 Vocab files and model destination (downloaded via setup.sh)
docs/                   pytorch-to-gguf.md: Conversion walkthrough & documentation
setup.sh                One-click compiler & asset builder script
```

---

## 🔧 Configuration

The runner behavior can be controlled using the following environment variables:

| Env Var | Default | Purpose |
| :--- | :--- | :--- |
| `QQ_MODEL` | `llm/models/logicsim_v2-bash.gguf` | Path to the model `.gguf` file |
| `QQ_VOCAB` | Path next to the `.gguf` file | Path to the structural tokenizer `vocab.txt` |
| `QQ_LLM` | Next to `qq.zsh` | Path to the built `qq-llm` binary |
| `QQ_RELEASE_URL`| Latest repository Release URL | Remote source directory for `setup.sh` fetches |

---

## 🛠️ Maintainer Notes

Model weights (`.gguf`) are gitignored and fetched on-demand during setup from GitHub Releases. The small `<model>.vocab.txt` file is committed to track token dictionaries.

To publish new weights:
```bash
# Export the weights from an hh_llm checkpoint (see docs/pytorch-to-gguf.md)
# Then create a new GitHub Release:
gh release create v2 llm/models/logicsim_v2-bash.gguf --title "logicsim_v2-bash" \
  --notes "25M logicsim_v2 NL→bash model (GGUF). Run with setup.sh."
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). It links [llama.cpp](https://github.com/ggml-org/llama.cpp) which is also under the MIT License.
