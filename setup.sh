#!/usr/bin/env bash
# One-command setup for ShellWhisper: patch llama.cpp, build the runner, fetch models.
#
#   git clone --recursive <url> && cd CmdLanguageModel && ./setup.sh
#
# Prereqs: git, cmake, a C++17 compiler, curl. (Re-converting models also needs python
# with the `gguf` package, but a normal install just downloads them.)
set -euo pipefail
cd "$(dirname "$0")"

# Base URL of the GitHub Release that holds the .gguf model file. Override via env, e.g.
#   QQ_RELEASE_URL=https://github.com/me/CmdLanguageModel/releases/download/v2 ./setup.sh
: "${QQ_RELEASE_URL:=https://github.com/thomasreolon/CmdLanguageModel/releases/latest/download}"

echo "==> 1/4  llama.cpp submodule"
[ -f llm/llama.cpp/CMakeLists.txt ] || git submodule update --init llm/llama.cpp

echo "==> 2/4  logicsim_v2 arch patch"
if grep -q LLM_ARCH_LOGICSIM_V2 llm/llama.cpp/src/llama-arch.h 2>/dev/null; then
    echo "    already applied."
else
    git -C llm/llama.cpp apply "$PWD/llm/logicsim_v2-arch.patch"
    echo "    patched."
fi

echo "==> 3/4  build qq-llm (also builds llama.cpp — a few minutes the first time)"
command -v cmake >/dev/null || { echo "error: cmake not found — install cmake + a C++ compiler"; exit 1; }
cmake -S llm/runner -B llm/runner/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
cmake --build llm/runner/build -j "$(nproc 2>/dev/null || echo 4)"

echo "==> 4/4  fetch models (one .gguf per committed *.vocab.txt)"
for v in llm/models/*.vocab.txt; do
    name=$(basename "$v" .vocab.txt)
    out="llm/models/$name.gguf"
    if [ -f "$out" ]; then echo "    have $name.gguf"; continue; fi
    echo "    downloading $name.gguf ..."
    curl -fL -o "$out" "$QQ_RELEASE_URL/$name.gguf" \
        || { echo "    !! download failed. Set QQ_RELEASE_URL, or run llm/export/convert_to_gguf.py."; rm -f "$out"; }
done

echo
echo "Built: llm/runner/build/qq-llm"
echo "Next:  ./terminal/install.sh    # adds the !! widget to ~/.zshrc"
echo "Then in a new zsh:  !! list png files modified today"
