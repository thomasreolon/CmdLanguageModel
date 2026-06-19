// qq-llm — the local-model backend for qq_terminal.
//
// Contract (the zsh widget depends only on this):
//   argv[1..]  the natural-language request
//   stdout     ONE line: the suggested shell command
//   stderr     diagnostics only
//   exit 0 on success; non-zero => widget keeps the user's text unchanged
//
// We link libllama and feed PRE-TOKENIZED ids, so llama.cpp's tokenizer is never used.
// Our structural tokenizer (vocab.txt, generated from hh_llm) owns encode/decode.
// Framing: <bash><in>{request}<out>  then greedy-decode until <eos>.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"

#ifndef QQ_REPO_DIR
#define QQ_REPO_DIR "."
#endif

// ── structural tokenizer: mirror of hh_llm StructuralTokenizer ───────────────────────
// vocab.txt has one token per line; line number == token id. The vocab itself drives
// encoding (no hardcoded rules), so it works for both the char-level v1 tokenizer and
// the word/symbol-level v2 one. encode() scans left->right:
//   1. '<'..'>' that is a known token -> one structural token
//   2. greedy longest-match over the multi-char tokens (keywords, "&&", "->", ...)
//   3. else one character (unknown chars are skipped, matching the reference)
struct Tokenizer {
    std::vector<std::string> itos;
    std::unordered_map<std::string, int> stoi;
    std::vector<std::string> multi;   // non-bracket tokens with len>1, longest-first
    int eos = -1;

    bool load(const std::string & path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        while (std::getline(f, line)) { stoi[line] = (int)itos.size(); itos.push_back(line); }
        auto it = stoi.find("<eos>");
        if (it != stoi.end()) eos = it->second;
        for (const auto & t : itos)
            if (t.size() > 1 && !(t.front() == '<' && t.back() == '>')) multi.push_back(t);
        std::sort(multi.begin(), multi.end(),
                  [](const std::string & a, const std::string & b) { return a.size() > b.size(); });
        return !itos.empty() && eos >= 0;
    }
    std::vector<llama_token> encode(const std::string & s) const {
        std::vector<llama_token> ids;
        for (size_t i = 0; i < s.size();) {
            if (s[i] == '<') {                              // structural token <...>
                size_t j = s.find('>', i);
                if (j != std::string::npos) {
                    auto it = stoi.find(s.substr(i, j - i + 1));
                    if (it != stoi.end()) { ids.push_back(it->second); i = j + 1; continue; }
                }
            }
            bool matched = false;                           // greedy longest multi-char token
            for (const auto & m : multi)
                if (s.compare(i, m.size(), m) == 0) { ids.push_back(stoi.at(m)); i += m.size(); matched = true; break; }
            if (!matched) {
                auto it = stoi.find(std::string(1, s[i]));   // single char (skip if unknown)
                if (it != stoi.end()) ids.push_back(it->second);
                i += 1;
            }
        }
        return ids;
    }
    std::string decode(const std::vector<llama_token> & ids) const {
        std::string out;
        for (auto id : ids) out += itos[id];
        return out;
    }
};

static void quiet_log(ggml_log_level, const char *, void *) {}  // keep stdout clean

static std::string env_or(const char * key, const std::string & def) {
    const char * v = std::getenv(key);
    return v ? std::string(v) : def;
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: qq-llm <request>\n"); return 2; }
    std::string request;
    for (int i = 1; i < argc; ++i) { if (i > 1) request += ' '; request += argv[i]; }

    const std::string model_path = env_or("QQ_MODEL", QQ_REPO_DIR "/llm/models/bash-v2.gguf");
    // each model carries its own vocab next to it: foo.gguf -> foo.vocab.txt
    std::string default_vocab = model_path;
    if (default_vocab.size() > 5 && default_vocab.substr(default_vocab.size() - 5) == ".gguf")
        default_vocab = default_vocab.substr(0, default_vocab.size() - 5) + ".vocab.txt";
    const std::string vocab_path = env_or("QQ_VOCAB", default_vocab);
    const int max_new = 128;

    Tokenizer tok;
    if (!tok.load(vocab_path)) { std::fprintf(stderr, "qq-llm: cannot load vocab %s\n", vocab_path.c_str()); return 1; }

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                                    // CPU; the model is tiny
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "qq-llm: failed to load %s\n", model_path.c_str()); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 256;                       // == training block_size
    llama_context * ctx = llama_init_from_model(model, cp);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // prompt: <bash><in>{request}<out>  then greedy-decode the command until <eos>.
    // The logicsim mixer is recurrent (cumsum / EMA over the whole sequence), so we run
    // CACHE-FREE: clear the KV memory and re-decode the full sequence each step. Tiny
    // model + <=256 tokens => microseconds, and it is correct for plain attention too.
    std::vector<llama_token> seq = tok.encode("<bash><in>" + request + "<out>");
    std::vector<llama_token> gen;
    llama_memory_t mem = llama_get_memory(ctx);
    int rc = 0;

    for (int step = 0; step < max_new && (int)seq.size() < (int)cp.n_ctx; ++step) {
        llama_memory_clear(mem, true);
        llama_batch batch = llama_batch_get_one(seq.data(), (int)seq.size());
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "qq-llm: decode failed\n"); rc = 1; break; }
        const float * logits = llama_get_logits_ith(ctx, -1);   // last position
        llama_token best = 0;                                   // greedy argmax
        for (int v = 1; v < n_vocab; ++v) if (logits[v] > logits[best]) best = v;
        if (best == tok.eos) break;
        gen.push_back(best);
        seq.push_back(best);
    }

    if (rc == 0) { std::string cmd = tok.decode(gen); std::printf("%s\n", cmd.c_str()); }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
