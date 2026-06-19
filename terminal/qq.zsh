# qq.zsh — the "!!" tweak for qq_terminal.
#
# Type a line beginning with "!!" and press Enter: instead of executing, the rest
# of the line is sent to your local model (qq-llm), and the buffer is REPLACED with
# the model's suggested command — left editable so you can tweak it and press Enter
# to run (or Ctrl-C to discard). Any other line runs normally.
#
# Install: source this file from ~/.zshrc  (see terminal/install.sh).
#
# Config (override before sourcing, or in ~/.zshrc):
#   QQ_LLM   path to the qq-llm backend (default: resolved next to this file)
: ${QQ_LLM:="${0:A:h}/../llm/runner/build/qq-llm"}

qq-accept-line() {
  if [[ $BUFFER == '!!'* ]]; then
    local request=${BUFFER#'!!'}
    # trim a single leading space so "!! foo" and "!!foo" behave the same
    request=${request# }
    if [[ -z $request ]]; then
      zle accept-line          # bare "!!" — nothing to ask; act like a normal line
      return
    fi
    # Give feedback while the model runs (decode is fast, but be honest about it).
    zle -M "qq: thinking…"
    local suggestion
    suggestion=$("$QQ_LLM" "$request" 2>/dev/null)
    local rc=$?
    zle -M ""                  # clear the message
    if (( rc == 0 )) && [[ -n $suggestion ]]; then
      BUFFER=$suggestion       # replace the line; NOT executed
      CURSOR=$#BUFFER          # cursor to end, fully editable
      zle redisplay
    else
      zle -M "qq: backend unavailable (QQ_LLM=$QQ_LLM) — buffer left as-is"
    fi
  else
    zle accept-line            # normal command: run it
  fi
}

zle -N qq-accept-line
bindkey '^M' qq-accept-line    # Enter
bindkey '^J' qq-accept-line    # keypad/newline Enter
