#!/usr/bin/env bash
# PreToolUse hook: block Read/Edit/Write/MultiEdit/NotebookEdit on .env files,
# and Bash commands that reference them.
input=$(cat)
tool=$(jq -r '.tool_name' <<<"$input")

deny() {
  echo "Blocked: access to .env files is not allowed." >&2
  exit 2
}

if [[ "$tool" == "Bash" ]]; then
  cmd=$(jq -r '.tool_input.command // ""' <<<"$input")
  # match .env, .env.local, etc. but not .envrc-like names or e.g. "foo.environment"
  grep -Eq '(^|[^[:alnum:]_])\.env(\.[[:alnum:]_.-]+)?([^[:alnum:]_.-]|$)' <<<"$cmd" && deny
else
  path=$(jq -r '.tool_input.file_path // .tool_input.notebook_path // ""' <<<"$input")
  base=$(basename -- "$path")
  [[ "$base" == ".env" || "$base" == .env.* ]] && deny
fi
exit 0
