#!/bin/bash
# Aider CLI integration harness against a running `rapid-mlx serve`.
#
# What it proves:
#   1. Aider's REPL can connect to rapid-mlx's OpenAI-compatible endpoint.
#   2. Aider's SEARCH/REPLACE edit-and-write pipeline actually rewrites a
#      local file the way we asked ("fix the bug in add.py — add, not
#      subtract").
#
# Why the correctness signal is NOT OpenAI tool_calls: Aider does not
# use function-calling. It sends the file + user instruction as plain
# messages, expects the LLM to emit ``SEARCH ... REPLACE ...`` blocks,
# and applies those edits locally. So the pass gate is whether
# ``add.py`` really contains ``return a + b`` after aider exits.
#
# Usage:
#   test_aider.sh --model <alias> (--base-url <url> | --port <port>) [--timeout <secs>]
#
# ``--base-url`` takes the full ``http[s]://host:port/v1`` URL and is the
# preferred form — it lets the Python wrapper pass whatever URL the
# ``rapid_mlx_server`` fixture is actually pointed at (which may be
# non-localhost in CI shards or a remote-serve run). ``--port`` is kept
# for standalone local invocations and defaults host to ``127.0.0.1``.
#
# Env vars (set automatically, but overridable):
#   HOME              — overridden to a scratch dir so aider's config /
#                       cache / analytics files don't touch the operator's
#                       real ``~/.aider*`` state
#   AIDER_BIN         — full path to the aider binary; skipped-search if set
#   AIDER_ANALYTICS_ASKED=1
#   AIDER_CHECK_UPDATE=false
#
# Exit codes:
#   0  — aider completed and add.py now contains ``return a + b``
#   1  — arg parse / setup error
#   2  — aider CLI exited non-zero
#   3  — aider ran but the file wasn't corrected (edit format didn't
#        apply; SEARCH/REPLACE parse failed; LLM refused; etc.)
#   4  — timeout

set -u
set -o pipefail

TIMEOUT=300
MODEL=""
PORT=""
BASE_URL=""
VERBOSE=0

usage() {
    echo "Usage: $0 --model <alias> (--base-url <url> | --port <port>) [--timeout <secs>] [-v]" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        -v|--verbose) VERBOSE=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [ -z "$MODEL" ] || { [ -z "$BASE_URL" ] && [ -z "$PORT" ]; }; then
    usage
fi

# Derive BASE_URL from --port only if --base-url wasn't given (back-compat
# for the standalone invocation shape kept for local docs/dev). --base-url
# wins so a Python wrapper that always passes the full URL is authoritative.
if [ -z "$BASE_URL" ]; then
    BASE_URL="http://127.0.0.1:${PORT}/v1"
fi

# Locate aider — never PATH-search on the operator's box because the
# harness must NOT accidentally trigger a fresh install. The 2026-07-06
# Tier-1 install (v0.86.2) sits at ~/.local/bin/aider.
AIDER_BIN="${AIDER_BIN:-/Users/raullenstudio/.local/bin/aider}"
if [ ! -x "$AIDER_BIN" ]; then
    # Fall back to PATH so the harness still runs on CI where the pinned
    # path doesn't exist; but never install.
    AIDER_BIN="$(command -v aider 2>/dev/null || true)"
    if [ -z "$AIDER_BIN" ]; then
        echo "ERROR: aider binary not found (checked /Users/raullenstudio/.local/bin/aider and PATH)" >&2
        exit 1
    fi
fi

# Scratch state — HOME override so aider drops its config / cache into a
# throw-away tree we can nuke on exit. WORKDIR is a fresh scratch repo.
SCRATCH_HOME="${SCRATCH_HOME:-/tmp/aider-test-home-$$}"
WORKDIR="$(mktemp -d)"
mkdir -p "$SCRATCH_HOME"

cleanup() {
    local rc=$?
    if [ "$VERBOSE" -eq 0 ]; then
        rm -rf "$WORKDIR" "$SCRATCH_HOME" 2>/dev/null || true
    else
        echo "VERBOSE: preserved WORKDIR=$WORKDIR SCRATCH_HOME=$SCRATCH_HOME" >&2
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

# Toy file with an obvious bug: subtraction masquerading as addition.
# Pass gate = LLM must emit an edit block that flips ``- b`` → ``+ b``.
cat > "$WORKDIR/add.py" <<'PYEOF'
def add(a, b):
    return a - b  # BUG
PYEOF

# Sanity: is the server actually up? A quick /v1/models probe with a
# 5 s timeout catches "operator forgot to boot serve" instantly instead
# of eating the 300 s aider timeout.
if ! curl -sS -m 5 "$BASE_URL/models" >/dev/null 2>&1; then
    echo "ERROR: rapid-mlx server not reachable at $BASE_URL" >&2
    exit 1
fi

# Aider needs LiteLLM's ``openai/`` prefix to route through the
# OpenAI-compatible chat completions path — without it LiteLLM tries
# to pick a provider from the alias string and fails on non-canonical
# rapid-mlx aliases.
LITELLM_MODEL="openai/${MODEL}"

echo "[test_aider.sh] model=$MODEL base_url=$BASE_URL timeout=${TIMEOUT}s"
echo "[test_aider.sh] litellm-model=$LITELLM_MODEL"
echo "[test_aider.sh] scratch home=$SCRATCH_HOME workdir=$WORKDIR"
echo "[test_aider.sh] BEFORE add.py:"
cat "$WORKDIR/add.py"
echo "--------"

# Run aider one-shot (``--message`` runs a single round then exits). We
# deliberately pass every "quiet, don't touch the network, don't pollute
# the operator's box" flag we can find:
#   --no-git             — don't require a git repo; don't create commits
#   --no-analytics       — skip PostHog analytics
#   --no-check-update    — skip pip-index poll
#   --no-show-model-warnings — model isn't in aider's known list; silence
#   --no-pretty          — plain text, no ANSI (easier to grep on failure)
#   --no-stream          — Rapid-MLX supports streaming, but non-stream is
#                          less flaky on slow local inference
#   --map-tokens 0       — don't burn a turn building a repo map
#   --yes-always         — take all prompts as "yes"
LOG="$WORKDIR/aider.log"
STATUS=0

# ``timeout(1)`` (coreutils) may not be present on macOS; use ``gtimeout``
# if available, else fall back to a background-process kill approach.
if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(gtimeout --preserve-status "$TIMEOUT")
elif command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(timeout --preserve-status "$TIMEOUT")
else
    # PID-kill fallback — spawn aider in the background, watchdog kills it.
    TIMEOUT_CMD=()
fi

(
    cd "$WORKDIR"
    HOME="$SCRATCH_HOME" \
    AIDER_ANALYTICS_ASKED=1 \
    AIDER_CHECK_UPDATE=false \
    OPENAI_API_BASE="$BASE_URL" \
    OPENAI_API_KEY="rapidmlx" \
    "${TIMEOUT_CMD[@]}" \
    "$AIDER_BIN" \
        --model "$LITELLM_MODEL" \
        --openai-api-base "$BASE_URL" \
        --openai-api-key "rapidmlx" \
        --no-git \
        --no-analytics \
        --no-check-update \
        --no-show-model-warnings \
        --no-pretty \
        --no-stream \
        --map-tokens 0 \
        --yes-always \
        --message "Fix the bug in add.py — this function should add, not subtract. Change the '-' operator to '+' in the return statement." \
        add.py \
        >"$LOG" 2>&1
) &
AIDER_PID=$!

# Fallback watchdog only when ``timeout``/``gtimeout`` missing.
if [ ${#TIMEOUT_CMD[@]} -eq 0 ]; then
    ( sleep "$TIMEOUT" && kill -TERM "$AIDER_PID" 2>/dev/null ) &
    WATCHDOG_PID=$!
    wait "$AIDER_PID" || STATUS=$?
    kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
else
    wait "$AIDER_PID" || STATUS=$?
fi

# Detect timeout: 124 = coreutils timeout; 143 = SIGTERM from watchdog.
if [ "$STATUS" -eq 124 ] || [ "$STATUS" -eq 143 ]; then
    echo "[test_aider.sh] TIMEOUT after ${TIMEOUT}s" >&2
    echo "--- last 60 lines of aider log ---" >&2
    tail -60 "$LOG" >&2 || true
    exit 4
fi

echo "[test_aider.sh] aider exit=$STATUS"
echo "--- last 40 lines of aider log ---"
tail -40 "$LOG" || true
echo "--------"
echo "[test_aider.sh] AFTER add.py:"
cat "$WORKDIR/add.py"
echo "--------"

if [ "$STATUS" -ne 0 ]; then
    echo "[test_aider.sh] FAIL: aider exited $STATUS" >&2
    exit 2
fi

# The correctness signal: does add.py now say ``return a + b``?
# We accept either exact ``return a + b`` or the same expression with
# extra whitespace. We reject anything that still contains ``return a - b``.
if grep -qE '^\s*return\s+a\s*\+\s*b' "$WORKDIR/add.py" && \
   ! grep -qE '^\s*return\s+a\s*-\s*b' "$WORKDIR/add.py"; then
    echo "[test_aider.sh] PASS: add.py was corrected to 'return a + b'"
    exit 0
else
    echo "[test_aider.sh] FAIL: add.py was NOT corrected" >&2
    echo "--- final add.py ---" >&2
    cat "$WORKDIR/add.py" >&2
    exit 3
fi
