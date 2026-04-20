#!/usr/bin/env python3
"""
Prompt Compression Agent
Compresses prompts and conversation histories to reduce Claude API token usage
while preserving 100% of the semantic meaning and intent.

Usage:
    python agent.py                          # interactive (paste, then Ctrl+D)
    echo "your long prompt" | python agent.py
    python agent.py --file prompt.txt
    python agent.py --quiet                  # compressed output only (pipe-friendly)
"""

import anthropic
import sys
import argparse

# NOTE: swap to "claude-haiku-4-5" in production — this utility is called
# frequently and Haiku handles summarisation well at a fraction of the cost.
MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are a prompt compression engine. Your sole purpose is to reduce the \
user's text to the fewest tokens possible while preserving 100% of its semantic meaning, \
technical constraints, and intent.

Compression rules:
- Remove filler words, pleasantries, and redundant phrasing
- Condense verbose explanations into single precise statements
- Preserve all technical specifics: code, variable names, numbers, examples, constraints
- Strip meta-commentary ("I'd like you to…", "Could you please…", "As an AI…")
- Merge related points where unambiguous
- Use common abbreviations where clear (e.g. w/ for "with", b/c for "because")

Output ONLY the compressed text — no preamble, no explanation, no framing."""


def count_tokens(client: anthropic.Anthropic, text: str) -> int:
    """Return the token count of a user message."""
    result = client.messages.count_tokens(
        model=MODEL,
        messages=[{"role": "user", "content": text}],
    )
    return result.input_tokens


def compress(client: anthropic.Anthropic, text: str, quiet: bool = False) -> dict:
    """
    Compress a prompt or conversation history.

    In default mode the compressed text streams to stdout in real time.
    Stats are always written to stderr so piping stdout stays clean.

    Returns a dict with: compressed, before_tokens, after_tokens, savings_pct.
    """
    before_tokens = count_tokens(client, text)
    compressed_text = ""

    with client.messages.stream(
        model=MODEL,
        max_tokens=8192,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                # Cache the system prompt — it never changes between calls
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": text}],
    ) as stream:
        for chunk in stream.text_stream:
            compressed_text += chunk
            if not quiet:
                # Stream to stdout so the user sees progress immediately
                print(chunk, end="", flush=True)

    after_tokens = count_tokens(client, compressed_text)
    savings_pct = (
        round((1 - after_tokens / before_tokens) * 100, 1) if before_tokens else 0.0
    )

    return {
        "compressed": compressed_text,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "savings_pct": savings_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress prompts to reduce Claude API token usage."
    )
    parser.add_argument("--file", "-f", metavar="PATH", help="Read input from a file")
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Print only the compressed text (no stats, pipe-friendly)",
    )
    args = parser.parse_args()

    # ── Gather input ─────────────────────────────────────────────────────────
    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Paste your prompt then press Ctrl+D:\n", file=sys.stderr)
        try:
            text = sys.stdin.read()
        except KeyboardInterrupt:
            sys.exit(0)

    text = text.strip()
    if not text:
        print("Error: no input provided.", file=sys.stderr)
        sys.exit(1)

    # ── Compress ──────────────────────────────────────────────────────────────
    client = anthropic.Anthropic()

    if not args.quiet:
        print("Compressing…\n", file=sys.stderr)

    result = compress(client, text, quiet=args.quiet)

    # ── Output ────────────────────────────────────────────────────────────────
    if args.quiet:
        # Clean stdout for piping
        print(result["compressed"])
    else:
        # Compressed text already streamed above; print stats to stderr
        print(f"\n\n{'─' * 44}", file=sys.stderr)
        print(f"Before : {result['before_tokens']:>6,} tokens", file=sys.stderr)
        print(f"After  : {result['after_tokens']:>6,} tokens", file=sys.stderr)
        print(f"Savings: {result['savings_pct']:>5}%", file=sys.stderr)


if __name__ == "__main__":
    main()
