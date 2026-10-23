"""tools/qwen35/kv_kl.py - two --dump-full-logits files of strata-q36 (float32, one vocabulary row per step) against each
other: mean KL(A || B), the top-1 agreement and the largest logit gap, for comparing two --kv stores on one teacher-forced
sequence (the same --tokens and --force in both runs):

    strata-q36 --model M.gguf --max-context 32768 --kv int8 --tokens PROMPT_IDS --force NEXT_IDS --dump-full-logits a.bin
    strata-q36 --model M.gguf --max-context 32768 --kv k8v4 --tokens PROMPT_IDS --force NEXT_IDS --dump-full-logits b.bin
    python tools/qwen35/kv_kl.py a.bin b.bin [--vocab 248320]

The prompt is read in chunks (the prompt attention fills the cache) and the forced ids one by one (the decode attention
reads it): use a prompt of real text of the length of interest.  The rows are the steps after the prompt.
"""
from __future__ import annotations

import argparse

import numpy as np


def rows(path: str, v: int) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float32)
    if a.size % v:
        raise SystemExit(f"{path}: {a.size} floats is not a multiple of the vocabulary {v}")
    return a.reshape(-1, v).astype(np.float64)


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--vocab", type=int, default=248320)
    o = ap.parse_args()
    a, b = rows(o.a, o.vocab), rows(o.b, o.vocab)
    n = min(len(a), len(b))
    if len(a) != len(b):
        print(f"note: {len(a)} vs {len(b)} rows, comparing the first {n}")
    la, lb = log_softmax(a[:n]), log_softmax(b[:n])
    kl = (np.exp(la) * (la - lb)).sum(axis=1)
    top = (a[:n].argmax(1) == b[:n].argmax(1))
    print(f"rows {n}: mean KL(A||B) {kl.mean():.5f}  median {np.median(kl):.5f}  max {kl.max():.5f}  "
          f"top-1 agreement {top.mean() * 100:.1f}%  max |logit gap| {np.abs(a[:n] - b[:n]).max():.3f}")


if __name__ == "__main__":
    main()
