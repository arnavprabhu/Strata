# RTX 4070 Laptop GPU 8 GB: Qwen3.6-35B-A3B UD-Q4_K_XL at 204,800 context, `--kv int8` (2026-10-18)

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the
engine's own timings; first-token times are measured by the client over HTTP. The suite measures speed only; the
long prompts and needles below are separate runs, 1 run each. The engine is the fork branch `qwen36-8gb` (upstream
477ac13c + open PRs #2917, #2698, #2742 + opt-in work): what it changes and the step-by-step screens are in
[docs/QWEN36_8GB.md](../../../docs/QWEN36_8GB.md).

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GiB VRAM (8,188 MiB), sm_89, PCIe 4.0 x8
- CPU: 13th Gen Intel Core i9-13900H (14 cores, 20 threads); installed RAM: 30.2 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.99.02 (`nvidia-smi` reports CUDA 13.2); engine built with CUDA 13.1
  for sm_89
- Strata: engine built from branch `qwen36-8gb` at 9de30633 (version 0.1.45 in the source). The benchmark client
  ran from a 477ac13c checkout with the `strata_bench.py` fix of 71f9fa9b, which is why `results.json` says
  `"strata_commit": "477ac13c"` and `"engine_version": null`.
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: Qwen3.6-35B-A3B UD-Q4_K_XL (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, the revision setup pins),
  with its MTP layer
- Context: 204,800 tokens; parallel slots: 1
- Hybrid mode: 4.40 GiB of weights on the GPU; 0 of 256 experts per layer in VRAM; all 18.75 GiB of experts in pinned
  host memory, computed by the CPU pool (11 threads)
- Settings: thinking off for the suite; temperature 0; the model card's sampling (the server fills in
  `presence_penalty` 1.5)

```text
strata --native Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 204800 --prefill 2048 --mtp 2 --pool-workers 10 --kv int8
```

Environment of the config: `STRATA_Q36_TURN_CKPT=1`, `STRATA_Q36_PENALTY_SPEC=1`, `STRATA_Q36_ATTN_GQA=1` (all three are
off by default; see the doc).

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the
config); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a
unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and
  continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per
  turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 45.2 (41.8-52.8) | 0.58 (0.55-0.64) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 953.9 (951.4-960.4) | not measured | 4.22 (4.20-4.23) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 967.4 (962.7-970.7) | not measured | 16.62 (16.56-16.71) |
| agent session, follow-up turns | 14532 (14523-14550) at the last turn | see below | up to 64 | 3 | see below | 45.7 (39.8-49.2) | 4.13 (2.12-4.70) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.2 % (68.1-68.2); earlier-turn tokens read again per session: 261 (252-279)
(earlier turns that the cache did not hold; the 13697 new tokens of the files are read once); first turn (nothing
cached) first token: 2.05 s (2.05-2.12).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 297.1 | 2.12 | 40.7 |
| 1 | 3767 | 602 | 3165 | 767.2 | 4.704 | 48.5 |
| 2 | 5286 | 3760 | 1526 | 724.4 | 2.135 | 49.2 |
| 3 | 8705 | 5279 | 3426 | 827.9 | 4.172 | 42.6 |
| 4 | 11608 | 8698 | 2910 | 709.2 | 4.137 | 46.5 |
| 5 | 14550 | 11601 | 2949 | 692.9 | 4.283 | 45.5 |

Per-request data: `results.json` (total time of the suite: 181 s).

### Long prompts and needles (same server config, 1 run each)

From the engine's log (read time, prompt and decode speed) and the client (first token). Code-file prompts:
`bench/results/2026-09-29-rtx3090-epyc-milan/data/strata-bench.py --contexts 32k,128k --repeats 1 --max-tokens 256`.
Needles: `tools/needle_bench.py --lengths 64k,190k --depths 10,50,90`.

| Prompt | Prompt tokens | Read in | Prompt tok/s | Generated | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| code files, 32K | 29,509 | 31.9 s (first token 32.7 s) | 923.9 | 256 | 41.9 |
| needle 64K, depths 10 / 50 / 90 | 66,168-66,169 | 84.1 s | 787.2 | 9-10 | - |
| code files, 128K | 147,910 | 263.7 s (first token 264.0 s) | 561.0 | 256 | 37.8 |
| needle 190K, depths 10 / 50 / 90 | 190,462-190,464 | 389.0 s | 489.7 | 9-10 | - |

Needles found: 6 of 6.

### Where the VRAM goes (engine log at startup)

The sessions' K / V and state take 2.27 GiB at 204,800 cells. The packed Q8_0 decode copies (1.78 GiB) are given
back before the sessions. The prompt attention runs in blocks of 16,384 cells (FP16 K / V 0.031 GiB instead of
0.391 GiB), with split-K partials of 0.098 GiB. The prompt staging is one buffer (0.55 GiB) instead of a pair
(1.10 GiB), and the attention scores buffer is 64 MB instead of 256. That leaves 0.92 GiB free for the hybrid
buffers.

## Correctness and limitations

The suite does not check answers; the needles are the only answer check at long context. The conversation uses the
model's own replies, so token counts differ a little between runs. One machine, CUDA sm_89 only (no HIP or SYCL
build), one slot, Linux. The three pull requests are not merged upstream. Everything that was not tested is listed in
[docs/QWEN36_8GB.md](../../../docs/QWEN36_8GB.md#not-tested).
