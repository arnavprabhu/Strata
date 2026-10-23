# Qwen3.6-35B-A3B on an 8 GB card with a 200K context (branch `qwen36-8gb`)

This branch of a fork of [Strata](https://github.com/Niko1221/Strata) runs
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) UD-Q4_K_XL ([QWEN36.md](QWEN36.md)) on an 8 GB laptop
card with a context of **204,800 tokens**, a 2,048-token prompt chunk and the MTP drafts. Upstream main (477ac13c)
starts this model on the same card at 8,192 tokens and no more.

It is upstream main 477ac13c, plus three open upstream pull requests merged here (not yet merged upstream), plus new
opt-in work on the Qwen3.6 engine path (`src/qwen35/q36_engine.cu`):

- [#2917](https://github.com/Niko1221/Strata/pull/2917) by **CC-David-CC**: the hybrid mode gives the optional
  packed Q8_0 copies back so the prompt's staging buffers fit on a small card;
- [#2698](https://github.com/Niko1221/Strata/pull/2698) by **TorBere**: `--kv int8` K / V caches (int8 codes and an
  FP16 scale per 64 values) and the vectorized int8 decode reads;
- [#2742](https://github.com/Niko1221/Strata/pull/2742) by **ggeorgovassilis**: its single staging buffer
  (`STRATA_Q36_SINGLE_STAGE=1`), ported into 20309430. Its own int8 K / V cache was measured (VB below) but not kept;
- new here, all of it only under `--kv int8` or behind an environment variable that is off by default: the VRAM
  savings that fit 204,800 cells at chunk 2048, blocked prompt attention, MTP drafts for greedy requests with a
  presence penalty, a turn checkpoint for agent follow-up turns and a GQA-shared decode attention kernel.

Everything below was measured on one laptop. See [Not tested](#not-tested).

## The machine

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB (8,188 MiB), sm_89, PCIe 4.0 x8; driver 595.99.02.
- CPU: Intel Core i9-13900H (14 cores, 20 threads); RAM: 30.2 GiB; OS: Linux 7.0.0-38.
- Engine: built from this branch with CUDA 13.1 for sm_89 (the source says 0.1.45). `nvidia-smi` reports CUDA 13.2
  (the driver's version).
- Model: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` from `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (the revision setup pins,
  5bc3e238), with its MTP layer. In the hybrid mode: 4.40 GiB of weights on the GPU, 0 of 256 experts per layer in
  VRAM at 204,800 cells, all 18.75 GiB of experts in pinned host memory, computed by the CPU pool (11 threads).
- Dates: 2026-10-17 (the step-by-step screens) and 2026-10-18 (the final runs). Background load and power limits
  were not recorded.

## Results with the final configuration

`tools/strata_bench.py` suite 1, **3 runs** per test, medians with the lowest and highest run, 2026-10-18. Thinking
off, temperature 0, the model card's sampling (the server fills in `presence_penalty` 1.5). Full report:
[bench/results/2026-10-18-community-rtx-4070-laptop-qwen3-6-35b-a3b-int8-200k](../bench/results/2026-10-18-community-rtx-4070-laptop-qwen3-6-35b-a3b-int8-200k/).

| Test | Prompt tokens | Prompt tok/s | Decode tok/s | First token (s) |
| --- | ---: | ---: | ---: | ---: |
| short chat (9 requests) | 43-51 | - | 45.2 (41.8-52.8) | 0.58 (0.55-0.64) |
| 4K prompt, fresh | 4,019 | 953.9 (951.4-960.4) | - | 4.22 (4.20-4.23) |
| 16K prompt, fresh | 16,058 | 967.4 (962.7-970.7) | - | 16.62 (16.56-16.71) |
| agent session, follow-up turns | 14,532 (14,523-14,550) at the last turn | - | 45.7 (39.8-49.2) | 4.13 (2.12-4.70) |

Agent session: 68.2% (68.1-68.2) of the follow-up turns' prompt tokens came from the conversation cache; 261
(252-279) earlier-turn tokens were read again per session (30,177 without the turn checkpoint, see below).

**Long prompts**, same configuration, **1 run each** unless noted (2026-10-18; prompt and decode speeds from the
engine's log, first token from the client):

| Prompt | Prompt tokens | Read in | Prompt tok/s | Generated | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| code files, 32K (`strata-bench.py`, below) | 29,509 | 31.9 s (first token 32.7 s) | 923.9 | 256 | 41.9 |
| needle, 64K (3 depths) | 66,168-66,169 | 84.1 s | 787.2 | 9-10 | (too few tokens) |
| this engine's source file, thinking on (GQA test below; its run on the final build) | 110,094 | 169.3 s | 650.4 | 128 | 42.9 |
| code files, 128K (`strata-bench.py`, below) | 147,910 | 263.7 s (first token 264.0 s) | 561.0 | 256 | 37.8 |
| needle, 190K (3 depths) | 190,462-190,464 | 389.0 s | 489.7 | 9-10 | (too few tokens) |

The code-file prompts come from `bench/results/2026-09-29-rtx3090-epyc-milan/data/strata-bench.py --contexts
32k,128k --repeats 1 --max-tokens 256`.

**Needles** (`tools/needle_bench.py --lengths 64k,190k --depths 10,50,90`: one code word such as `meadow-copper-504`
hidden in filler text and asked for at the end): **6 of 6 found**, at 64K and 190K tokens, depths 10%, 50% and 90%. This is the only check of the answers at long context;
there is no KL or perplexity measurement.

## How to run it

**Build** the engine from this branch like any CUDA build. `./setup.sh --build` compiles it with setup's flags, or by
hand (the flags setup uses for CUDA; `89` is the RTX 40 series, use your card's):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSTRATA_GGML_DIR=<strata>/third_party/llama.cpp
cmake --build build --target strata
```

With `-DSTRATA_MMQ_KQUANTS=ON` on CUDA the `strata` target also builds `build/libstrata_kq.so` (the K-quant prompt
kernels) as a dependency; keep it next to the `strata` binary, where the engine looks for it. Get the model and its tokenizer
with setup (`--family qwen36 --model UD-Q4_K_XL`, [QWEN36.md](QWEN36.md)).

**The config** used for every final number (`serve/server.py --engine strata --config <this file>`; paths are
placeholders):

```json
{
  "exe": "<strata>/build/strata",
  "args": ["--native", "<models>/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
           "--max-context", "204800", "--prefill", "2048", "--mtp", "2", "--pool-workers", "10", "--kv", "int8"],
  "tokenizer": "<data>/packs/qwen36-ud-q4_k_xl/tokenizer",
  "model_name": "qwen3.6-35b-a3b-ud-q4_k_xl",
  "arch": "qwen35moe",
  "lib_dirs": ["/usr/local/cuda-13.1/bin", "/usr/local/cuda-13.1/lib64"],
  "host": "127.0.0.1",
  "port": 8080,
  "env": {"STRATA_Q36_TURN_CKPT": "1", "STRATA_Q36_PENALTY_SPEC": "1", "STRATA_Q36_ATTN_GQA": "1"}
}
```

- `--kv int8`: int8 K / V caches (#2698). This flag also turns on this branch's VRAM savings (below).
- `--prefill 2048`: the prompt chunk. At 204,800 cells the automatic choice was 512 on the VC build (the final build
  was not started with `--prefill auto` at 204,800); 2048 reads prompts 3.7 times as fast on this card (see VC and VD
  below).
- `--pool-workers 10`: the CPU pool's threads (the engine logs 11 with the main thread).
- `--mtp 2`: the MTP drafts, as setup writes it.

Keep `host` at `127.0.0.1`, or set `--api-key` before you open the server to other machines.

At startup the engine logs where the VRAM goes. At 204,800 cells, chunk 2048:

```text
strata-q36: the sessions' K / V and state (2.27 GiB at a context of 204800) do not fit in the 0.91 GiB free: the packed Q8_0 decode copies (1.78 GiB) given back before them, 2.84 GiB free now
strata-q36: context 204800 cells, prompt chunk 2048, MTP block loaded, 0.54 GiB of VRAM free
strata-q36: the prompt attention over the context in blocks of 16384 cells: its FP16 K / V 0.031 GiB instead of 0.391 GiB; split-K partials for 8 rows, 0.098 GiB
strata-q36: the prompt's staging pair (1.10 GiB) does not fit in the 0.64 GiB free: one staging buffer (0.55 GiB), as STRATA_Q36_SINGLE_STAGE=1
strata-q36: the prompt staging (0.55 GiB) did not fit: the attention scores buffer at 64 MB instead of 256, 0.92 GiB free now
strata-q36: hybrid experts: 0 of 256 per layer in VRAM (0.00 GiB), all 18.75 GiB in pinned host memory
```

## The switches and their defaults

| Switch | Default | What it does |
| --- | --- | --- |
| `--kv int8` (also `k8v8`, `k8v32`, `k32v8`; #2698) | `fp32` | int8 K / V stores. With an int8 store this branch also: gives the packed Q8_0 decode copies back **before** the sessions when the sessions would not fit (3ee94325; with FP32 stores the old order and its failure are kept), sizes the CPU pool's outputs and the grouped experts' scratch for the hybrid batch instead of the chunk (120 + 98 MiB less at chunk 2048), puts the MMQ gate+up and SwiGLU buffers in the experts' output rows (96 MiB less), sizes the split-K partials by their bound (32.1 -> 8.0 MiB) (these MiB from the code's arithmetic, not measured; logged at chunk 2048, VC -> VD: sessions 2.34 -> 2.27 GiB, 0.34 -> 0.54 GiB free after the MTP block), and uses one staging buffer and a smaller attention scores buffer when the pair does not fit. |
| `STRATA_Q36_KV16_BLOCK=cells` | 16384 under an int8 store at a context above 16,384 cells; 0 (off) otherwise | Prompt attention over the context in blocks of this many cells, with an online softmax, so its FP16 copy of K / V is one block (0.031 GiB) instead of the whole context (0.391 GiB at 204,800). `0` turns it off. |
| `STRATA_Q36_SINGLE_STAGE=1` (#2742) | off; automatic under an int8 store when the staging pair does not fit | One prompt staging buffer instead of two: a layer's copy waits for the previous layer to finish. |
| `STRATA_Q36_KEEP_PACKED=1` | unset | Keeps the packed Q8_0 decode copies (existing switch). |
| `STRATA_Q36_PENALTY_SPEC=1` | off | A temperature-0 request that has a penalty (the model card fills in `presence_penalty` 1.5) took no MTP drafts (`drafts accepted 0 of 0`). With this set it takes them: each verify row is picked on the host from that row's penalized logits, and the window stops at the first draft that does not match. |
| `STRATA_Q36_SPEC_SAMPLE=1` | off | With `--mtp N`, a request with temperature above 0 (Claude Code, chat pages: 0.6 - 0.7 with top_p / top_k) took no MTP drafts. With this set it takes them as exact speculative sampling: draft i is accepted with the probability the sampler gives that token at its verify row (same penalties, top_k, min_p and top_p as one token at a time, the earlier accepted drafts counted in the penalty history), the first rejection draws its replacement from that row's distribution with the draft taken out, and a window that accepts every draft draws the bonus token from the last row. The text has the same distribution as without drafts; for a fixed seed it is a different sample (the draws are used differently). Not measured on a GPU yet; CPU test of the accept / reject rule: `tools/q36_spec_sample_test.cpp`. |
| `STRATA_Q36_TURN_CKPT=1` | off | The Qwen3.6 chat template renders an earlier answer differently once the next turn exists (it drops the empty `<think></think>`), so the held state was never the start of the next prompt and every agent turn read the whole chat again. With this set, a prompt is read up to its last `<|im_start|>`, the recurrent state there (GDN conv and state, about 63 MiB by the code, not logged) is copied to pinned host memory, then the rest is read; the next turn resumes from that copy. |
| `STRATA_Q36_PARK_MIB=N` | 0 (off) | The engine holds one conversation, so a short side request between two turns of an agent's chat (Claude Code's title and topic calls, about 340 tokens) replaced it and the next turn read the whole chat again. With a budget the conversation a request is about to overwrite is parked in host RAM (its K / V cells, its recurrent state at the end and at its turn checkpoint; 338 MiB at 19,135 tokens, logged), and a prompt that starts with a parked conversation's tokens brings it back (31-37 ms at 19K, logged) instead of reading them. A prompt read from the start is also cut at the end of its first message (the system prompt and tools) when that is `STRATA_Q36_PARK_ROOT` tokens or more (default 2048), and the state there is parked as a shared root, so a new chat with the same system prompt (a subagent, a new session) reads only what follows it. `STRATA_Q36_PARK_SLOTS` (default 4) entries at most, oldest use leaving first and roots last; `STRATA_Q36_PARK_MIN` (default 1024) is the shortest conversation parked; `STRATA_Q36_PARK_FREE_MIB` (default 1024) is the RAM (MemAvailable) a park must leave. A batch slot's request (`BGEN`) neither parks nor restores; nor does a prompt with pictures. |
| `STRATA_Q36_ATTN_GQA=1` | off | Decode attention with one block per (token, KV head, chunk of cells) for the 8 query heads that share a KV head: each K / V cell is read once per KV head instead of once per query head. int8 stores on sm_80 and newer: Q.K and P.V on FP16 tensor cores (FP32 sums); FP32 stores, and int8 before sm_80: CUDA cores. Unset: the old kernels and launches. |
| `STRATA_Q36_ATTN_GQA_CHUNK=cells` | 1024 | The chunk of cells per block for the GQA kernel (a multiple of its 256-cell tile). |

**The default path.** Without `--kv` and without these variables the engine allocates and computes as before. Checked
on this laptop: 3 prompts x 96 greedy tokens (thinking off, `presence_penalty` 0), 32,768 context, chunk auto, with
the build of main + #2917 and with this branch's final build: **the same text, byte for byte**. The base of this
check is main + #2917, not main alone, because main does not start at 32,768 on this card. #2917 itself changes the
default path (it is an upstream pull request, not part of this work).

## Step by step

Each step was screened with `tools/strata_bench.py` suite 1, **1 run per test**, on 2026-10-17, same model and
machine. The model card's `presence_penalty` 1.5 was in every request unless noted, so `--mtp 2` took no drafts
before `STRATA_Q36_PENALTY_SPEC`.

**Which contexts start** (`--prefill auto` unless a chunk is given):

Blank: not tried.

| Build | 8K | 16K | 32K | 64K | 128K | 200K (204,800) |
| --- | --- | --- | --- | --- | --- | --- |
| V0: main 477ac13c | starts | fails (hybrid buffers: 1,159 MiB needed, 812 free) | fails (12 MiB free) | fails (out of memory) | fails | - |
| V1: + #2917 | | | starts (1 of 256 experts per layer in VRAM) | fails (out of memory) | | |
| VA: V1 + #2698, `--kv int8` | | | starts (14 per layer) | starts (7 per layer) | fails | fails |
| VB: V1 + #2742, `--kv int8` | | | starts | starts (7 per layer) | fails | fails (also with `STRATA_Q36_SINGLE_STAGE=1`; 256K fails too) |
| VC: VB + blocked prompt attention + early give-back | | | | starts | starts (5 per layer) | starts at chunk 512 and 1024; **fails at chunk 2048** (692 MiB needed, 636 free) |
| VD: VA's int8 + VC's changes + the new savings (20309430, 3ee94325) | | | | | | **starts at chunk 2048** (0 per layer, 0.92 GiB free after staging) |

**Speed at each step** (1 run; "agent" = the follow-up turns of the 5-turn agent session):

| Step | Context, chunk | 4K prompt tok/s | 16K prompt tok/s | Short chat decode | Agent decode | Agent first token (s) | Cache hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 | 8K, 512 | 276.7 | - | 39.4 | - | - | - |
| V1 | 32K, 512 | 276.6 | 276.7 | 41.9 | 36.1 | 31.24 | 0.0% |
| VA | 64K, 512 | 276.2 | 276.6 | 42.8 | 44.5 | 30.92 | 0.0% |
| VB | 64K, 512 | 276.1 | 276.5 | 41.1 | 29.9 | 30.91 | 0.0% |
| VC | 200K, 512 | 265.4 | 265.5 | 41.9 | 28.8 | 37.12 | 0.0% |
| VC | 200K, 1024 | 518.4 | 519.8 | 42.4 | 25.9 | 17.72 | 0.0% |
| VD | 200K, 2048 | 982.7 | 985.8 | 43.2 | 41.3 | 10.13 | 0.0% |
| VD, requests with `presence_penalty` 0 (drafts on) | 200K, 2048 | 982.1 | 986.4 | 47.2 | 41.8 | 10.11 | 0.0% |
| VD + `TURN_CKPT` (requests with `presence_penalty` 0) | 200K, 2048 | 958.2 | 969.9 | 46.9 | 45.3 | 4.18 | 68.2% |
| VD + `TURN_CKPT` + `PENALTY_SPEC` (model card penalty) | 200K, 2048 | 957.3 | 969.8 | 45.1 | 43.4 | 4.14 | 68.2% |
| final: + `ATTN_GQA` (3 runs, 2026-10-18) | 200K, 2048 | 953.9 | 967.4 | 45.2 | 45.7 | 4.13 | 68.2% |

Why each step:

1. **#2917.** Main fails at 16K (8K is the largest that starts): the hybrid mode's buffers and staging do not fit beside the weights and the FP32
   caches. #2917 gives the packed Q8_0 copies (1.78 GiB) back for the staging; 32K starts, 64K does not.
2. **`--kv int8`.** The FP32 caches take 2 KiB per cell per store in each of the 10 attention layers (20 KiB per
   cell per store). Two int8 pull requests were tried. #2698 (VA) and #2742 (VB) both start at 64K, but at the agent
   session's last turn (14.5K tokens in the K / V cache, the whole chat read again) VA decodes at 42.5
   tok/s and VB at 24.0. The branch keeps #2698's layout and decode kernels, and only #2742's single staging buffer.
3. **Past 64K** (VC). At 128K and 200K the sessions' caches no longer fit before the hybrid buffers. The packed
   copies are now given back before the sessions are allocated, and the prompt attention's FP16 K / V is blocked.
   Blocked prompt attention measured on its own (VC, 64K context, chunk 2048, 1 run, needle prompts): a 61,342-token
   prompt read at 967.5 tok/s blocked against 826.8 with `STRATA_Q36_KV16_BLOCK=0`, 32,792 tokens at 1,089.8 against
   1,077.7. The needle was found both ways.
4. **The prompt chunk.** With every expert on the CPU pool, prompt speed follows the chunk: 512 -> 265 tok/s, 1024 ->
   519, 2048 -> 983 (VC, VD at 200K). VC runs chunk 2048 at 64K (1,101 tok/s at 16K) but not at 204,800. The 190K needle took 485.5 s at
   chunk 1024 (VC, 394.6 tok/s) and 390.0 s at chunk 2048 (VD, 489.9 tok/s). VD's savings (287 MiB less in the init
   and scratch buffers, and the hybrid buffers' need from 692 to 572 MiB, by the code's arithmetic) make chunk 2048
   fit at 204,800.
5. **MTP drafts with the penalty** (`STRATA_Q36_PENALTY_SPEC`). With the model card's penalty, `--mtp 2` took no drafts:
   the short chat decoded at 43.2 tok/s. Sending `presence_penalty` 0 instead gave 47.2 with drafts. With the switch
   and the model card's penalty: 45.1 (1 run) and 45.2 (final, 3 runs, range 41.8-52.8), with drafts accepted (for
   example 62 of 74).
6. **Turn checkpoint** (`STRATA_Q36_TURN_CKPT`; both rows with requests at `presence_penalty` 0). Without it the
   cache hit over the follow-up turns was 0.0% and 30,177 earlier-turn tokens were read again per session. With it:
   68.2%, 277 tokens, and the follow-up turns' first token went from 10.11 s to 4.18 s. Fresh prompts read about 2%
   slower in these single runs (4K: 982.1 -> 958.2 tok/s); probably because the read is split in two at the turn
   boundary, but that cause was not measured, and the only spread available is the final suite's 3 runs (951-960).
7. **GQA decode attention** (`STRATA_Q36_ATTN_GQA`). At 150,000 cells the old decode attention took 6.0 ms per
   attention layer for one int8 row, 60 ms over the 10 layers (microbenchmark below). The like-for-like measurement
   is the end-to-end on / off test below: 13.6 -> 42.9 tok/s after 110,094 tokens. Decode after a 147,915-token
   prompt (256 tokens, 1 run each): 10.5 tok/s with VD + `TURN_CKPT` and requests at `presence_penalty` 0 (drafts
   on), 37.8 with the final configuration (model card penalty 1.5, `PENALTY_SPEC`, `ATTN_GQA`); after 29,508 / 29,509 tokens:
   38.4 -> 41.9. The sampling differs between these two runs, so they change more than the kernel. Details below.
8. **Parked conversations** (`STRATA_Q36_PARK_MIB=3072`, on top of the final configuration, 1 run each). An agent-like
   sequence: a 19,099-token chat (a system prompt of 19,075 tokens) for 5 turns with a 46-token side request after each, then
   a second chat with the same system prompt for 2 turns, then the first chat's last turn (13 requests, 48-token greedy
   answers). Without parking 12.4% of the prompt tokens were reused and the prompts took 147.4 s (every main turn read
   ~19.2K tokens in 20.5 s). With it 87.2% and 29.8 s: each main turn read 47-78 tokens in 0.7-1.2 s, the second chat
   read 22 tokens after the borrowed root, and each park took 173-256 ms. Same tokens: a 3K-token chat for 4 turns (96-token
   greedy answers, `STRATA_Q36_PARK_MIN=500 STRATA_Q36_PARK_ROOT=1000`) with a side request after each turn (every turn
   restored from the park) gave the same text, byte for byte, as the same chat without side requests (every turn
   continued in the live session): 4 of 4. Against reading the chat in full (parking off) the texts differ after a few
   words, as they do for the turn checkpoint: the read is cut at other points.

## The GQA decode attention kernel

Qwen3.6-35B-A3B has 16 query heads and 2 KV heads (head size 256) in its 10 attention layers. The old split-K kernel
reads every K / V cell once per query head. The new one reads it once for the 8 heads that share it, in 256-cell
tiles with an online softmax, and writes the same partials; a new merge combines the chunks.

**Microbenchmark** (a standalone harness built from the engine's kernels, random K / V, 150,000 cells, ms per
attention layer for part + merge, 20 runs after 3 warm-ups, RTX 4070 Laptop, sm_89 build; new = chunk 1024, the
default; T = query rows in the window):

| Store | T | old | new | GB/s of K / V read once (new) |
| --- | ---: | ---: | ---: | ---: |
| int8 | 1 | 6.00 | 0.785 | 202 |
| int8 | 2 | 10.48 | 0.709 | 224 |
| int8 | 3 | 15.71 | 0.745 | 213 |
| fp32 | 1 | 31.66 | 2.53 | 243 |
| fp32 | 2 | 64.29 | 2.56 | 240 |
| fp32 | 3 | 96.85 | 2.58 | 238 |

The same harness compiled as compute_75 PTX and JIT-compiled on this sm_89 card (the CUDA-core int8 path that cards
before sm_80 take): int8 T=1 5.25 -> 0.730 ms, T=3 15.71 -> 1.35 ms; fp32 as above.

**Parity** (same harness, outputs after the merge and the gate, against a double-precision reference, max |x - ref|
/ max |ref|): at 150,000 cells, T=1, int8 old 2.87e-6 / new 1.75e-6, fp32 old 9.55e-7 / new 5.43e-7. Over 1,000,
32,768 and 150,000 cells, T = 1-3 and chunks 256-4096 the new kernel stays within 3.3e-6 of the reference (int8)
and 1.25e-6 (fp32). A row's output is bitwise the same alone or in a 3-row window, and with or without the graph's
grid (30 of 30 cases). The compute_75 build passes the same checks.

**End to end** (final configuration with and without `STRATA_Q36_ATTN_GQA=1`, a 110,094-token prompt (the engine's
source file as one user message), 128 greedy tokens, thinking on, **2 runs each**, 2026-10-17/18; the first run of
each side, on 2026-10-17, was on the build just before the final one, the second on the final build):

| | without | with |
| --- | ---: | ---: |
| decode after 110,094 tokens | 13.6 / 13.6 tok/s | 42.8 / 42.9 tok/s |
| prompt read | 169.45 / 169.47 s | 169.24 / 169.26 s |
| decode, 23-token prompt, 40 tokens | 52.0 / 52.9 tok/s | 56.3 / 56.5 tok/s |

The text is **not** the same with and without: the two 128-token answers agree for the first 107 characters, then
one says "source file" where the other says "source code file". The kernels sum in a different order. The 23-token
answer starts the same way in both. The new kernel was not compared with the old one by KL or perplexity: the checks
are the parity harness above and the six needles.

## Compared with Qwen3.8-Flash-Next Coder IQ1_M on the same laptop

The upstream engine (0.1.45, commit 477ac13c) with this laptop's tuned config for the Coder IQ1_M (`--kv q4_0`,
`--kv-resident 20480`, `--mmap-experts`, `--spec 4 --mtp rt` and other flags, full line below), also at 204,800
context. Suite: 3 runs, 2026-10-17. Long prompts and needles: 1 run. These are **different models** at very
different bit widths. The table compares speed only, not answer quality.

| | Qwen3.6-35B-A3B UD-Q4_K_XL (this branch) | Flash-Next Coder IQ1_M (upstream) |
| --- | ---: | ---: |
| short chat decode | 45.2 tok/s | 30.8 |
| short chat first token | 0.58 s | 1.61 |
| 4K prompt | 953.9 tok/s | 510.0 |
| 16K prompt | 967.4 tok/s | 575.4 |
| agent follow-ups: decode / first token / cache hit | 45.7 tok/s / 4.13 s / 68.2% | 29.9 / 7.34 / 68.7% |
| 29.5K prompt: first token / decode (client) | 32.7 s / 41.8 tok/s | 80.0 / 28.9 |
| 148K prompt: first token / decode (client) | 264.0 s / 37.7 tok/s | 258.5 / 28.6 (stopped after 72 tokens) |
| needles found | 6 of 6 (64K and 190K, depths 10/50/90) | 2 of 2 (64K and 190K, depth 50) |
| 64K / 190K needle request | 84.4 s / 389.4 s | 131.0 / 309.2 (190K: 16,384 tokens reused from the cache) |
| prompt read, engine log: 148K / 190K | 561.0 / 489.7 tok/s | 574.8 / 565.2 (190K: over the 174,079 tokens read) |

The Coder's 190K needle request reused 16,384 cached tokens (`190463 tokens = 16384 reused + 174079 read`), so it was
not a fresh read like Qwen's. From about 148K up the Coder reads prompts faster, and its first token at 148K came
earlier (258.5 s against 264.0). Qwen is faster on the 4K-64K prompts, in decode, and on the first token of short
prompts.

The Coder's command line (from its benchmark report, 2026-10-17, file paths shortened to names):

```text
strata --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 204800 --kv-resident 20480 --kv q4_0 --mmap-experts --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97 --mtp-draft-vocab draft_vocab_en.bin --vram-reserve-mib 1150 --prefill-head-alias --short-read 0
```

## Not tested

- **Only CUDA on sm_89.** HIP and SYCL were not built. The new code is in `q36_engine.cu`, which the HIP build
  compiles too; nothing here was compiled or run on an AMD card. Intel Arc does not run this model family.
- **One card.** No other NVIDIA card was run. The pre-sm_80 path of the GQA kernel ran only as compute_75 PTX
  JIT-compiled on this sm_89 card, in the parity harness. No Turing or older card, and no sm_86 / sm_90 / sm_120.
- **One machine, one model.** UD-Q4_K_XL only (not UD-Q6_K_XL or Q8_0), one slot (no `--batch` / `"parallel"`), no
  pictures (`--vision`), Linux only (not Windows).
- **Quality.** No KL or perplexity check of `--kv int8`, of the blocked prompt attention or of the GQA kernel against
  the old kernels. The checks are the parity harness and the needles.
- **Same tokens.** Not checked token for token: that a penalized greedy request gives the same text with and without
  `STRATA_Q36_PENALTY_SPEC`, and that a turn resumed from `STRATA_Q36_TURN_CKPT` gives the same tokens as reading the
  whole chat again. Both are designed to give the same text, but that was not measured.
- **Parking** (`STRATA_Q36_PARK_MIB`). Measured with the scripted sequence above, not a real Claude Code session; not
  with `--batch` (a batch slot's request neither parks nor restores) or pictures; no conversation above 19.4K tokens was parked (the image grows
  about 11.6 KB per token by the code: about 1.7 GiB at 150K). The extra cut at the system prompt's end (only when
  that root is not parked yet) was not timed apart.
- **Speed runs.** The step-by-step screens are 1 run each. Only the final suite has 3 runs; the end-to-end GQA test
  has 2.
- **Upstream.** #2917, #2698 and #2742 are open pull requests, not merged upstream. Their merges here were resolved by
  hand onto 477ac13c (#2698 was written against an older main and was ported to the sessions and M-RoPE code main
  has since). Upstream's gates in `tools/qwen35/` are not among the results here.

## Credits and license

Strata is by Niko1221 and the Strata contributors ([github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)).
This branch is a fork and a derivative work, under the same [MIT License](../LICENSE). The upstream pull requests
merged here are the work of their authors: **CC-David-CC** (#2917), **TorBere** (#2698) and **ggeorgovassilis**
(#2742). The model is Qwen's (Apache 2.0) and the GGUF files are Unsloth's.
