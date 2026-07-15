# AGENTS.md — kukuru-tts

## Project Overview

kukuru-tts is a training recipe for fine-tuning [Kokoro TTS](https://github.com/hexgrad/kokoro)
(82M parameters, based on StyleTTS 2) for **Thai** with a five-tone phoneme
format. Forked from [kikiri-tts](https://github.com/semidark/kikiri-tts)
(German). The runtime that serves the trained models lives in
[FastThaiG2P](https://github.com/cstorm125/FastThaiG2P) — this repo produces
the checkpoints, that repo consumes them.

The repo contains:

- `StyleTTS2/` — **vendored** (not a submodule) patched StyleTTS2: wandb audio
  logging, checkpoint-loader module-prefix fix, stage-2 resume fix,
  configurable grad clip
- `kokoro/` — inference package as a git submodule (`semidark/kokoro`)
- `scripts/` — Thai pipeline: `prepare_thai_dataset.py` → `train_thai.sh` →
  `package_thai.py`; `test_inference.py` for checkpoint verification;
  `extract_voicepack.py` / `baseline_thai_samples.py` as building blocks
- `configs/` — `thai_stage1.yml` (the proven recipe), stage-2 resume example,
  smoke-test config
- `docs/THAI_RUNBOOK.md` — **the canonical guide**: end-to-end recipe with
  measured reference numbers and a failure table

- **Primary language:** Python 3.10–3.12 (pinned via `.python-version`)
- **Package manager:** `uv` (lockfile: `uv.lock`)
- **License:** Apache 2.0
- **Repository:** `https://github.com/cstorm125/kukuru-tts`

## Build & Install

See `docs/THAI_RUNBOOK.md` §1 — `uv sync` plus training extras plus
`fastthaig2p`. Run the sanity gate before anything else.

## Key invariants (violating these silently breaks training)

- Train on **one GPU** — DDP converges but audibly degrades voices
  (InstanceNorm micro-batch statistics)
- LR ceiling ≈ 1e-4 at any batch size; Stage 2 adversarial phase needs
  batch 8 on 80 GB
- Voicepack extraction mixes the **Stage 1** style encoder with the
  **Stage 2** prosody encoder
- Every phoneme char in training lists must exist in `config.json`'s vocab —
  `prepare_thai_dataset.py` enforces this; never bypass it
- Do not INT8-quantize exported ONNX (AdaIN layers break)

## Verification

- `configs/thai_smoke.yml` — 2-epoch smoke of both stages on a small dataset
- `uv run scripts/test_inference.py` — zero-config synthesis from the
  released model (needs GITHUB_TOKEN while FastThaiG2P is private)
- Training curves: https://wandb.ai/cstorm125/kokoro-thai
