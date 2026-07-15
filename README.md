# kukuru-tts

Thai fork of [kikiri-tts](https://github.com/semidark/kikiri-tts): fine-tune
[Kokoro-82M](https://github.com/hexgrad/kokoro) for **Thai** (five-tone
phoneme format) with a vendored, patched [StyleTTS2](https://github.com/yl4579/StyleTTS2).
Runtime lives in [FastThaiG2P](https://github.com/cstorm125/FastThaiG2P) —
this repo produces the checkpoints it serves.

**Start here → [`docs/THAI_RUNBOOK.md`](docs/THAI_RUNBOOK.md)** — the
measured, end-to-end recipe: dataset prep → Stage 1 → Stage 2 → package →
ship. It encodes the lessons from the original training run (single-GPU only, LR ceiling,
Stage 2 OOM handling, voicepack mixing, ONNX validation). The full training
curves — including the failed multi-GPU and LR-scaling experiments worth
not repeating — are public on
[wandb.ai/cstorm125/kokoro-thai](https://wandb.ai/cstorm125/kokoro-thai).

Thai-specific entry points:
- `scripts/prepare_thai_dataset.py` — (text, wav) pairs → validated training lists
- `scripts/train_thai.sh` — both stages, watchdogged
- `scripts/package_thai.py` — checkpoints → pth/ONNX/voicepack/samples
- `configs/thai_stage1.yml` / `configs/thai_stage2_resume_example.yml`

Differences from upstream kikiri-tts: StyleTTS2 is vendored (not a
submodule) with Thai-run patches committed — wandb audio logging,
checkpoint-loader prefix fix, stage-2 resume fix, configurable grad clip.
German-specific configs/scripts/docs were removed; `docs/ARCHITECTURE.md`
and `docs/TROUBLESHOOTING.md` are inherited from upstream and document the
shared StyleTTS2 machinery (some examples still reference German runs).

---

For the German recipe this repo was forked from, see [kikiri-tts](https://github.com/semidark/kikiri-tts).
