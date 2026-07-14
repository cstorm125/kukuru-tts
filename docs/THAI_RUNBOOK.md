# Thai Kokoro Fine-tune — Runbook

End-to-end: Thai `(text, wav)` pairs → deployable ONNX voice in
[FastThaiG2P](https://github.com/cstorm125/FastThaiG2P). Every number in
here was measured on the 2026-07-13/14 v2 run (p5.48xlarge, H100s);
lessons are marked ⚠.

## 0. What you need

- `(text, wav)` pairs: 24 kHz mono WAVs + a `metadata` file (`filename|text`).
  ~20k utterances of a single speaker worked well.
- `kokoro_base.pth` (Kokoro-82M in StyleTTS2 format) and `config.json`
  (115-symbol v2 vocab, `↑`=170) — in `s3://fast-thai-g2p/kokoro-thai/`.
- FastThaiG2P ≥ 2db2c26 (5-tone `ipa_to_kokoro`).
- One 80 GB GPU. ⚠ More GPUs do NOT help: DDP training converged to the
  same val mel but audibly worse voices (InstanceNorm layers see
  micro-batch statistics); large batches need LR scaling the GAN can't
  tolerate (2.8e-4 diverged, 1.4e-4 NaN'd at epoch 8; 1e-4 is the proven
  ceiling at any batch size we tried).

## 1. Setup

```bash
git clone https://github.com/cstorm125/kukuru-tts && cd kukuru-tts
git submodule update --init   # kokoro (StyleTTS2 is vendored, patches included)
uv sync
uv pip install torchaudio==2.6.0 librosa munch pandas matplotlib tensorboard \
    einops einops-exts nltk accelerate pythainlp wandb pytest \
    onnx onnxruntime \
    "git+https://github.com/resemble-ai/monotonic_align.git"
uv pip install "git+https://github.com/cstorm125/FastThaiG2P.git"
# training/: put kokoro_base.pth + config.json + kokoro_symbols.py here
```

Sanity gate (must print `True 115` and `maː↑ maː↗`):
```bash
.venv/bin/python3 -c "
import json; from fastthaig2p import ipa_to_kokoro
v = json.load(open('training/config.json'))['vocab']
print(v.get('↑') == 170, len(v)); print(ipa_to_kokoro('/maː˦˥/ /maː˩˩˦/'))"
```

## 2. Dataset → training lists

```bash
.venv/bin/python3 scripts/prepare_thai_dataset.py \
    --metadata dataset/metadata.csv --audio-dir training/audio \
    --config training/config.json --output-dir training
```

Normalizes text (numbers/emails/times → spoken Thai), G2Ps, maps to the
5-tone Kokoro format, splits train/val, writes OOD list. ⚠ It hard-fails
on any phoneme character outside the vocab — this caught literal `-`
characters that would have silently corrupted training. Never bypass it.

## 3. Train

```bash
GPU=0 bash scripts/train_thai.sh all    # stage1 then stage2, watchdogged
```

`configs/thai_stage1.yml` is the proven config: **batch 16, flat lr 1e-4,
20 epochs**, no warmup/decay/early-stop (grad_clip 0). Reference numbers
on one H100 (~22 min/epoch, both stages):

| signal | healthy | broken |
|---|---|---|
| stage 1 val mel | 0.53 → 0.31 → 0.26 → … → **≈0.225 plateau by ep 12-14** | starts 7-8: weights didn't load. Jumps to 1.6 + disc loss collapsing to <2: LR too high, restart lower |
| stage 1 gen/disc | gen ~3, disc ~4, both flat | disc → 0 or gen climbing while disc falls |
| stage 2 total | ~0.3 → 0.23 pre-adversarial | NaN: symbol mapping wrong |
| epoch 1 val | ~0.53 (worse than v1's 0.28 is EXPECTED — the ↑ tone embedding starts cold and catches up by epoch 3) | |

⚠ **Stage 2 OOMs at batch 16 when the adversarial phase starts**
(`joint_epoch`, displayed epoch 6): decoder+GAN+WavLM grads exceed 80 GB.
Options: batch_size 8 from the start, or let it crash after
`epoch_2nd_00004.pth` and resume with `configs/thai_stage2_resume_example.yml`
(halved batch, `second_stage_load_pretrained: true`, `load_only_params: false`).

⚠ Two resume bugs are fixed in this repo's vendored StyleTTS2 — if you
ever rebase onto upstream, re-check them: `models.py::load_checkpoint`
silently loaded zero tensors when `module.` prefixes mismatched
(strict=False hid it); `train_second.py` wiped the trained
predictor_encoder on every resume by unconditionally re-copying the
style encoder over it.

## 4. Package

```bash
.venv/bin/python3 scripts/package_thai.py \
    --stage2-ckpt StyleTTS2/logs/thai/epoch_2nd_00008.pth \
    --stage1-ckpt StyleTTS2/logs/thai/first_stage.pth \
    --audio-dir training/audio --config training/config.json \
    --output-dir dist/thai_v2
```

Emits inference `.pth`, validated ONNX (expect waveform corr > 0.99 vs
torch), voicepack (`.pt` + `.npy`), config, and sound-check WAVs.
⚠ The voicepack must use the **stage 1** style encoder with the stage 2
prosody encoder — that's what `--style-encoder-model` is for.
⚠ Do NOT INT8-quantize the ONNX: the AdaIN style layers break outright
(corr 0.02). fp32 CPU RTF is ~0.2 (8 threads) / ~0.7 (1 thread) anyway.
⚠ Listen to `samples/tones.wav`-style minimal pairs (มาใหม่ ไม้ใหม่
ไม่ไหม้ ไหม) — val mel cannot tell you whether the 5 tones are distinct.

## 5. Ship to FastThaiG2P

Upload `kokoro_thai.onnx`, `thai_som.npy`, `config.json` as assets on a
GitHub release of FastThaiG2P, then bump `_RELEASE_URL` in
`fastthaig2p/tts.py`. Zero-arg `TTS()` serves the new voice.

## Sync out — the training box is ephemeral

```bash
aws s3 sync dist/thai_v2/ s3://fast-thai-g2p/kokoro-thai/checkpoints/v2/...
aws s3 cp StyleTTS2/logs/thai/first_stage.pth s3://.../checkpoints/v2/
```

## Smoke test (verify the pipeline before a real run)

The whole chain was verified 2026-07-14 on a 400-utterance synthetic set:
`prepare_thai_dataset.py` → 2-epoch stage 1 (val 1.02, ~8 min) → 2-epoch
stage 2 (val 0.83) → `package_thai.py` produced pth+ONNX+voicepack+samples.
`configs/thai_smoke.yml` is that config — run it after any change to the
vendored StyleTTS2 to confirm nothing broke. (Loss values that high are
expected at 2 epochs on 400 clips; the point is no crash/NaN and audible
speech in `dist/smoke/samples/`.)

## History

The v2 (5-tone) run this runbook is distilled from: Stage 1 matched the
v1 single-GPU reference exactly (best val mel 0.2257 vs 0.2247) after we
abandoned multi-GPU. Full artifacts under
`s3://fast-thai-g2p/kokoro-thai/` (checkpoints/v2, samples/v2, voices).
