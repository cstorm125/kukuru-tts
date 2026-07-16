# Thai Kokoro Fine-tune — Runbook

End-to-end: Thai `(text, wav)` pairs → deployable ONNX voice in
[FastThaiG2P](https://github.com/cstorm125/FastThaiG2P). Every number in
here was measured on the original 2026-07 training run (p5.48xlarge, H100s);
lessons are marked ⚠. All training curves are public in the
[wandb project](https://wandb.ai/cstorm125/kokoro-thai) — the runs to
compare yours against are `kokoro-thai-v1-replica-stage1` and
`…-stage2`; the other runs are the failed batch/LR experiments described
below (useful as what-divergence-looks-like references).

## 0. What you need

- `(text, wav)` pairs: 24 kHz mono WAVs + a `metadata` file (`filename|text`).
  ~20k utterances of a single speaker worked well.
- `kokoro_base.pth` (Kokoro-82M in StyleTTS2 format) and `config.json`
  (115-symbol vocab, `↑`=170) — in `s3://fast-thai-g2p/kokoro-thai/`.
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
# (while FastThaiG2P is private: use https://<user>:<token>@github.com/... )
# training/: put kokoro_base.pth + config.json + kokoro_symbols.py here
```

wandb: run `wandb login` once (or `WANDB_DISABLED=true` to opt out).
`WANDB_PROJECT=<name>` picks the project — use a scratch project for test
runs so real training curves stay clean.

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

Then make the audio visible where the config expects it
(`root_path: ../training/audio`):

```bash
ln -s "$(pwd)/dataset/audio" training/audio
```

**Bring-your-own-data requirements:** 24 kHz mono WAVs (resample first —
the loader does not resample); single speaker (the recipe runs
`multispeaker: false`); clips roughly 1–15 s; text is plain Thai (the
normalizer converts numbers/latin, but heavy code-switching gets
letter-spelled). ~20k utterances produced the shipped voice; a few
hundred is enough to smoke-test the pipeline but not for quality.

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
| stage 2 total | ~0.3 → 0.23 pre-adversarial; steps to ~0.33 at `joint_epoch` (expected: decoder leaves its L1 optimum + GT target switches to real audio) then flat | NaN: symbol mapping wrong |
| stage 2 val | 0.35 → 0.32 pre-adv; 0.324 → 0.318 through the adversarial epochs, F0 1.53 → 1.22 | val climbing epoch-over-epoch after joint_epoch |
| adversarial ear check | first joint epoch sounds ROUGHER than the epoch before it — recovers within ~2 epochs; final A/B'd audibly equivalent to pre-adversarial with better F0 | still rougher after 3+ joint epochs |
| epoch 1 val | ~0.53 (high epoch-1 val is EXPECTED — the ↑ tone embedding starts cold and catches up by epoch 3) | |

⚠ **Stage 2 must run at batch 8 once the adversarial phase starts**
(`joint_epoch`, displayed epoch 6): decoder+GAN+WavLM grads exceed 80 GB
at batch 16. Simplest: run stage 1, then edit `batch_size: 16` → `8` in
`configs/thai_stage1.yml` before launching stage 2 (both stages read the
same config; stage 1 genuinely wants 16). Or resume a crashed run with
`configs/thai_stage2_resume_example.yml` (halved batch,
`second_stage_load_pretrained: true`, `load_only_params: false`).

⚠ **Checkpoints save every `save_freq` (2) epochs and the final epoch is
NOT special-cased** — a 10-epoch stage 2 leaves `epoch_2nd_00008.pth` as
the last artifact. Fine in practice (val is flat by then), but set
`save_freq: 1` if you want the literal last epoch.

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
    --output-dir dist/thai
```

Emits inference `.pth`, validated ONNX (expect waveform corr > 0.99 vs
torch), voicepack (`.pt` + `.npy`), config, and sound-check WAVs.
⚠ The voicepack must use the **stage 1** style encoder with the stage 2
prosody encoder — that's what `--style-encoder-model` is for.
⚠ Do NOT INT8-quantize the ONNX: the AdaIN style layers break outright
(corr 0.02). fp32 on CPU is already fast — RTF ≈ 0.2 with 8 threads
(1 s of speech costs 0.2 s of compute; lower is better), and even a
single thread manages RTF ≈ 0.7, still faster than realtime.
⚠ Listen to `samples/tones.wav`-style minimal pairs (มาใหม่ ไม้ใหม่
ไม่ไหม้ ไหม) — val mel cannot tell you whether the 5 tones are distinct.

## 5. Ship to FastThaiG2P

Upload `kokoro_thai.onnx`, `thai_som.npy`, `config.json` as assets on a
GitHub release of FastThaiG2P, then bump `_RELEASE_URL` in
`fastthaig2p/tts.py`. Zero-arg `TTS()` serves the new voice.

## Sync out — the training box is ephemeral

```bash
aws s3 sync dist/thai/ s3://<your-bucket>/kokoro-thai/checkpoints/...
aws s3 cp StyleTTS2/logs/thai/first_stage.pth s3://<your-bucket>/kokoro-thai/checkpoints/
```

## Dress rehearsal (2026-07-15)

The runbook was replayed literally from a fresh `git clone` on a
1,000-pair stand-in dataset: env setup → sanity gate → dataset prep →
full stage 1 (20 ep) → full stage 2 (10 ep @ batch 8, adversarial phase
included) → package → `test_inference.py` on the packaged ONNX. Curves:
[wandb kukuru-rehearsal](https://wandb.ai/cstorm125/kukuru-rehearsal).
Every gap found was folded back into this document.

## Smoke test (verify the pipeline before a real run)

The whole chain was verified 2026-07-14 on a 400-utterance synthetic set:
`prepare_thai_dataset.py` → 2-epoch stage 1 (val 1.02, ~8 min) → 2-epoch
stage 2 (val 0.83) → `package_thai.py` produced pth+ONNX+voicepack+samples.
`configs/thai_smoke.yml` is that config — run it after any change to the
vendored StyleTTS2 to confirm nothing broke. (Loss values that high are
expected at 2 epochs on 400 clips; the point is no crash/NaN and audible
speech in `dist/smoke/samples/`.)

## Reference run (2026-07-13 → 07-15, completed end-to-end)

The run this runbook is distilled from, start to finish on one H100:

| stage | config | wall-clock | result |
|---|---|---|---|
| Stage 1 | batch 16, flat 1e-4, 20 epochs | ~7.5 h (~22 min/epoch) | best val mel **0.2257** @ epoch 12 (`first_stage.pth`) |
| Stage 2 pre-adv | batch 16, epochs 1-5 | ~2 h (~22 min/epoch) | val 0.347 → 0.332, dur 0.23 → 0.21 |
| Stage 2 adversarial | batch 8 (resumed), epochs 6-10 | ~4 h (~50 min/epoch) | val **0.318**, F0 **1.22**; last saved ckpt `epoch_2nd_00008.pth` |
| Package | `package_thai.py` | ~15 min | ONNX corr > 0.995 vs torch; CPU RTF ≈ 0.2 |

Shipped as [FastThaiG2P v0.3.0](https://github.com/cstorm125/FastThaiG2P/releases/tag/v0.3.0)
(zero-arg `TTS()` downloads it). Full artifacts under
`s3://fast-thai-g2p/kokoro-thai/`; training curves at
[wandb.ai/cstorm125/kokoro-thai](https://wandb.ai/cstorm125/kokoro-thai)
(runs `kokoro-thai-v1-replica-stage1` / `-stage2`; other runs are the
failed batch/LR experiments — useful divergence references).
