# Architecture and Compatibility Notes

> Inherited from upstream [kikiri-tts](https://github.com/semidark/kikiri-tts) (German).
> The StyleTTS2 machinery it documents is shared; some examples reference German runs.
> For the Thai recipe, start at [`THAI_RUNBOOK.md`](THAI_RUNBOOK.md).


Technical reference for Kokoro-82M fine-tuning compatibility.

For how-to training steps, use `THAI_RUNBOOK.md`.

## Kokoro-82M Component Layout

Reference component sizes used for checkpoint compatibility checks:

| Component | Parameters |
|---|---|
| bert (PLBERT) | 6.29M |
| bert_encoder | 0.39M |
| predictor | 16.19M |
| text_encoder | 5.61M |
| decoder (ISTFTNet) | 53.28M |
| Total | 81.76M |

Voicepack target shape:
- `[510, 1, 256]` (float32)

## weight_norm API Compatibility

### Why it matters

Old API (`torch.nn.utils.weight_norm`) and new API (`torch.nn.utils.parametrizations.weight_norm`) create different state-dict key layouts.

If StyleTTS2 is trained with old API and inference expects new API, checkpoint loading can be brittle and may fail silently under non-strict loading paths.

### Required status

StyleTTS2 patched files must use new parametrizations API:
- `StyleTTS2/models.py`
- `StyleTTS2/Modules/istftnet.py`
- `StyleTTS2/Modules/hifigan.py`
- `StyleTTS2/Modules/discriminators.py`

## Symbol Mapping Compatibility

Kokoro and default StyleTTS2 use different token index assignments.

Implication:
- same symbol set size does not imply index compatibility

Requirement:
- `StyleTTS2/text_utils.py` must use Kokoro mapping (`kokoro_symbols.py`)

## German G2P Notes

- G2P backend: `misaki` + `espeak-ng`
- German code path uses `espeak.EspeakG2P(language='de')`
- Symbol `ʏ` is not in Kokoro vocab and must be normalized to `y`

## German Phoneme Compatibility

All standard German phonemes are covered by Kokoro's 178-token set:

| Sound | IPA | Unicode | Kokoro ID |
|-------|-----|---------|-----------|
| ich-Laut | `ç` | U+00E7 | 78 |
| ach-Laut | `x` | U+0078 | 66 |
| ö long | `ø` | U+00F8 | 116 |
| ö short | `œ` | U+0153 | 120 |
| ü long | `y` | U+0079 | 67 |
| ts affricate | `ʦ` | U+02A6 | 20 |
| schwa-r | `ɐ` | U+0250 | 70 |
| sch | `ʃ` | U+0283 | 131 |
| ng | `ŋ` | U+014B | 112 |
| vowel length | `ː` | U+02D0 | 158 |
| schwa | `ə` | U+0259 | 83 |
| uvular r | `ʁ` | U+0281 | 128 |
| glottal stop | `ʔ` | U+0294 | 148 |

### Missing symbol

| IPA | Unicode | Meaning | Fix |
|-----|---------|---------|-----|
| `ʏ` | U+028F | short ü | Map to `y` (U+0079) |

`ʏ` is produced by `espeak-ng` for short ü (e.g., in "Bücher"). It is not in Kokoro's vocabulary. Replace it with `y` (long ü) in post-processing. The model learns the duration difference from the audio context.

### Diacritics (stress markers)

| Symbol | Meaning | Kokoro ID |
|--------|---------|-----------|
| `ˈ` | primary stress | 156 |
| `ˌ` | secondary stress | 157 |

These are produced by `espeak-ng` and are in Kokoro's vocabulary. Do not strip them.

## Sequence Length Constraint

- PLBERT max position embeddings: 512
- Practical training cap: 510 cleaned tokens

Samples above this should be filtered before batching.

## Inference Packaging Notes

When exporting trained checkpoints for `KModel`, ensure the expected components are present and keys align with Kokoro inference code:
- `bert`
- `bert_encoder`
- `predictor`
- `text_encoder`
- `decoder`

Use `scripts/test_inference.py` to verify conversion and produce sample outputs.
