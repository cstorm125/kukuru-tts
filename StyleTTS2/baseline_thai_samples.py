#!/usr/bin/env python3
"""Synthesize Thai samples with PRETRAINED (non-finetuned) Kokoro weights.

End-to-end sanity check of the inference path before fine-tuning:
FastThaiG2P → ipa_to_kokoro → KModel(kokoro_base.pth) → WAV.

The voice will sound wrong (English voicepack, no Thai training) but it
verifies the phoneme mapping, vocab, model loading, and synthesis pipeline.

Usage:
    .venv/bin/python3 scripts/baseline_thai_samples.py --output-dir test_output/baseline
"""

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_kokoro_submodule = _repo_root / "kokoro"
if _kokoro_submodule.exists() and str(_kokoro_submodule) not in sys.path:
    sys.path.insert(0, str(_kokoro_submodule))

TEST_SENTENCES = [
    # Greeting (polite particles ค่ะ/ครับ)
    "สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา",
    # Numbers / money
    "ยอดเงินคงเหลือสามพันห้าร้อยบาท",
    # All five tones packed in
    "ใครขายไข่ไก่ ใกล้ๆ บ้านเรา",
    # Longer sentence with clusters
    "กรุณากดหมายเลขหนึ่งเพื่อติดต่อพนักงาน",
    # Question prosody
    "วันนี้อากาศเป็นอย่างไรบ้าง",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="test_output/baseline")
    parser.add_argument(
        "--model",
        default=str(_repo_root / "training" / "kokoro_base.pth"),
        help="StyleTTS2-format checkpoint ({'net': {...}}) or flat Kokoro weights",
    )
    parser.add_argument("--voicepack", default=None, help="Defaults to af_heart from HF")
    parser.add_argument("--config", default=str(_repo_root / "training" / "config.json"))
    parser.add_argument(
        "--legacy-tones",
        action="store_true",
        help="Fold high tone ↑ back to ↗ for models trained on v1 data "
        "(pre-5-tone mapping, where high and rising shared ↗)",
    )
    args = parser.parse_args()

    import torch
    import soundfile as sf
    from kokoro import KModel
    from fastthaig2p import G2P, ipa_to_kokoro

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # KModel wants a flat {component: state_dict} file; kokoro_base.pth wraps it in 'net'
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model_path = args.model
    if "net" in ckpt:
        model_path = str(out / "kokoro_base_flat.pth")
        torch.save(ckpt["net"], model_path)
        print(f"Unwrapped 'net' → {model_path}")

    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=args.config, model=model_path)
    kmodel = kmodel.to(device).eval()

    if args.voicepack:
        voicepack_path = args.voicepack
    else:
        from huggingface_hub import hf_hub_download

        voicepack_path = hf_hub_download("hexgrad/Kokoro-82M", "voices/af_heart.pt")
    voice = torch.load(voicepack_path, map_location="cpu", weights_only=True)
    print(f"Voicepack: {voicepack_path} {tuple(voice.shape)}")

    g2p = G2P()
    for i, text in enumerate(TEST_SENTENCES, 1):
        ipa = g2p.convert(text)
        phonemes = ipa_to_kokoro(ipa)
        if args.legacy_tones:
            phonemes = phonemes.replace("↑", "↗")
        print(f"\n[{i}/{len(TEST_SENTENCES)}] {text}")
        print(f"  phonemes: {phonemes}")
        ref_s = voice[len(phonemes) - 1]
        audio = kmodel(phonemes, ref_s)
        wav_path = out / f"baseline_{i:02d}.wav"
        sf.write(str(wav_path), audio.numpy(), 24000)
        print(f"  saved: {wav_path} ({len(audio) / 24000:.1f}s)")

    print(f"\nDone. Samples in {out}/")


if __name__ == "__main__":
    main()
