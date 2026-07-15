#!/usr/bin/env python3
"""Verify a trained checkpoint end-to-end: convert → synthesize Thai samples.

Zero-config (downloads the released Thai ONNX model + voicepack):
    uv run scripts/test_inference.py

Verify a fresh Stage 2 training checkpoint:
    uv run scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/thai/epoch_2nd_00008.pth \
        --voicepack dist/thai/thai_som.pt

Set GITHUB_TOKEN for the zero-config path while the FastThaiG2P repo is
private. Requires `pip install fastthaig2p` (and onnxruntime for .onnx).
"""

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Default reference model for zero-config runs — the released Thai voice.
DEFAULT_RELEASE_URL = (
    "https://github.com/cstorm125/FastThaiG2P/releases/download/v0.3.0"
)
DEFAULT_ASSETS = ("kokoro_thai.onnx", "thai_som.npy", "config.json")
MODEL_CACHE_DIR = REPO / "test_output" / ".model_cache"

TEST_SENTENCES = [
    "สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา",
    "ยอดเงินคงเหลือสามพันห้าร้อยบาทถ้วนค่ะ",
    "ใครขายไข่ไก่ ใกล้ๆ บ้านเรา",
    "กรุณากดหมายเลขหนึ่งเพื่อติดต่อพนักงาน",
    "มาใหม่ ไม้ใหม่ ไม่ไหม้ ไหม",  # 4-tone minimal pairs — listen closely
]


def convert_checkpoint(checkpoint_path: str, output_path: str) -> str:
    """Convert a StyleTTS2 Stage 2 checkpoint to Kokoro KModel format.

    Extracts the 5 inference components (bert, bert_encoder, predictor,
    text_encoder, decoder) from the training checkpoint. All state dict
    keys must have the 'module.' prefix for KModel's loading fallback
    to work correctly.

    Requires that training was done with the new parametrizations API
    (torch.nn.utils.parametrizations.weight_norm/spectral_norm) so the
    state dict keys are natively compatible with Kokoro's KModel.
    """
    import torch

    print(f"Converting checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = ckpt["net"]

    def ensure_module_prefix(state_dict):
        """Ensure all keys have 'module.' prefix for KModel compatibility."""
        return {
            ("module." + k if not k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    kokoro_weights = {}
    for key in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]:
        if key in net:
            kokoro_weights[key] = ensure_module_prefix(net[key])
            print(f"  {key}: {len(kokoro_weights[key])} keys")
        else:
            print(f"  WARNING: '{key}' not found in checkpoint")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, str(output))
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Saved Kokoro-format weights: {output} ({size_mb:.1f} MB)")
    return str(output)


def download_reference_file(filename: str) -> Path:
    """Lazily download a released asset into the cache. GITHUB_TOKEN is
    used via the API while the release repo is private."""
    import json
    import urllib.request

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_CACHE_DIR / filename
    if dest.exists():
        return dest
    print(f"Downloading {filename} ...")
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        tag = DEFAULT_RELEASE_URL.rsplit("/", 1)[-1]
        api = f"https://api.github.com/repos/cstorm125/FastThaiG2P/releases/tags/{tag}"
        req = urllib.request.Request(api, headers={"Authorization": f"token {token}"})
        release = json.load(urllib.request.urlopen(req))
        asset = next(a for a in release["assets"] if a["name"] == filename)
        req = urllib.request.Request(
            asset["url"],
            headers={"Authorization": f"token {token}",
                     "Accept": "application/octet-stream"},
        )
    else:
        req = urllib.request.Request(f"{DEFAULT_RELEASE_URL}/{filename}")
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test a fine-tuned Kokoro Thai model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--checkpoint",
        help="StyleTTS2 training checkpoint (.pth) — converted automatically",
    )
    group.add_argument(
        "--model",
        help="already-converted Kokoro weights (.pth) or ONNX (.onnx); "
        "omitted → downloads the released model",
    )
    ap.add_argument("--voicepack", help="voicepack (.pt/.npy)")
    ap.add_argument("--config", default=str(REPO / "training" / "config.json"))
    ap.add_argument("--output-dir", default=str(REPO / "test_output" / "inference"))
    args = ap.parse_args()

    model, voicepack, config = args.model, args.voicepack, args.config
    if args.checkpoint:
        model = str(Path(args.output_dir) / "converted.pth")
        convert_checkpoint(args.checkpoint, model)
    if not model:
        model = str(download_reference_file(DEFAULT_ASSETS[0]))
        voicepack = voicepack or str(download_reference_file(DEFAULT_ASSETS[1]))
        config = str(download_reference_file(DEFAULT_ASSETS[2]))
    if not voicepack:
        ap.error("--voicepack is required with --checkpoint/--model")

    from fastthaig2p import TTS

    tts = TTS(model, voicepack, config_path=config, device="cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(TEST_SENTENCES, 1):
        p = tts.synthesize(text, str(out / f"test_{i:02d}.wav"))
        print(f"  [{i}/{len(TEST_SENTENCES)}] {text[:40]} → {p}")
    print(f"\nListen to {out}/ — test_05 is the tone minimal-pair check.")


if __name__ == "__main__":
    main()
