#!/usr/bin/env python3
"""Trained Thai checkpoints → deployable artifacts, one command.

    python scripts/package_thai.py \
        --stage2-ckpt StyleTTS2/logs/thai/epoch_2nd_00008.pth \
        --stage1-ckpt StyleTTS2/logs/thai/first_stage.pth \
        --audio-dir training/audio \
        --config training/config.json \
        --output-dir dist/thai_v2

Produces in output-dir:
    kokoro_thai.pth      inference-format weights (KModel-loadable)
    kokoro_thai.onnx     ONNX export (CPU serving; validated vs torch)
    thai_som.pt          voicepack [510,1,256] (torch)
    thai_som.npy         voicepack (numpy — torch-free ONNX serving)
    config.json          model config (vocab)
    samples/*.wav        sound-check sentences

The voicepack mixes the STAGE 1 style encoder with the STAGE 2 prosody
encoder — stage 1's style encoder saw the adversarial-free acoustics the
decoder was trained against; using stage 2's for both degrades timbre.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-ckpt", required=True, type=Path)
    ap.add_argument("--stage1-ckpt", required=True, type=Path)
    ap.add_argument("--audio-dir", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--skip-onnx", action="store_true")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # 1. inference weights
    sys.path.insert(0, str(REPO / "scripts"))
    from test_inference import convert_checkpoint

    model_pth = out / "kokoro_thai.pth"
    convert_checkpoint(str(args.stage2_ckpt), str(model_pth))

    # 2. voicepack (stage-1 style encoder + stage-2 prosody)
    vp = out / "thai_som.pt"
    subprocess.run(
        [py, str(REPO / "scripts" / "extract_voicepack.py"),
         "--model", str(args.stage2_ckpt),
         "--style-encoder-model", str(args.stage1_ckpt),
         "--audio-dir", str(args.audio_dir),
         "--output", str(vp), "--device", "cpu"],
        check=True,
    )
    import numpy as np
    import torch

    np.save(out / "thai_som.npy",
            torch.load(vp, map_location="cpu", weights_only=True).numpy())

    # 3. config
    shutil.copy(args.config, out / "config.json")

    # 4. ONNX (needs fastthaig2p's exporter on PYTHONPATH or installed)
    if not args.skip_onnx:
        exporter = None
        for cand in (Path.home() / "FastThaiG2P" / "scripts" / "export_onnx.py",):
            if cand.exists():
                exporter = cand
        if exporter is None:
            print("export_onnx.py not found — skipping ONNX (pass --skip-onnx to silence)")
        else:
            subprocess.run(
                [py, str(exporter),
                 "--model", str(model_pth), "--config", str(out / "config.json"),
                 "--output", str(out / "kokoro_thai.onnx"),
                 "--voicepack", str(vp), "--validate"],
                check=True,
            )

    # 5. sound-check samples
    samples = out / "samples"
    samples.mkdir(exist_ok=True)
    subprocess.run(
        [py, str(REPO / "scripts" / "baseline_thai_samples.py"),
         "--model", str(model_pth), "--voicepack", str(vp),
         "--config", str(out / "config.json"), "--output-dir", str(samples)],
        check=True,
    )
    print(f"\nAll artifacts in {out}/ — listen to samples/ before shipping.")


if __name__ == "__main__":
    main()
