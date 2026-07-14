#!/usr/bin/env python3
"""Thai (text, wav) pairs → StyleTTS2 training lists.

Input: a metadata file (csv/jsonl) mapping wav filenames to Thai text,
plus the wav directory. Output: train_list.txt / val_list.txt /
OOD_texts.txt in the 5-tone Kokoro phoneme format, hard-validated
against the model vocab.

    python scripts/prepare_thai_dataset.py \
        --metadata dataset/metadata.csv \        # wav|text  (or .jsonl with
        --audio-dir dataset/audio \              #  {"file":..., "text":...})
        --config training/config.json \
        --output-dir training \
        [--val-ratio 0.05] [--ood-file extra_texts.txt] [--seed 1234]

Requires: pip install fastthaig2p (>= the 5-tone ipa_to_kokoro).
Every phoneme character is checked against config.json's vocab — any
out-of-vocab character fails the run listing the offending rows, because
one bad symbol silently corrupts training.
"""

import argparse
import json
import random
import sys
from pathlib import Path


def read_metadata(path: Path):
    rows = []
    if path.suffix == ".jsonl":
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            rows.append((r["file"], r["text"]))
    else:  # csv/psv: filename|text  (or filename,text with no header)
        for line in path.open(encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            sep = "|" if "|" in line else ","
            fn, text = line.split(sep, 1)
            rows.append((fn.strip(), text.strip()))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--audio-dir", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path, help="config.json with vocab")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--ood-file", type=Path, help="extra texts for SLM adversarial OOD")
    ap.add_argument("--speaker", default="0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-phonemes", type=int, default=10,
                    help="skip rows shorter than this (min_length guard)")
    args = ap.parse_args()

    from fastthaig2p import G2P, ipa_to_kokoro, normalize

    g2p = G2P()
    vocab = set(json.loads(args.config.read_text())["vocab"])

    rows = read_metadata(args.metadata)
    print(f"{len(rows)} metadata rows")

    entries, errors, skipped = [], [], 0
    for fn, text in rows:
        wav = args.audio_dir / fn
        if not wav.exists():
            errors.append(f"missing wav: {fn}")
            continue
        phonemes = ipa_to_kokoro(g2p.convert(normalize(text)))
        if len(phonemes) < args.min_phonemes:
            skipped += 1
            continue
        bad = {c for c in phonemes if c not in vocab}
        if bad:
            errors.append(f"out-of-vocab {sorted(bad)} in {fn}: {text[:50]}")
            continue
        entries.append(f"{fn}|{phonemes}|{args.speaker}")

    if errors:
        print(f"\nFAILED — {len(errors)} problem rows:", file=sys.stderr)
        for e in errors[:20]:
            print("  " + e, file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        sys.exit(1)

    random.Random(args.seed).shuffle(entries)
    n_val = max(1, int(len(entries) * args.val_ratio))
    val, train = entries[:n_val], entries[n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_list.txt").write_text("\n".join(train) + "\n")
    (args.output_dir / "val_list.txt").write_text("\n".join(val) + "\n")

    # OOD list: phonemes|speaker (no filename) for the SLM adversarial stage
    ood_src = args.ood_file
    ood_lines = []
    if ood_src and ood_src.exists():
        for line in ood_src.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            ph = ipa_to_kokoro(g2p.convert(normalize(line)))
            if ph and all(c in vocab for c in ph):
                ood_lines.append(f"{ph}|{args.speaker}")
    else:
        # fall back to reusing training phonemes — fine for small runs
        ood_lines = [e.split("|")[1] + f"|{args.speaker}" for e in train]
    (args.output_dir / "OOD_texts.txt").write_text("\n".join(ood_lines) + "\n")

    print(f"train {len(train)} | val {len(val)} | OOD {len(ood_lines)} | "
          f"skipped-short {skipped}")
    print(f"lists written to {args.output_dir}/")


if __name__ == "__main__":
    main()
