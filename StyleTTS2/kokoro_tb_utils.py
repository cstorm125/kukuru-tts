"""Shared TensorBoard inference helpers for Kokoro-faithful audio previews.

Used by both train_first.py (Stage 1) and train_second.py (Stage 2) to generate
TensorBoard audio that matches actual Kokoro inference output — predicted duration,
F0, and energy from a voicepack, not ground-truth reconstruction.
"""

import logging
import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def _unwrap(m):
    """Unwrap DDP/DataParallel so submodule attributes are reachable."""
    return getattr(m, "module", m)

# Thai test sentences — same set as scripts/baseline_thai_samples.py
TEST_SENTENCES = [
    "สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา",
    "ยอดเงินคงเหลือสามพันห้าร้อยบาท",
    "ใครขายไข่ไก่ ใกล้ๆ บ้านเรา",
    "กรุณากดหมายเลขหนึ่งเพื่อติดต่อพนักงาน",
    "วันนี้อากาศเป็นอย่างไรบ้าง",
]


def prepare_test_tokens(text_cleaner):
    """Convert Thai test sentences to token ID lists via FastThaiG2P.

    Returns a list of (display_text, token_ids) tuples. Sentences that
    fail G2P or produce sequences longer than 510 tokens are skipped.
    """
    try:
        from fastthaig2p import G2P, ipa_to_kokoro

        g2p = G2P()
    except Exception as e:
        logger.warning(f"Could not load Thai G2P for TensorBoard inference: {e}")
        return []

    result = []
    for text in TEST_SENTENCES:
        try:
            phonemes = ipa_to_kokoro(g2p.convert(text))
            token_ids = text_cleaner(phonemes)
            if not token_ids or len(token_ids) > 510:
                logger.warning(
                    f"Skipping test sentence (token length {len(token_ids)}): {text[:40]}"
                )
                continue
            result.append((text, token_ids))
        except Exception as e:
            logger.warning(f'G2P failed for test sentence "{text[:40]}": {e}')
    return result


def extract_voicepack(model, root_path, device, n_samples=200):
    """Extract a mini voicepack from audio files — mirrors extract_voicepack.py.

    Randomly samples up to n_samples WAV files from root_path, computes
    mel spectrograms with the same params as meldataset.py, runs them through
    model.style_encoder and model.predictor_encoder, and averages the results.

    Returns:
        voicepack: torch.FloatTensor [256] — combined acoustic+prosodic style
        acoustic_norm: float — output norm of style_encoder (health check)
        prosodic_norm: float — output norm of predictor_encoder (health check)
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        n_mels=80,
    ).to(device)
    mel_mean, mel_std = -4, 4

    wav_files = list(Path(root_path).rglob("*.wav"))
    if not wav_files:
        logger.warning(f"extract_voicepack: no WAV files found in {root_path}")
        return None, 0.0, 0.0

    random.shuffle(wav_files)
    wav_files = wav_files[:n_samples]

    acoustic_styles = []
    prosodic_styles = []

    with torch.no_grad():
        for wav_path in wav_files:
            try:
                data, file_sr = sf.read(str(wav_path), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                waveform = torch.from_numpy(data).unsqueeze(0)
                if file_sr != 24000:
                    waveform = torchaudio.functional.resample(waveform, file_sr, 24000)
                waveform = waveform.to(device)
                mel = mel_transform(waveform)
                mel = (torch.log(1e-5 + mel) - mel_mean) / mel_std
                if mel.shape[-1] < 80:
                    continue
                mel_input = mel.unsqueeze(1)  # [1, 1, 80, T]
                acoustic_styles.append(_unwrap(model.style_encoder)(mel_input).cpu())
                prosodic_styles.append(
                    _unwrap(model.predictor_encoder)(mel_input).cpu()
                )
            except Exception as e:
                logger.warning(f"extract_voicepack: skipping {wav_path.name}: {e}")

    if not acoustic_styles:
        logger.warning("extract_voicepack: no valid audio files processed")
        return None, 0.0, 0.0

    avg_acoustic = torch.cat(acoustic_styles, dim=0).mean(dim=0)  # [128]
    avg_prosodic = torch.cat(prosodic_styles, dim=0).mean(dim=0)  # [128]
    voicepack = torch.cat([avg_acoustic, avg_prosodic], dim=0)  # [256]

    acoustic_norm = avg_acoustic.norm().item()
    prosodic_norm = avg_prosodic.norm().item()

    return voicepack.to(device), acoustic_norm, prosodic_norm


def run_kokoro_inference(model, test_tokens, voicepack, device, text_cleaner):
    """Run Kokoro-faithful inference for all test sentences.

    Mirrors KModel.forward_with_tokens exactly:
      - Predict duration from the predictor
      - Build alignment from predicted duration
      - Predict F0 and energy
      - Decode audio using voicepack acoustic style

    Returns a list of (display_text, audio_numpy) tuples.
    """
    if voicepack is None or not test_tokens:
        return []

    # voicepack [256]: first 128 = acoustic (decoder), last 128 = prosodic (predictor)
    ref_acoustic = voicepack[:128].unsqueeze(0)  # [1, 128]
    ref_prosodic = voicepack[128:].unsqueeze(0)  # [1, 128]

    bert = _unwrap(model.bert)
    bert_encoder = _unwrap(model.bert_encoder)
    predictor = _unwrap(model.predictor)
    text_encoder = _unwrap(model.text_encoder)
    decoder = _unwrap(model.decoder)

    results = []
    with torch.no_grad():
        for text, token_ids in test_tokens:
            try:
                # Build input token tensor with BOS/EOS (token 0)
                input_ids = torch.LongTensor([[0, *token_ids, 0]]).to(device)
                input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(device)
                text_mask = torch.gt(
                    torch.arange(input_lengths.max())
                    .unsqueeze(0)
                    .expand(1, -1)
                    .type_as(input_lengths)
                    + 1,
                    input_lengths.unsqueeze(1),
                ).to(device)

                # BERT + encoder
                bert_dur = bert(input_ids, attention_mask=(~text_mask).int())
                d_en = bert_encoder(bert_dur).transpose(-1, -2)

                # Predict duration
                s_prosodic = ref_prosodic  # [1, 128]
                d = predictor.text_encoder(
                    d_en, s_prosodic, input_lengths, text_mask
                )
                x, _ = predictor.lstm(d)
                duration = predictor.duration_proj(x)
                duration = torch.sigmoid(duration).sum(axis=-1)
                pred_dur = torch.round(duration.squeeze()).clamp(min=1).long()
                if pred_dur.dim() == 0:
                    pred_dur = pred_dur.unsqueeze(0)

                # Build alignment matrix from predicted duration
                n_tokens = input_ids.shape[1]
                total_frames = int(pred_dur.sum().item())
                if total_frames == 0:
                    continue
                pred_aln_trg = torch.zeros(n_tokens, total_frames).to(device)
                c_frame = 0
                for i in range(n_tokens):
                    dur_i = int(pred_dur[i].item())
                    pred_aln_trg[i, c_frame : c_frame + dur_i] = 1
                    c_frame += dur_i
                pred_aln_trg = pred_aln_trg.unsqueeze(0)  # [1, n_tokens, total_frames]

                # Predict F0 and energy
                en = d.transpose(-1, -2) @ pred_aln_trg
                F0_pred, N_pred = predictor.F0Ntrain(en, s_prosodic)

                # Text encoder + decode
                t_en = text_encoder(input_ids, input_lengths, text_mask)
                asr = t_en @ pred_aln_trg
                audio = decoder(asr, F0_pred, N_pred, ref_acoustic)
                results.append((text, audio.squeeze().cpu().numpy()))
            except Exception as e:
                logger.warning(f'Inference failed for "{text[:40]}": {e}')

    return results
