"""Optional Weights & Biases logging for Stage 1/2 training.

Piggybacks on the existing TensorBoard logging via sync_tensorboard: every
writer.add_scalar/add_audio/add_figure call is mirrored to wandb, so the
training scripts need no per-metric changes.

Runs online when credentials exist (WANDB_API_KEY or `wandb login`),
otherwise falls back to offline mode — sync later with `wandb sync`.
Disable entirely with WANDB_DISABLED=true.
"""

import logging
import netrc
import os

logger = logging.getLogger(__name__)


def _has_wandb_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        hosts = netrc.netrc(os.path.expanduser("~/.netrc")).hosts
        return any("wandb" in h for h in hosts)
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def init_wandb(config: dict, name: str):
    """Start a wandb run mirroring TensorBoard. Call BEFORE creating SummaryWriter.

    Returns the wandb run, or None if wandb is unavailable/disabled.
    """
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true"):
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping wandb logging")
        return None

    mode = os.environ.get("WANDB_MODE") or (
        "online" if _has_wandb_credentials() else "offline"
    )
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "kokoro-thai"),
        name=name,
        config=config,
        mode=mode,
        sync_tensorboard=True,
    )
    logger.info(f"wandb run '{name}' started in {mode} mode: {run.dir}")
    if mode == "offline":
        logger.info("No wandb credentials found — logging offline. "
                    "Run `wandb login` then `wandb sync <run dir>` to upload.")
    return run


def log_audio(tag: str, audio, epoch: int, sample_rate: int, caption: str = None):
    """Log audio explicitly to wandb — sync_tensorboard drops add_audio events.

    No explicit step: sync_tensorboard pins wandb's step counter to TB's
    global step, so a small epoch-valued step would be silently discarded.
    The epoch goes into the caption instead. No-op when wandb is
    disabled or not initialized.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    try:
        text = f"epoch {epoch}" + (f" — {caption}" if caption else "")
        wandb.log({tag: wandb.Audio(audio, sample_rate=sample_rate, caption=text)})
    except Exception as e:
        logger.warning(f"wandb audio logging failed for {tag}: {e}")
