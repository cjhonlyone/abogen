"""VibeVoice TTS adapter for abogen.

Uses the vendored vvembed package (``abogen.libs.vibevoice``). VibeVoice is
opt-in: install the ``vibevoice`` extra and place the model weights under the
expected cache directory. See ``abogen/libs/vibevoice/SOURCE.md`` for the
upstream source and licensing.

The adapter exposes a small set of "virtual" voice ids (``V1``..``V4``)
backed by procedurally generated reference samples (sine + formants), so no
audio assets need to be shipped or uploaded by the user. This mirrors the
synthetic voice fallback in the upstream ComfyUI nodes.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


DEFAULT_VIBEVOICE_VOICES = ("V1", "V2", "V3", "V4")

# Built-in models supported by the upstream wrapper. Users place the actual
# weights inside ``<cache>/vibevoice/<model_name>/``. The first existing
# match is used at runtime.
DEFAULT_VIBEVOICE_MODELS = (
    "VibeVoice-1.5B",
    "VibeVoice-Large",
    "VibeVoice-Large-Q8",
    "VibeVoice-Large-Q4",
)

VIBEVOICE_SAMPLE_RATE = 24000


@dataclass
class VibeVoiceSegment:
    graphemes: str
    audio: np.ndarray


# ---------------------------------------------------------------------------
# Helpers (mostly mirrored from tts_supertonic.py for consistency)
# ---------------------------------------------------------------------------


def _ensure_float32_mono(wav: Any) -> np.ndarray:
    arr = np.asarray(wav, dtype="float32")
    if arr.ndim == 3:
        # (batch, channels, samples) -> first channel of first batch
        arr = arr[0, 0]
    elif arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            arr = arr.reshape(-1)
        else:
            arr = arr[:, 0]
    return arr.reshape(-1)


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio
    ratio = dst_rate / float(src_rate)
    new_len = int(round(audio.size * ratio))
    if new_len <= 1:
        return np.zeros(0, dtype="float32")
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype("float32", copy=False)


def _split_text(text: str, *, split_pattern: Optional[str], max_words: int) -> List[str]:
    """Split text into chunks of at most ``max_words`` words.

    VibeVoice limits generation by token count rather than characters; the
    upstream wrapper splits on word count (default 250).
    """

    stripped = (text or "").strip()
    if not stripped:
        return []
    if split_pattern:
        try:
            parts = [p.strip() for p in re.split(split_pattern, stripped) if p.strip()]
        except re.error:
            parts = [stripped]
    else:
        parts = [stripped]

    out: List[str] = []
    for part in parts:
        words = part.split()
        if len(words) <= max_words:
            out.append(part)
            continue
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i : i + max_words]).strip()
            if chunk:
                out.append(chunk)
    return out


def _synthetic_voice(speaker_idx: int, sample_rate: int = VIBEVOICE_SAMPLE_RATE) -> np.ndarray:
    """Procedurally generated 1 s reference clip per speaker index.

    Mirrors ``BaseVibeVoiceNode._create_synthetic_voice_sample`` from the
    upstream ComfyUI nodes so behaviour matches the wrapper that the vvembed
    code was designed for.
    """
    duration = 1.0
    samples = int(sample_rate * duration)
    t = np.linspace(0.0, duration, samples, endpoint=False)

    base_frequencies = (120.0, 180.0, 140.0, 200.0)
    base_freq = base_frequencies[speaker_idx % len(base_frequencies)]
    formant1 = 800.0 + speaker_idx * 100.0
    formant2 = 1200.0 + speaker_idx * 150.0

    voice = (
        0.6 * np.sin(2 * np.pi * base_freq * t)
        + 0.25 * np.sin(2 * np.pi * base_freq * 2 * t)
        + 0.15 * np.sin(2 * np.pi * base_freq * 3 * t)
        + 0.1 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t * 2)
        + 0.05 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t * 3)
        + 0.02 * np.random.default_rng(speaker_idx).normal(0, 1, samples)
    )
    vibrato_freq = 4.0 + speaker_idx * 0.3
    envelope = np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * vibrato_freq * t))
    voice = voice * envelope * 0.08
    return voice.astype(np.float32)


def _voice_to_index(voice: str) -> int:
    raw = (voice or "").strip().upper()
    if raw in DEFAULT_VIBEVOICE_VOICES:
        return DEFAULT_VIBEVOICE_VOICES.index(raw)
    return 0


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------


def _candidate_model_dirs(model_name: str) -> List[Path]:
    """Where we look for VibeVoice weights on disk.

    Searches (in order):
      1. ``$ABOGEN_VIBEVOICE_DIR/<model_name>``
      2. ``<user cache>/vibevoice/<model_name>``
      3. ``<user cache>/vibevoice/models--*--<model_name>/snapshots/<hash>``
      4. Standard HuggingFace hub cache: ``~/.cache/huggingface/hub/models--*--<model_name>/snapshots/<hash>``
         (also respects ``HF_HOME`` / ``HF_HUB_CACHE`` env vars)
    """
    from abogen.utils import get_user_cache_path

    cands: List[Path] = []
    env = os.environ.get("ABOGEN_VIBEVOICE_DIR")
    if env:
        cands.append(Path(env).expanduser() / model_name)

    try:
        base = Path(get_user_cache_path()) / "vibevoice"
    except Exception:
        base = Path.home() / ".cache" / "abogen" / "vibevoice"
    cands.append(base / model_name)

    # HuggingFace cache layout under abogen cache dir.
    if base.exists():
        for entry in sorted(base.glob(f"models--*--{model_name}/snapshots/*")):
            if entry.is_dir():
                cands.append(entry)

    # Standard HuggingFace hub cache (respects HF_HOME / HF_HUB_CACHE env vars).
    hf_hub_cache = (
        Path(os.environ["HF_HUB_CACHE"])
        if "HF_HUB_CACHE" in os.environ
        else Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
    if hf_hub_cache.exists():
        for entry in sorted(hf_hub_cache.glob(f"models--*--{model_name}/snapshots/*")):
            if entry.is_dir():
                cands.append(entry)

    return cands


_HF_REPO_ID = "microsoft/VibeVoice-1.5B"
_HF_TOKENIZER_REPO_ID = "Qwen/Qwen2.5-1.5B"


def _hf_download_model(model_name: str) -> Path:
    """Download model from HuggingFace Hub into the abogen vibevoice cache."""
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    try:
        base = Path(__import__("abogen.utils", fromlist=["get_user_cache_path"]).get_user_cache_path()) / "vibevoice"
    except Exception:
        base = Path.home() / ".cache" / "abogen" / "vibevoice"
    local_dir = base / model_name
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading VibeVoice model '%s' from %s ...", model_name, _HF_REPO_ID)
    snapshot_download(repo_id=_HF_REPO_ID, local_dir=str(local_dir))
    return local_dir


def _hf_download_tokenizer(dest_dir: Path) -> Path:
    """Download Qwen2.5-1.5B tokenizer files from HuggingFace Hub."""
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading VibeVoice tokenizer from %s ...", _HF_TOKENIZER_REPO_ID)
    snapshot_download(
        repo_id=_HF_TOKENIZER_REPO_ID,
        local_dir=str(dest_dir),
        ignore_patterns=["*.bin", "*.safetensors", "*.pt"],  # tokenizer files only
    )
    return dest_dir


def _resolve_model_dir(model_name: str) -> Path:
    for cand in _candidate_model_dirs(model_name):
        if cand.exists() and any(cand.iterdir()):
            return cand
    # Not found locally — auto-download from HuggingFace.
    logger.info("VibeVoice model '%s' not found locally, attempting HuggingFace download.", model_name)
    try:
        return _hf_download_model(model_name)
    except Exception as exc:
        searched = "\n  ".join(str(c) for c in _candidate_model_dirs(model_name))
        raise RuntimeError(
            f"VibeVoice model '{model_name}' not found and auto-download failed: {exc}\n"
            f"Searched:\n  {searched}\n"
            f"You can manually download from https://huggingface.co/{_HF_REPO_ID}"
        ) from exc


def scan_vibevoice_models() -> List[str]:
    """Return model names that are actually present on disk.

    Checks every known model name from *DEFAULT_VIBEVOICE_MODELS* plus any
    unrecognised directories discovered inside the search roots.  Returns a
    deduplicated, sorted list so the UI can offer a meaningful dropdown.
    """
    found: dict[str, bool] = {}  # name → already_added guard

    # 1. Known model names whose candidate dirs exist and are non-empty.
    for name in DEFAULT_VIBEVOICE_MODELS:
        for cand in _candidate_model_dirs(name):
            if cand.exists() and any(cand.iterdir()):
                found[name] = True
                break

    # 2. Any extra model directories inside the abogen vibevoice cache root
    #    that don't match a known name (e.g. user placed a custom model there).
    try:
        from abogen.utils import get_user_cache_path
        base = Path(get_user_cache_path()) / "vibevoice"
    except Exception:
        base = Path.home() / ".cache" / "abogen" / "vibevoice"
    if base.exists():
        for entry in base.iterdir():
            if entry.is_dir() and any(entry.iterdir()):
                found[entry.name] = True

    # 3. Standard HuggingFace hub cache — extract user-friendly names.
    hf_hub_cache = (
        Path(os.environ["HF_HUB_CACHE"])
        if "HF_HUB_CACHE" in os.environ
        else Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
    if hf_hub_cache.exists():
        for model_dir in hf_hub_cache.glob("models--*--VibeVoice*/snapshots/*"):
            if model_dir.is_dir() and any(model_dir.iterdir()):
                # Extract the model name from "models--<org>--<name>" pattern.
                repo_part = model_dir.parent.parent.name  # e.g. models--aoi-ot--VibeVoice-Large
                name = repo_part.split("--", 2)[-1] if "--" in repo_part else repo_part
                found[name] = True

    return sorted(found.keys())


def _resolve_tokenizer_dir(model_dir: Path) -> Path:
    base = model_dir.parent
    for cand in (
        base / "tokenizer",
        base / "models--Qwen--Qwen2.5-1.5B",
        model_dir,  # some HF snapshots include tokenizer files in the model dir
    ):
        if cand.exists() and any(cand.glob("tokenizer*.json")):
            return cand
    # Not found — auto-download tokenizer only.
    tokenizer_dir = base / "tokenizer"
    logger.info("VibeVoice tokenizer not found locally, attempting HuggingFace download.")
    try:
        return _hf_download_tokenizer(tokenizer_dir)
    except Exception as exc:
        raise RuntimeError(
            f"VibeVoice tokenizer (Qwen2.5-1.5B) not found and auto-download failed: {exc}\n"
            f"You can manually download from https://huggingface.co/{_HF_TOKENIZER_REPO_ID} "
            f"and place tokenizer files under {tokenizer_dir}."
        ) from exc


def _select_torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class VibeVoicePipeline:
    """Kokoro-compatible pipeline backed by vendored VibeVoice."""

    def __init__(
        self,
        *,
        sample_rate: int = VIBEVOICE_SAMPLE_RATE,
        model_name: str = "VibeVoice-1.5B",
        diffusion_steps: int = 20,
        cfg_scale: float = 1.3,
        attention: str = "auto",
        quantize: str = "none",
        max_words_per_chunk: int = 250,
        device: Optional[str] = None,
        use_sampling: bool = False,
        temperature: float = 0.95,
        top_p: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.model_name = model_name
        self.diffusion_steps = max(5, min(100, int(diffusion_steps)))
        self.cfg_scale = max(1.0, min(2.0, float(cfg_scale)))
        self.attention = attention or "auto"
        self.quantize = (quantize or "none").lower()
        self.max_words_per_chunk = max(50, int(max_words_per_chunk))
        self.device = device or _select_torch_device()
        self.use_sampling = bool(use_sampling)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed)

        self._model = None
        self._processor = None

    # -- Lazy initialisation -------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "VibeVoice requires the 'vibevoice' extra. "
                "Install with: pip install \"abogen[vibevoice]\""
            ) from exc

        model_dir = _resolve_model_dir(self.model_name)
        tokenizer_dir = _resolve_tokenizer_dir(model_dir)

        from abogen.libs.vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from abogen.libs.vibevoice.processor.vibevoice_processor import (
            VibeVoiceProcessor,
        )

        logger.info(
            "Loading VibeVoice model from %s (device=%s, attn=%s, quant=%s)",
            model_dir, self.device, self.attention, self.quantize,
        )

        load_kwargs: dict[str, Any] = {}
        if self.attention not in ("auto", "", None):
            load_kwargs["attn_implementation"] = self.attention

        # Optional bitsandbytes quantization.
        if self.quantize in {"4bit", "8bit"} and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                if self.quantize == "4bit":
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype="bfloat16",
                    )
                else:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as exc:
                logger.warning("Could not configure %s quantization: %s", self.quantize, exc)

        try:
            import torch

            dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
            self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                str(model_dir), torch_dtype=dtype, **load_kwargs
            )
            if "quantization_config" not in load_kwargs:
                self._model = self._model.to(self.device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load VibeVoice model from {model_dir}: {exc}"
            ) from exc

        try:
            self._processor = VibeVoiceProcessor.from_pretrained(str(tokenizer_dir))
        except Exception:
            # Fall back to looking inside the model dir itself (legacy layout).
            self._processor = VibeVoiceProcessor.from_pretrained(str(model_dir))

    # -- Public API ----------------------------------------------------------

    def __call__(
        self,
        text: str,
        *,
        voice: str,
        speed: float = 1.0,
        split_pattern: Optional[str] = None,
        diffusion_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
    ) -> Iterator[VibeVoiceSegment]:
        self._ensure_loaded()

        import torch

        speaker_idx = _voice_to_index(voice)
        reference = _synthetic_voice(speaker_idx, self.sample_rate)
        steps = int(diffusion_steps if diffusion_steps is not None else self.diffusion_steps)
        steps = max(5, min(100, steps))
        cfg = float(cfg_scale if cfg_scale is not None else self.cfg_scale)
        cfg = max(1.0, min(2.0, cfg))

        chunks = _split_text(text, split_pattern=split_pattern, max_words=self.max_words_per_chunk)
        for chunk in chunks:
            formatted = f"Speaker 0: {chunk}"
            inputs = self._processor(
                [formatted],
                voice_samples=[[reference]],
                return_tensors="pt",
                return_attention_mask=True,
            )
            device = next(self._model.parameters()).device
            inputs = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
            }

            # Inject diffusion steps if the model exposes the attribute.
            if hasattr(self._model, "set_ddpm_inference_steps"):
                try:
                    self._model.set_ddpm_inference_steps(num_steps=steps)
                except Exception:
                    pass

            with torch.no_grad():
                if self.use_sampling:
                    output = self._model.generate(
                        **inputs,
                        tokenizer=self._processor.tokenizer,
                        cfg_scale=cfg,
                        max_new_tokens=None,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                else:
                    output = self._model.generate(
                        **inputs,
                        tokenizer=self._processor.tokenizer,
                        cfg_scale=cfg,
                        max_new_tokens=None,
                        do_sample=False,
                    )

            speech = getattr(output, "speech_outputs", None)
            if not speech:
                logger.error(
                    "VibeVoice produced no audio for chunk (%d chars); skipping.",
                    len(chunk),
                )
                continue

            if isinstance(speech, list) and len(speech) > 0:
                audio_tensor = torch.cat([t for t in speech if t is not None], dim=-1)
            else:
                audio_tensor = speech

            audio_np = _ensure_float32_mono(audio_tensor.detach().cpu().float().numpy())
            if self.sample_rate != VIBEVOICE_SAMPLE_RATE:
                audio_np = _resample_linear(audio_np, VIBEVOICE_SAMPLE_RATE, self.sample_rate)

            # `speed` is honored by simple linear resampling (changes pitch
            # slightly but matches the upstream wrapper's quick path).
            if speed and abs(speed - 1.0) > 1e-3:
                target_len = max(1, int(round(audio_np.size / float(speed))))
                if target_len != audio_np.size:
                    audio_np = _resample_linear(audio_np, audio_np.size, target_len)

            yield VibeVoiceSegment(graphemes=chunk, audio=audio_np)
