"""Central registry of TTS providers supported by abogen.

The webui pipeline historically hard-coded the set ``{"kokoro", "supertonic"}``
in ~20 places. As we add more providers (e.g. VibeVoice), keeping a single
source of truth here avoids drift and makes it easy to introduce additional
backends in the future.

Pipelines registered here must implement a callable matching the Kokoro
``KPipeline.__call__`` contract used by ``conversion_runner.synthesize_text``::

    pipeline(text, *, voice, speed, split_pattern=None, **provider_kwargs)
        -> Iterable[Segment]

where each ``Segment`` exposes ``.graphemes`` and ``.audio`` (a float32 mono
numpy array at ``SAMPLE_RATE``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Tuple


SAMPLE_RATE = 24000


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    is_discrete: bool
    default_voice: str
    voices: Tuple[str, ...] = ()
    settings_keys: Tuple[str, ...] = ()


_KOKORO = ProviderSpec(
    name="kokoro",
    label="Kokoro",
    is_discrete=False,
    default_voice="af_heart",
    voices=(),  # Voices come from VOICES_INTERNAL; mixes use formulas.
    settings_keys=(),
)

_SUPERTONIC = ProviderSpec(
    name="supertonic",
    label="Supertonic",
    is_discrete=True,
    default_voice="M1",
    voices=("M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"),
    settings_keys=("supertonic_total_steps", "supertonic_speed", "supertonic_lang"),
)

# VibeVoice voices are synthesized procedurally from a small set of speaker
# indices (sine + formants). They give the user multiple distinct "voices"
# without bundling reference audio. See ``tts_vibevoice._synthetic_voice``.
_VIBEVOICE = ProviderSpec(
    name="vibevoice",
    label="VibeVoice",
    is_discrete=True,
    default_voice="V1",
    voices=("V1", "V2", "V3", "V4"),
    settings_keys=(
        "vibevoice_model",
        "vibevoice_diffusion_steps",
        "vibevoice_cfg_scale",
        "vibevoice_attention",
        "vibevoice_quantize",
    ),
)


REGISTRY: dict[str, ProviderSpec] = {
    _KOKORO.name: _KOKORO,
    _SUPERTONIC.name: _SUPERTONIC,
    _VIBEVOICE.name: _VIBEVOICE,
}


def known_providers() -> set[str]:
    return set(REGISTRY)


def get_spec(name: str) -> ProviderSpec:
    key = (name or "").strip().lower()
    if key in REGISTRY:
        return REGISTRY[key]
    return _KOKORO


def discrete_providers() -> Iterable[ProviderSpec]:
    return [spec for spec in REGISTRY.values() if spec.is_discrete]


def infer_provider_from_spec(value: Any, fallback: str = "kokoro") -> str:
    """Best-effort identification of which provider owns a voice spec string.

    - Kokoro formulas contain '*' or '+'.
    - Discrete providers (supertonic, vibevoice, ...) own their voice id sets.
    - Otherwise fall back to ``fallback``.
    """

    raw = str(value or "").strip()
    if not raw:
        return fallback if fallback in REGISTRY else "kokoro"
    if "*" in raw or "+" in raw:
        return "kokoro"
    upper = raw.upper()
    for spec in discrete_providers():
        if upper in spec.voices:
            return spec.name
    return fallback if fallback in REGISTRY else "kokoro"


def normalize_discrete_voice(provider: str, value: Any, fallback: Any = None) -> str:
    """Coerce ``value`` into a valid discrete voice id for ``provider``.

    Falls back to ``fallback`` (also normalized) and then to the provider's
    ``default_voice``. Kokoro spec strings (containing '*' or '+') are rejected.
    """

    spec = get_spec(provider)
    if not spec.is_discrete:
        return str(value or "")

    raw = str(value or "").strip()
    if not raw or "*" in raw or "+" in raw:
        raw = str(fallback or "").strip()
    if not raw or "*" in raw or "+" in raw:
        return spec.default_voice

    upper = raw.upper()
    if upper in spec.voices:
        return upper

    fb = str(fallback or "").strip().upper() if fallback else ""
    if fb in spec.voices:
        return fb
    return spec.default_voice
