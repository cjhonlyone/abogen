"""Tests for the TTS provider registry and voice-profile normalization
for the newly added VibeVoice provider and Supertonic 3 upgrade."""

import pytest

from abogen import tts_providers
from abogen.voice_profiles import normalize_profile_entry


def test_known_providers_include_vibevoice():
    names = tts_providers.known_providers()
    assert {"kokoro", "supertonic", "vibevoice"}.issubset(names)


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("M1", "supertonic"),
        ("F3", "supertonic"),
        ("V1", "vibevoice"),
        ("v2", "vibevoice"),
        ("af_heart*1", "kokoro"),
        ("af_heart+bf_emma", "kokoro"),
        ("", "kokoro"),
        ("unknown_voice", "kokoro"),
    ],
)
def test_infer_provider_from_spec(spec, expected):
    assert tts_providers.infer_provider_from_spec(spec) == expected


def test_normalize_discrete_voice_supertonic():
    assert tts_providers.normalize_discrete_voice("supertonic", "m1", "M1") == "M1"
    assert tts_providers.normalize_discrete_voice("supertonic", "BAD", "M1") == "M1"


def test_normalize_discrete_voice_vibevoice():
    assert tts_providers.normalize_discrete_voice("vibevoice", "v2", "V1") == "V2"
    assert tts_providers.normalize_discrete_voice("vibevoice", "X9", "V1") == "V1"


def test_voice_profile_normalize_vibevoice():
    profile = normalize_profile_entry(
        {
            "provider": "vibevoice",
            "voice": "v3",
            "model": "VibeVoice-1.5B",
            "diffusion_steps": 30,
            "cfg_scale": 1.5,
        }
    )
    assert profile["provider"] == "vibevoice"
    assert profile["voice"] == "V3"
    assert profile["model"] == "VibeVoice-1.5B"
    assert profile["diffusion_steps"] == 30
    assert profile["cfg_scale"] == 1.5


def test_voice_profile_normalize_vibevoice_defaults_clamp():
    profile = normalize_profile_entry(
        {"provider": "vibevoice", "voice": "weird", "diffusion_steps": 999, "cfg_scale": 5.0}
    )
    assert profile["voice"] == "V1"
    assert profile["diffusion_steps"] == 100
    assert profile["cfg_scale"] == 2.0


def test_voice_profile_normalize_supertonic_still_works():
    profile = normalize_profile_entry(
        {"provider": "supertonic", "voice": "F3", "total_steps": 8, "speed": 1.1}
    )
    assert profile["provider"] == "supertonic"
    assert profile["voice"] == "F3"
    assert profile["total_steps"] == 8
    assert profile["speed"] == pytest.approx(1.1)
