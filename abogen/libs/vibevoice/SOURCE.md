# Vendored VibeVoice (vvembed)

This directory is a verbatim copy of `vvembed/` from
<https://github.com/Enemyx-net/VibeVoice-ComfyUI> at commit
`b0a8efc23c9d902d7ebd8fe3f79515a23d343bc2`.

That upstream is itself an embedded copy of Microsoft's VibeVoice
(`https://github.com/microsoft/VibeVoice`, MIT-licensed). See
`LICENSE` in this directory for the original copyright and license.

No functional modifications have been made; imports are already relative
in the upstream copy.

Heavy runtime dependencies (transformers, diffusers, accelerate, torch,
torchaudio, librosa, soundfile, peft, bitsandbytes, av, aiortc, …) are
declared as the optional extra `[vibevoice]` in `pyproject.toml` so
casual installs of abogen are not affected.
