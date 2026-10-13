"""Optional server-side speech output.

The default is TTS_BACKEND=browser: the page speaks with the Web Speech API and
the server does nothing. TTS_BACKEND=say uses the macOS `say` command and is
for local use only.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

MAX_CHARS = 1200


class TTSError(RuntimeError):
    pass


def say_available() -> bool:
    return platform.system() == "Darwin" and shutil.which("say") is not None


def synthesize_say(text: str, voice: str = "Samantha", rate: int = 150) -> bytes:
    """Return M4A audio for `text`. Nothing is left on disk."""
    if not say_available():
        raise TTSError("the `say` backend needs macOS")
    text = text.strip()[:MAX_CHARS]
    if not text:
        raise TTSError("nothing to say")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "speech.m4a"
        try:
            # Text goes in on stdin (`-f -`), so it can never be read as an option.
            subprocess.run(
                ["say", "-v", voice, "-r", str(rate), "-o", str(out), "-f", "-"],
                input=text.encode(),
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log.error("say failed: %s", exc)
            raise TTSError("speech synthesis failed") from exc
        return out.read_bytes()
