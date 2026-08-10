"""macOS text-to-speech announcements via `say`, fired at key harness milestones."""

import subprocess

VOICE = "Tingting"  # macOS built-in Chinese (Mandarin) female voice


def speak(text: str) -> None:
    """Speak `text` asynchronously (fires the process and returns immediately, never blocks).

    Uses subprocess.Popen with an argv list (no shell=True) so `text` is passed straight to
    `say` as a single argument — arbitrary tool-result text can't break out into shell syntax.
    Silently no-ops if `say` isn't available (e.g. running this off macOS) since voice is a
    nice-to-have, not something that should ever crash the pick-place loop.
    """
    try:
        subprocess.Popen(
            ["say", "-v", VOICE, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass
