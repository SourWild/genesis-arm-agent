#!/usr/bin/env bash
# Record a short screen (+ system audio, if available) demo of the agent running.
#
# macOS has no built-in loopback device, so capturing the `say` voice announcements
# requires a virtual audio driver like BlackHole:
#   brew install blackhole-2ch
# Then, in Audio MIDI Setup, create a "Multi-Output Device" combining your speakers
# + BlackHole 2ch, and select it as your system output — that way you still hear
# audio live while ffmpeg also records it. Without that setup, this script falls
# back to video-only (no error, just a warning).
#
# Usage:
#   ./record_demo.sh                      # 30s screen (+ system audio if BlackHole set up)
#   ./record_demo.sh --duration 60
#   ./record_demo.sh --list-devices        # print ffmpeg's avfoundation device indices
#   VIDEO_DEVICE=4 AUDIO_DEVICE=2 ./record_demo.sh   # force specific device indices

set -euo pipefail

DURATION=30
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/demo_recordings"

if [[ "${1:-}" == "--list-devices" ]]; then
    (ffmpeg -f avfoundation -list_devices true -i "" 2>&1 || true) | grep -E "AVFoundation (video|audio) devices|^\[AVFoundation.*\]\s+\[[0-9]+\]"
    exit 0
fi

if [[ "${1:-}" == "--duration" ]]; then
    DURATION="${2:?--duration needs a number of seconds}"
fi

mkdir -p "$OUT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="$OUT_DIR/demo_${TIMESTAMP}.mp4"

DEVICE_LIST="$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1 || true)"

VIDEO_DEVICE="${VIDEO_DEVICE:-$(echo "$DEVICE_LIST" | grep -m1 "Capture screen" | sed -E 's/.*\[([0-9]+)\].*/\1/')}"
if [[ -z "$VIDEO_DEVICE" ]]; then
    echo "Could not auto-detect a 'Capture screen' device. Run with --list-devices and set VIDEO_DEVICE manually." >&2
    exit 1
fi

AUDIO_DEVICE="${AUDIO_DEVICE:-$(echo "$DEVICE_LIST" | grep -m1 -i "blackhole" | sed -E 's/.*\[([0-9]+)\].*/\1/')}"

echo "Recording ${DURATION}s of screen (device ${VIDEO_DEVICE}) to: $OUT_FILE"

if [[ -n "$AUDIO_DEVICE" ]]; then
    echo "System audio device found (device ${AUDIO_DEVICE}, e.g. BlackHole) — recording with audio."
    ffmpeg -y -f avfoundation -pix_fmt uyvy422 -framerate 30 \
        -i "${VIDEO_DEVICE}:${AUDIO_DEVICE}" \
        -t "$DURATION" \
        -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
        -c:a aac \
        "$OUT_FILE"
else
    echo "No system-audio loopback device found (no BlackHole) — recording video only." >&2
    echo "See the comment at the top of this script to enable audio capture." >&2
    ffmpeg -y -f avfoundation -pix_fmt uyvy422 -framerate 30 \
        -i "${VIDEO_DEVICE}:none" \
        -t "$DURATION" \
        -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
        "$OUT_FILE"
fi

echo "Saved: $OUT_FILE"
