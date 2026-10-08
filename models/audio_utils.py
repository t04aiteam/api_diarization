"""Audio decode/validation helpers shared by all backends.

Per the HDSD doc: input is downmixed to mono and resampled to 24 kHz before
processing, regardless of backend or original format/sample rate.
"""

import librosa

TARGET_SR = 24000
MIN_AUDIO_SECONDS = 0.1


def decode_audio(path: str):
    """Decode an audio file on disk to (mono float32 array at 24kHz, duration_seconds).

    Raises ValueError on anything that can't be decoded.
    """
    try:
        audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc
    duration = len(audio) / TARGET_SR
    return audio, duration
