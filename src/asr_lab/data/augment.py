"""Degrade clean speech the way real usage does: background noise and a telephone channel.

Experiment E10 builds usage sessions from clean FLEURS recordings. The noise and the channel are
applied to the audio itself, so their effect on the model's errors is real, not simulated.

Every function takes and returns mono float32 waveforms at 16 kHz; the random ones take a
``numpy.random.Generator``, so that each session can be rebuilt exactly from its seed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import soxr

SAMPLE_RATE = 16_000
TELEPHONE_SAMPLE_RATE = 8_000  # narrowband telephony: 4 kHz of bandwidth
MU = 255  # G.711 mu-law, the codec of traditional telephone networks


def rms(waveform: np.ndarray) -> float:
    """Root mean square: the average level of a signal."""
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` to ``speech`` so that the signal-to-noise ratio is ``snr_db`` decibels.

    The noise is looped or cut to the length of the speech. Both powers are measured over the
    whole utterance, pauses included. If the mixture would clip, it is scaled down as a whole,
    which keeps the ratio.

    >>> speech = np.ones(4, dtype=np.float32) * 0.1
    >>> noise = np.array([1, -1], dtype=np.float32)
    >>> mixture = mix_at_snr(speech, noise, 20.0)
    >>> print(f"{20 * np.log10(rms(speech) / rms(mixture - speech)):.2f} dB")
    20.00 dB

    Raises:
        ValueError: if the speech or the noise is silent (the ratio is undefined).
    """
    if rms(speech) == 0 or rms(noise) == 0:
        raise ValueError("cannot set the SNR of a silent signal")
    noise = np.resize(noise, len(speech))
    scale = rms(speech) / (rms(noise) * 10 ** (snr_db / 20))
    mixture = speech + scale * noise
    peak = float(np.max(np.abs(mixture)))
    if peak > 1:
        mixture = mixture / peak
    return mixture.astype(np.float32)


def pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary noise whose power falls with frequency (1/f), like fans, engines or traffic.

    White noise is shaped in the frequency domain; the result has an RMS of 1.
    """
    spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    frequencies = np.fft.rfftfreq(n_samples)
    frequencies[0] = frequencies[1] if n_samples > 2 else 1.0  # no infinite gain at 0 Hz
    noise = np.fft.irfft(spectrum / np.sqrt(frequencies), n_samples)
    return (noise / rms(noise)).astype(np.float32)


def babble(voices: Sequence[np.ndarray], n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Several people talking at once, as in a café or an open office.

    Each voice is brought to the same level, looped to ``n_samples`` from a random starting
    point (so that the voices do not all start together) and added; the result has an RMS of 1.
    """
    if not voices:
        raise ValueError("babble needs at least one voice")
    mixture = np.zeros(n_samples, dtype=np.float64)
    for voice in voices:
        start = int(rng.integers(len(voice)))
        mixture += np.resize(np.roll(voice, -start), n_samples) / rms(voice)
    return (mixture / rms(mixture)).astype(np.float32)


def mu_law_encode(waveform: np.ndarray) -> np.ndarray:
    """Compress samples in [-1, 1] to 8 bits with G.711 mu-law: fine steps for quiet sounds,
    coarse steps for loud ones."""
    clipped = np.clip(waveform, -1.0, 1.0)
    compressed = np.sign(clipped) * np.log1p(MU * np.abs(clipped)) / np.log1p(MU)
    return np.round((compressed + 1) / 2 * MU).astype(np.uint8)


def mu_law_decode(codes: np.ndarray) -> np.ndarray:
    """Expand 8-bit mu-law codes back to samples in [-1, 1]."""
    compressed = codes.astype(np.float64) / MU * 2 - 1
    return (np.sign(compressed) * np.expm1(np.abs(compressed) * np.log1p(MU)) / MU).astype(
        np.float32
    )


def telephone_channel(waveform: np.ndarray) -> np.ndarray:
    """A narrowband telephone call: 8 kHz sampling (nothing above 4 kHz) and 8-bit mu-law.

    The result is resampled back to 16 kHz, the rate every model receives, with the same length.
    """
    narrowband = soxr.resample(waveform, SAMPLE_RATE, TELEPHONE_SAMPLE_RATE)
    coded = mu_law_decode(mu_law_encode(narrowband))
    wideband = soxr.resample(coded, TELEPHONE_SAMPLE_RATE, SAMPLE_RATE)
    return np.resize(wideband, len(waveform)).astype(np.float32)
