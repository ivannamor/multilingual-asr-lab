"""Tests for noise mixing and the telephone channel."""

import numpy as np
import pytest

from asr_lab.data.augment import (
    SAMPLE_RATE,
    babble,
    mix_at_snr,
    mu_law_decode,
    mu_law_encode,
    pink_noise,
    rms,
    telephone_channel,
)


def tone(frequency: float, seconds: float = 1.0, amplitude: float = 0.1) -> np.ndarray:
    time = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def snr_db(speech: np.ndarray, mixture: np.ndarray) -> float:
    return 20 * np.log10(rms(speech) / rms(mixture - speech))


@pytest.mark.parametrize("target", [15.0, 5.0, 0.0, -5.0])
def test_mix_at_snr_reaches_the_target(target: float) -> None:
    rng = np.random.default_rng(0)
    speech = tone(440)
    mixture = mix_at_snr(speech, rng.standard_normal(SAMPLE_RATE).astype(np.float32), target)
    assert mixture.dtype == np.float32
    assert snr_db(speech, mixture) == pytest.approx(target, abs=0.01)


def test_mix_at_snr_loops_a_short_noise() -> None:
    noise = np.array([0.5, -0.5], dtype=np.float32)
    mixture = mix_at_snr(tone(440), noise, 10.0)
    assert len(mixture) == SAMPLE_RATE


def test_mix_at_snr_scales_a_clipping_mixture_without_changing_the_ratio() -> None:
    speech = tone(440, amplitude=0.9)
    noise = np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32)
    mixture = mix_at_snr(speech, noise, 0.0)
    assert np.max(np.abs(mixture)) == pytest.approx(1.0)
    gain = np.dot(mixture, speech) / np.dot(speech, speech)  # the common scale factor
    assert snr_db(gain * speech, mixture) == pytest.approx(0.0, abs=0.1)


def test_mix_at_snr_rejects_silence() -> None:
    with pytest.raises(ValueError, match="silent"):
        mix_at_snr(np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32), 5.0)
    with pytest.raises(ValueError, match="silent"):
        mix_at_snr(tone(440), np.zeros(10, dtype=np.float32), 5.0)


def test_pink_noise_has_more_power_at_low_frequencies() -> None:
    noise = pink_noise(4 * SAMPLE_RATE, np.random.default_rng(0))
    assert rms(noise) == pytest.approx(1.0)
    power = np.abs(np.fft.rfft(noise)) ** 2
    frequencies = np.fft.rfftfreq(len(noise), 1 / SAMPLE_RATE)
    low = power[(frequencies > 100) & (frequencies < 200)].mean()
    high = power[(frequencies > 3200) & (frequencies < 6400)].mean()
    assert low / high == pytest.approx(32, rel=0.3)  # 1/f: 32 times the frequency, 1/32 the power


def test_random_noise_is_reproducible_from_its_seed() -> None:
    voices = [tone(200), tone(300, seconds=0.5)]
    for make in (lambda rng: pink_noise(1000, rng), lambda rng: babble(voices, 1000, rng)):
        assert np.array_equal(make(np.random.default_rng(7)), make(np.random.default_rng(7)))
        assert not np.array_equal(make(np.random.default_rng(7)), make(np.random.default_rng(8)))


def test_babble_mixes_voices_at_equal_level() -> None:
    loud, quiet = tone(200, amplitude=0.8), tone(300, amplitude=0.01)
    mixture = babble([loud, quiet], SAMPLE_RATE, np.random.default_rng(0))
    assert rms(mixture) == pytest.approx(1.0)
    spectrum = np.abs(np.fft.rfft(mixture))
    assert spectrum[200] == pytest.approx(spectrum[300], rel=0.01)  # 1 Hz per bin


def test_babble_needs_a_voice() -> None:
    with pytest.raises(ValueError, match="voice"):
        babble([], 10, np.random.default_rng(0))


def test_mu_law_round_trip() -> None:
    samples = np.linspace(-1, 1, 1001, dtype=np.float32)
    codes = mu_law_encode(samples)
    assert codes.dtype == np.uint8
    assert (codes[0], codes[-1]) == (0, 255)
    decoded = mu_law_decode(codes)
    # Quantization steps grow with the level: small error for quiet samples, larger for loud ones.
    assert np.all(np.abs(decoded - samples) <= 0.002 + 0.03 * np.abs(samples))
    assert np.array_equal(mu_law_encode(np.array([2.0, -2.0])), [255, 0])  # clipped


def test_telephone_channel_removes_frequencies_above_4_khz() -> None:
    low, high = tone(1000), tone(6000)
    assert len(telephone_channel(low)) == SAMPLE_RATE
    assert rms(telephone_channel(low)) == pytest.approx(rms(low), rel=0.05)
    assert rms(telephone_channel(high)) < rms(high) / 30
