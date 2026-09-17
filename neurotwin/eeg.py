"""Synthetic EEG signal-processing exercise, independent of the trial simulation.

Signals are generated in microvolts. These features are engineering examples,
not validated disease biomarkers or a BCI decoder.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}


def generate_synthetic_eeg(
    seed: int = 42,
    sample_rate: int = 256,
    duration_seconds: int = 10,
    n_channels: int = 8,
) -> np.ndarray:
    """Create channels x samples, with large artifacts in the final channel.

    This is an oscillator/noise test fixture, not a model of human physiology.
    Artifacts contaminate seconds 2-6 of the final channel.
    """
    if sample_rate < 80 or duration_seconds < 6 or n_channels < 2:
        raise ValueError("Use sample_rate >= 80 Hz, duration >= 6 s, and >= 2 channels.")
    rng = np.random.default_rng(seed)
    time = np.arange(sample_rate * duration_seconds, dtype=float) / sample_rate
    signals = np.empty((n_channels, time.size), dtype=float)
    for channel in range(n_channels):
        phases = rng.uniform(0, 2 * np.pi, size=3)
        alpha_amplitude = 10.0 + channel * 0.5
        signals[channel] = (
            alpha_amplitude * np.sin(2 * np.pi * 10.0 * time + phases[0])
            + 4.5 * np.sin(2 * np.pi * 6.0 * time + phases[1])
            + 2.5 * np.sin(2 * np.pi * 20.0 * time + phases[2])
            + rng.normal(0.0, 2.0, time.size)
        )
    contaminated = (time >= 2.0) & (time < 6.0)
    signals[-1, contaminated] += 180.0 * np.sin(2 * np.pi * 3.0 * time[contaminated])
    return signals


def reject_artifacts(
    signals: np.ndarray,
    sample_rate: int,
    epoch_seconds: float = 2.0,
    max_abs_uv: float = 100.0,
    max_peak_to_peak_uv: float = 150.0,
    max_rms_uv: float = 40.0,
    max_rejected_fraction: float = 0.2,
) -> list[dict]:
    """Flag nonfinite, flat, or high-amplitude epochs and reject poor channels.

    Fixed thresholds are deliberately transparent demo settings. They have not
    been calibrated for any EEG device or participant population. A channel is
    rejected if more than 20% of its epochs fail (default); failed epochs from
    retained channels are still excluded from spectral estimation.
    """
    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2 or sample_rate <= 0 or epoch_seconds <= 0:
        raise ValueError("Expected channels x samples and positive timing settings.")
    epoch_samples = int(sample_rate * epoch_seconds)
    if epoch_samples < 2 or signals.shape[1] < epoch_samples:
        raise ValueError("At least one full epoch containing two samples is required.")
    if signals.shape[1] % epoch_samples:
        raise ValueError("Signal length must contain a whole number of epochs.")
    if not 0 <= max_rejected_fraction <= 1:
        raise ValueError("max_rejected_fraction must lie between zero and one.")
    n_epochs = signals.shape[1] // epoch_samples
    report = []
    for index, channel in enumerate(signals):
        failed = []
        for epoch_index, epoch in enumerate(channel.reshape(n_epochs, epoch_samples)):
            reasons = []
            if not np.isfinite(epoch).all():
                reasons.append("nonfinite")
            else:
                centered = epoch - epoch.mean()
                if np.std(centered) < 1e-6:
                    reasons.append("flatline")
                if np.max(np.abs(epoch)) > max_abs_uv:
                    reasons.append("absolute_amplitude")
                if np.ptp(epoch) > max_peak_to_peak_uv:
                    reasons.append("peak_to_peak")
                if np.sqrt(np.mean(centered**2)) > max_rms_uv:
                    reasons.append("rms")
            if reasons:
                failed.append({"epoch_index": epoch_index, "reasons": reasons})
        bad_indices = {item["epoch_index"] for item in failed}
        fraction = len(failed) / n_epochs
        report.append({
            "channel": f"CH{index + 1:02d}",
            "accepted": fraction <= max_rejected_fraction and len(failed) < n_epochs,
            "n_epochs": n_epochs,
            "rejected_fraction": fraction,
            "rejected_epochs": failed,
            "accepted_epoch_indices": [i for i in range(n_epochs) if i not in bad_indices],
        })
    return report


def estimate_spectrum(epochs: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Average epoch Welch PSDs and integrate bandpower in microvolts squared.

    Endpoints are linearly interpolated so adjacent bands share an integration
    boundary without overlapping frequency intervals. PSD units are uV^2/Hz.
    """
    epochs = np.atleast_2d(np.asarray(epochs, dtype=float))
    if epochs.ndim != 2 or epochs.size == 0 or not np.isfinite(epochs).all():
        raise ValueError("Expected finite, nonempty epochs x samples.")
    if sample_rate <= 60 or epochs.shape[-1] < 2:
        raise ValueError("Sample rate must exceed 60 Hz with at least two samples.")
    frequencies, epoch_psd = welch(
        epochs, fs=sample_rate, window="hann", nperseg=min(2 * sample_rate, epochs.shape[-1]),
        noverlap=0, detrend="constant", scaling="density", axis=-1,
    )
    psd = epoch_psd.mean(axis=0)
    features = {}
    for name, (low, high) in BANDS.items():
        interior = (frequencies > low) & (frequencies < high)
        band_frequencies = np.concatenate(([low], frequencies[interior], [high]))
        band_psd = np.interp(band_frequencies, frequencies, psd)
        features[f"{name}_power_uv2"] = float(trapezoid(band_psd, band_frequencies))
    alpha = features["alpha_power_uv2"]
    features["theta_alpha_ratio"] = features["theta_power_uv2"] / alpha if alpha > 0 else None
    return frequencies, psd, features


def run_eeg_demo(seed: int = 42) -> dict:
    """Return a reproducible, JSON-serializable independent signal demo."""
    sample_rate = 256
    epoch_seconds = 2
    signals = generate_synthetic_eeg(seed=seed, sample_rate=sample_rate)
    qc = reject_artifacts(signals, sample_rate, epoch_seconds=epoch_seconds)
    features = []
    spectra = {}
    frequencies = None
    for channel_index, item in enumerate(qc):
        if not item["accepted"]:
            continue
        epochs = signals[channel_index].reshape(-1, sample_rate * epoch_seconds)
        clean_epochs = epochs[item["accepted_epoch_indices"]]
        frequencies, psd, channel_features = estimate_spectrum(clean_epochs, sample_rate)
        features.append({"channel": item["channel"], **channel_features})
        spectra[item["channel"]] = psd[frequencies <= 40].tolist()
    # Include the injected artifact in the preview (0-6 s); decimate display only.
    preview_indices = np.arange(0, sample_rate * 6, 4)
    return {
        "scope": "Synthetic EEG engineering exercise; independent of trial simulation.",
        "config": {
            "seed": seed, "sample_rate_hz": sample_rate, "duration_seconds": 10,
            "n_channels": 8, "signal_unit": "microvolt", "epoch_seconds": epoch_seconds,
            "artifact_thresholds": {
                "max_abs_uv": 100.0, "max_peak_to_peak_uv": 150.0,
                "max_centered_rms_uv": 40.0, "max_rejected_fraction": 0.2,
            },
            "injected_artifact": {"channel": "CH08", "start_seconds": 2, "end_seconds": 6},
        },
        "channel_qc": qc,
        "accepted_channels": [item["channel"] for item in qc if item["accepted"]],
        "rejected_channels": [item["channel"] for item in qc if not item["accepted"]],
        "features": features,
        "plot_data": {
            "trace_time_seconds": (preview_indices / sample_rate).tolist(),
            "traces_uv": {item["channel"]: signals[i, preview_indices].tolist() for i, item in enumerate(qc)},
            "psd_frequencies_hz": frequencies[frequencies <= 40].tolist() if frequencies is not None else [],
            "psd_uv2_per_hz": spectra,
            "trace_display_note": "Display-only decimation; all spectral calculations use full-rate signals.",
        },
        "limitations": [
            "Generated oscillations and artifacts are test fixtures, not recorded human EEG.",
            "Thresholds are illustrative and have not been validated on hardware or human data.",
            "Bandpowers and ratios are not validated Alzheimer disease biomarkers.",
            "No BCI intent decoding, device control, or relationship to trial outcomes is implemented.",
        ],
    }
