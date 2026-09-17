"""Check physical signal behavior, QC, and reproducibility of the EEG fixture."""

import json
import unittest

import numpy as np

from neurotwin.eeg import estimate_spectrum, generate_synthetic_eeg, reject_artifacts, run_eeg_demo


class EEGTests(unittest.TestCase):
    def test_alpha_sinusoid_peak_and_integrated_power(self):
        fs = 256
        time = np.arange(fs * 2) / fs
        amplitude = 12.0
        epochs = (amplitude * np.sin(2 * np.pi * 10 * time))[None, :]
        frequencies, psd, features = estimate_spectrum(epochs, fs)
        self.assertAlmostEqual(frequencies[np.argmax(psd)], 10.0)
        # A sinusoid with amplitude A has mean-square power A^2 / 2.
        self.assertAlmostEqual(features["alpha_power_uv2"], amplitude**2 / 2, delta=0.72)
        self.assertLess(features["theta_power_uv2"], 0.001)
        self.assertLess(features["theta_alpha_ratio"], 0.001)

    def test_injected_artifact_rejects_only_contaminated_channel(self):
        qc = reject_artifacts(generate_synthetic_eeg(), 256)
        self.assertEqual([item["channel"] for item in qc if not item["accepted"]], ["CH08"])
        self.assertEqual([item["epoch_index"] for item in qc[-1]["rejected_epochs"]], [1, 2])
        self.assertTrue(all(item["rejected_fraction"] == 0 for item in qc[:-1]))

    def test_isolated_bad_epoch_is_excluded_from_retained_channel(self):
        signals = generate_synthetic_eeg()
        signals[0, :512] = 200.0
        qc = reject_artifacts(signals, 256)
        self.assertTrue(qc[0]["accepted"])
        self.assertEqual(qc[0]["accepted_epoch_indices"], [1, 2, 3, 4])
        self.assertIn("flatline", qc[0]["rejected_epochs"][0]["reasons"])

    def test_nonfinite_epochs_fail_quality_control(self):
        signals = generate_synthetic_eeg()
        signals[0, :1024] = np.nan
        qc = reject_artifacts(signals, 256)
        self.assertFalse(qc[0]["accepted"])
        self.assertEqual(qc[0]["rejected_epochs"][0]["reasons"], ["nonfinite"])

    def test_demo_is_deterministic_and_strict_json_serializable(self):
        first = run_eeg_demo(seed=7)
        second = run_eeg_demo(seed=7)
        self.assertEqual(first, second)
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)
        self.assertEqual(len(first["features"]), 7)
        self.assertTrue(all(item["alpha_power_uv2"] > item["theta_power_uv2"] for item in first["features"]))


if __name__ == "__main__":
    unittest.main()
