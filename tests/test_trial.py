import json
import unittest

import numpy as np

from neurotwin.trial import ancova_hc2, fit_prognostic_model, partial_correlation, run_experiment


class TrialTests(unittest.TestCase):
    def test_hc2_matches_direct_sandwich(self):
        rng = np.random.default_rng(10)
        x = np.column_stack((np.ones(80), np.tile([0, 1], 40), rng.normal(size=80)))
        y = x @ np.array([2, -.6, 1.2]) + rng.normal(size=80) * (1 + .4 * x[:, 1])
        bread = np.linalg.inv(x.T @ x)
        beta = bread @ x.T @ y
        residual = y - x @ beta
        h = np.sum((x @ bread) * x, axis=1)
        meat = x.T @ ((residual ** 2 / (1 - h))[:, None] * x)
        expected_se = np.sqrt((bread @ meat @ bread)[1, 1])
        result = ancova_hc2(y, x)
        self.assertAlmostEqual(result["estimate"], beta[1], places=12)
        self.assertAlmostEqual(result["standard_error"], expected_se, places=12)

    def test_rank_deficiency_rejected(self):
        treatment = np.tile([0, 1], 10)
        x = np.column_stack((np.ones(20), treatment, treatment))
        with self.assertRaisesRegex(ValueError, "full column rank"):
            ancova_hc2(np.arange(20), x)

    def test_frozen_score_is_batch_independent(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(60, 4))
        model = fit_prognostic_model(x, x[:, 0] + rng.normal(size=60))
        self.assertAlmostEqual(model.predict(x[:1])[0], model.predict(x)[0])

    def test_partial_correlation_removes_common_covariate(self):
        # Orthogonal vectors isolate a known residual correlation of zero.
        q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(40, 4)))
        controls = q[:, :2]
        y, score = 10 * q[:, 0] + q[:, 2], 10 * q[:, 0] + q[:, 3]
        self.assertAlmostEqual(partial_correlation(y, score, controls), 0, places=12)

    def test_reproducible_json_and_summary_identities(self):
        kwargs = dict(seed=11, n_train=120, n_trial=60, replicates=8)
        result = run_experiment(**kwargs)
        self.assertEqual(result, run_experiment(**kwargs))
        json.dumps(result, allow_nan=False)
        self.assertIsNone(result["scenarios"]["null"]["score_ancova"]["power"])
        self.assertIsNone(result["scenarios"]["aligned"]["score_ancova"]["type_i_error"])
        for scenario in result["scenarios"].values():
            self.assertEqual(len(scenario["replicate_results"]), kwargs["replicates"])
            audit_mean = np.mean([row["score_estimate"] for row in scenario["replicate_results"]])
            self.assertAlmostEqual(audit_mean, scenario["score_ancova"]["mean_estimate"])
            expected = scenario["score_ancova"]["empirical_sd"] ** 2 / scenario["standard_ancova"]["empirical_sd"] ** 2
            self.assertAlmostEqual(scenario["paired_estimator_variance_ratio"], expected)
            self.assertTrue(0 <= scenario["score_ancova"]["coverage"] <= 1)

    def test_invalid_sizes_rejected(self):
        for kwargs in ({"replicates": 1}, {"n_trial": 8}, {"seed": -1}, {"n_train": 100.5}):
            with self.assertRaises(ValueError):
                run_experiment(**kwargs)


if __name__ == "__main__":
    unittest.main()
