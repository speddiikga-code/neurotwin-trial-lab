"""Frozen prognostic adjustment in fully synthetic randomized trials.

This is an independent educational simulation, not a reproduction of the paper
by Wang et al. (2025), DOI 10.1002/trc2.70181, or a validated digital twin.

DGP: age ~ Normal(70, 7), baseline cognition C ~ Normal(0, 1), functional
score F = .5*C + sqrt(.75)*Normal(0, 1), APOE indicator A ~ Bernoulli(.30).
The historical untreated outcome is .03*(age-70) + .65*C + .90*F + .50*A
+ Normal(0, 1). A ridge model is fitted once on that independent historical
cohort. Aligned/null trial outcomes use the same untreated equation. Shifted
trials instead use .65*C + Normal(0, 1): historical covariates lose all
incremental prognostic value beyond the cognition covariate already adjusted.
Each trial assigns exactly floor(n/2) randomly selected participants treatment;
the outcome adds a constant treatment effect (-.35 aligned/shifted, 0 null).
Scores are frozen before trial randomization; trial outcomes never train them.
All outcome units and covariates are invented for this demonstration.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import t as student_t


FEATURE_NAMES = ("age", "baseline_cognition", "functional_score", "apoe_indicator")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    n_train: int = 2000
    n_trial: int = 300
    replicates: int = 200
    alpha: float = 0.05
    ridge_penalty: float = 1.0

    def validate(self) -> None:
        for name in ("seed", "n_train", "n_trial", "replicates"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.seed < 0 or self.n_train < 20 or self.n_trial < 20:
            raise ValueError("seed must be nonnegative and cohort sizes at least 20")
        if self.replicates < 2:
            raise ValueError("replicates must be at least 2 for empirical variance")
        if not 0 < self.alpha < 1 or not np.isfinite(self.ridge_penalty) or self.ridge_penalty < 0:
            raise ValueError("alpha must be in (0, 1); ridge penalty must be finite and nonnegative")


@dataclass(frozen=True)
class FrozenPrognosticModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.intercept + ((features - self.mean) / self.scale) @ self.coefficients


def synthetic_features(rng: np.random.Generator, n: int) -> np.ndarray:
    cognition = rng.normal(size=n)
    return np.column_stack((
        rng.normal(70, 7, size=n), cognition,
        .5 * cognition + np.sqrt(.75) * rng.normal(size=n),
        rng.binomial(1, .30, size=n),
    ))


def untreated_mean(features: np.ndarray, shifted: bool = False) -> np.ndarray:
    age, cognition, function, apoe = features.T
    if shifted:
        return .65 * cognition
    return .03 * (age - 70) + .65 * cognition + .90 * function + .50 * apoe


def fit_prognostic_model(features: np.ndarray, outcome: np.ndarray, penalty: float = 1.0) -> FrozenPrognosticModel:
    """Ridge uses historical-cohort scaling, without penalizing the intercept."""
    features = np.asarray(features, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if features.ndim != 2 or outcome.shape != (features.shape[0],):
        raise ValueError("features must be a matrix and outcome a matching vector")
    if not np.isfinite(features).all() or not np.isfinite(outcome).all():
        raise ValueError("training data must be finite")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("training covariates must vary")
    standardized = (features - mean) / scale
    coefficients = np.linalg.solve(
        standardized.T @ standardized + penalty * np.eye(features.shape[1]),
        standardized.T @ (outcome - outcome.mean()),
    )
    return FrozenPrognosticModel(mean, scale, coefficients, float(outcome.mean()))


def ancova_hc2(outcome: np.ndarray, design: np.ndarray, alpha: float = .05) -> dict[str, float]:
    """OLS with HC2 sandwich SE and approximate Student-t inference.

    Column 0 must be an intercept and column 1 the treatment indicator.
    HC2 corrects residual squares by (1 - leverage); df = n - p.
    """
    y, x = np.asarray(outcome, dtype=float), np.asarray(design, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or y.shape != (x.shape[0],):
        raise ValueError("design must have at least two columns and match outcome")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not 0 < alpha < 1:
        raise ValueError("data must be finite and alpha must be in (0, 1)")
    n, p = x.shape
    if n <= p or np.linalg.matrix_rank(x) != p:
        raise ValueError("ANCOVA design must have full column rank and positive residual df")
    if not np.allclose(x[:, 0], 1) or not np.array_equal(np.unique(x[:, 1]), [0, 1]):
        raise ValueError("design requires an intercept then binary treatment with both arms")
    q, r = np.linalg.qr(x, mode="reduced")
    beta = np.linalg.solve(r, q.T @ y)
    residual = y - x @ beta
    leverage = np.einsum("ij,ij->i", q, q)
    if np.any(1 - leverage <= np.finfo(float).eps):
        raise ValueError("HC2 is undefined for unit-leverage observations")
    # x pseudoinverse maps outcomes to coefficients, allowing stable HC2 assembly.
    inverse_design = np.linalg.solve(r, q.T)
    variance = float(np.sum(inverse_design[1] ** 2 * residual ** 2 / (1 - leverage)))
    se = float(np.sqrt(max(variance, 0)))
    effect = float(beta[1])
    statistic = effect / se if se > 0 else (0.0 if effect == 0 else np.copysign(np.inf, effect))
    critical = float(student_t.ppf(1 - alpha / 2, n - p))
    return {
        "estimate": effect, "standard_error": se,
        "p_value": float(2 * student_t.sf(abs(statistic), n - p)),
        "ci_low": effect - critical * se, "ci_high": effect + critical * se,
    }


def partial_correlation(outcome: np.ndarray, score: np.ndarray, covariates: np.ndarray) -> float:
    """Residual correlation after removing intercept, treatment and cognition."""
    residual_y = outcome - covariates @ np.linalg.lstsq(covariates, outcome, rcond=None)[0]
    residual_s = score - covariates @ np.linalg.lstsq(covariates, score, rcond=None)[0]
    denominator = np.linalg.norm(residual_y) * np.linalg.norm(residual_s)
    return float(np.clip(residual_y @ residual_s / denominator, -1, 1)) if denominator > 0 else 0.0


def _summarize(fits: list[dict[str, float]], truth: float, alpha: float) -> dict[str, Any]:
    estimate = np.array([f["estimate"] for f in fits])
    rejection = float(np.mean([f["p_value"] < alpha for f in fits]))
    coverage = float(np.mean([f["ci_low"] <= truth <= f["ci_high"] for f in fits]))
    return {
        "mean_estimate": float(estimate.mean()), "bias": float(estimate.mean() - truth),
        "empirical_sd": float(estimate.std(ddof=1)),
        "mean_standard_error": float(np.mean([f["standard_error"] for f in fits])),
        "rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
        "coverage": coverage, "coverage_mc_se": float(np.sqrt(coverage * (1 - coverage) / len(fits))),
        "rejection_rate": rejection,
        "rejection_mc_se": float(np.sqrt(rejection * (1 - rejection) / len(fits))),
        "type_i_error": rejection if truth == 0 else None,
        "power": rejection if truth != 0 else None,
    }


def run_experiment(seed: int = 42, n_train: int = 2000, n_trial: int = 300,
                   replicates: int = 200) -> dict[str, Any]:
    """Return JSON-ready results conditional on one frozen historical model.

    Each method sees the same trial in each replicate. Trial replicates and
    scenarios are independent of the historical cohort; retraining uncertainty
    is not integrated. Small replicate counts are demonstration-level precision.
    """
    config = ExperimentConfig(seed, n_train, n_trial, replicates)
    config.validate()
    streams = np.random.SeedSequence(seed).spawn(4)
    historical_rng = np.random.default_rng(streams[0])
    features = synthetic_features(historical_rng, n_train)
    historical_outcome = untreated_mean(features) + historical_rng.normal(size=n_train)
    model = fit_prognostic_model(features, historical_outcome, config.ridge_penalty)
    scenarios: dict[str, Any] = {}
    for name, truth, stream in zip(("aligned", "null", "shifted"), (-.35, 0.0, -.35), streams[1:]):
        rng = np.random.default_rng(stream)
        standard_fits, augmented_fits, correlations = [], [], []
        for _ in range(replicates):
            x = synthetic_features(rng, n_trial)
            score = model.predict(x)
            treatment = np.zeros(n_trial)
            treatment[rng.permutation(n_trial)[:n_trial // 2]] = 1
            y = untreated_mean(x, name == "shifted") + truth * treatment + rng.normal(size=n_trial)
            standard_design = np.column_stack((np.ones(n_trial), treatment, x[:, 1]))
            augmented_design = np.column_stack((standard_design, score))
            standard_fits.append(ancova_hc2(y, standard_design, config.alpha))
            augmented_fits.append(ancova_hc2(y, augmented_design, config.alpha))
            correlations.append(partial_correlation(y, score, standard_design))
        standard = _summarize(standard_fits, truth, config.alpha)
        augmented = _summarize(augmented_fits, truth, config.alpha)
        r_squared = float(np.mean(np.square(correlations)))
        replicate_results = []
        for index, (standard_fit, score_fit, correlation) in enumerate(
                zip(standard_fits, augmented_fits, correlations), start=1):
            record = {"replicate": index, "partial_r": correlation}
            for prefix, fit in (("standard", standard_fit), ("score", score_fit)):
                for output_name, key in (("estimate", "estimate"), ("se", "standard_error"),
                                         ("p", "p_value"), ("ci_low", "ci_low"), ("ci_high", "ci_high")):
                    record[f"{prefix}_{output_name}"] = fit[key]
            replicate_results.append(record)
        scenarios[name] = {
            "true_treatment_effect": truth, "standard_ancova": standard, "score_ancova": augmented,
            "paired_estimator_variance_ratio": augmented["empirical_sd"] ** 2 / standard["empirical_sd"] ** 2,
            "mean_partial_correlation": float(np.mean(correlations)),
            "mean_partial_r_squared": r_squared,
            "approx_total_sample_reduction_percent": 100 * r_squared,
            "approx_control_only_reduction_percent": 100 * 2 * r_squared / (1 + r_squared),
            "prognostic_shift": name == "shifted",
            "replicate_results": replicate_results,
        }
    return {
        "schema_version": 1, "experiment": "synthetic_frozen_prognostic_ancova",
        "config": asdict(config), "scenarios": scenarios,
        "historical_model": {
            "type": "ridge regression", "features": list(FEATURE_NAMES),
            "intercept": model.intercept, "standardized_coefficients": model.coefficients.tolist(),
            "feature_means": model.mean.tolist(), "feature_scales": model.scale.tolist(),
            "training": "One independent untreated historical cohort; model fixed across all trials.",
        },
        "inference": "OLS treatment coefficient; HC2 standard errors; approximate two-sided t inference with n-p df.",
        "approximation_assumptions": (
            "Sample-size formulas are illustrative asymptotic calculations using trial-outcome residual r squared. "
            "They assume stable linear prognostic association, equal arm residual variances, constant effects, "
            "and equal starting allocation. Total reduction is r squared; control-only reduction is "
            "2*r squared/(1+r squared) when the original treatment-arm size is held fixed. "
            "Estimated partial correlation uses outcomes and is a retrospective diagnostic, not a prospective design guarantee."
        ),
        "limitations": [
            "All data and effects are synthetic; no patient data, clinical validation or pretrained foundation model.",
            "Conditional on one fitted historical model; uncertainty from retraining is not integrated.",
            "Shift stress test removes all score information beyond baseline cognition; it is not a comprehensive shift model.",
            "ANCOVA estimates the score coefficient anew; adding an uninformative score can slightly reduce precision.",
            "No missingness, dropout, site effects, repeated measures or treatment-effect heterogeneity.",
        ],
    }
