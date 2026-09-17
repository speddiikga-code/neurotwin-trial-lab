# Research notes

## Research question and scope

Can a prognostic score learned on independent historical data reduce uncertainty in a randomized treatment comparison? How does that benefit change when the score carries little signal or its relationship with the outcome changes?

The trial simulation studies this question with entirely synthetic participants and outcomes. A separate synthetic EEG exercise demonstrates spectral feature extraction as a starting point for neuroengineering practice. EEG features are not presented as validated Alzheimer's biomarkers, and the EEG example is not evidence for the clinical-trial simulation.

## What the source paper actually did

[Wang et al. (2025)](https://doi.org/10.1002/trc2.70181) trained a CRBM on harmonized historical clinical/observational data from 6,736 people, using a 50%/20%/30% training/tuning/test split followed by retraining on the first two partitions. The model was also evaluated on placebo participants from three independent trials. Predictions were then generated from baseline information in AWARE, a 453-participant trial with three tilavonemab dose groups and placebo. The analysis examined week-96 cognitive and functional change scores.

The CRBM forecasts disease progression under placebo. It does not learn individual treatment effects. Its prognostic predictions enter the randomized trial analysis as covariates; simulated predictions do not replace randomized control participants. The article considers both single-endpoint ANCOVA and repeated-measures models. The present project is a simplified simulation, with neither the source CRBM nor its MMRM analysis.

The article's main summary reports partial correlations of 0.30-0.39, residual variance reductions around 9%-15%, and potential sample-size savings of 9%-15% overall or 17%-26% in the control arm. These are retrospective findings and design projections, not prospective savings established by this repository. Figures 2-3 and Table 3 use different populations/comparators: Table 3's dose-specific CDR-SB results compare against baseline-adjusted ANCOVA and give approximately 9%-17% overall and 16%-29% control savings. Those ranges should not be mixed with the abstract's comparison. See paper Sections 2.6-2.7, 3.3-3.4, Figures 2-3, and Table 3.

## Statistical mechanism

Let `A` be randomized treatment, `Y` the observed endpoint, `X` prespecified baseline covariates, and `S = f(X)` a score trained independently of trial outcomes. Compare:

```text
Y = intercept + treatment_effect * A + baseline_terms + error
Y = intercept + treatment_effect * A + baseline_terms + score_weight * S + error
```

The treatment coefficient still compares randomized groups. The score explains predictable outcome variability. The relevant additional predictive correlation is the correlation remaining after residualizing both the outcome and score on the comparator's covariates; a raw correlation cannot generally be substituted for this partial correlation.

For a fixed additional scalar covariate, the ordinary least-squares residual sum-of-squares ratio is `1 - r_partial^2`. Residual variance estimates also include degrees-of-freedom corrections. The ratio of treatment-effect variances need not match this identity exactly in a finite randomized sample, because treatment and covariates may be imbalanced. Robust standard errors introduce further differences.

## Approximate sample-size arithmetic

For a simple two-arm model with balanced randomization, equal residual variances in the arms, a stable prognostic relationship, and a fixed effect/target power, use:

```text
q = r^2
adjusted total sample size / original total sample size ~= 1 - q
approximate total sample-size saving ~= q
```

These are large-sample planning approximations. They require the relevant correlation for the chosen comparator and do not account for attrition, longitudinal covariance, multiple endpoints, uncertainty in the estimated correlation, or small-sample degrees of freedom.

If the original design has `n` participants in **each** arm and the treated arm is held at `n`, equating the approximate treatment-effect variances gives:

```text
original variance = sigma^2 * (1/n + 1/n)
adjusted variance = sigma^2 * (1-q) * (1/n + 1/n_control)
n_control / n = (1-q)/(1+q)
control-only saving = 2*q/(1+q)
```

For `r=0.30`, this gives 9.0% overall or 16.5% control-only saving; for `r=0.39`, 15.2% or 26.4%. The control-only and overall percentages describe different designs, so they cannot be added. This derivation assumes the original 1:1 design, common adjusted variances, and unchanged treated-arm size; it is not a general formula for AWARE's four-arm allocation.

## What to inspect in the simulation

- **Independent training:** The score must be fitted before using the trial outcomes. Training on the evaluated outcomes creates leakage and invalidates the intended experiment.
- **Null experiments:** With the treatment effect set to zero, examine the false-positive proportion and confidence-interval coverage over repetitions. A single nonsignificant result is not a calibration test.
- **Alternative experiments:** Compare empirical estimator variability, uncertainty, and power against the chosen unadjusted or baseline-adjusted comparator. Report the seed and repetitions with results.
- **Weak signal and shift:** A weak score should confer little benefit. Changed relationships can erode benefit even when randomization still protects the treatment comparison in an appropriate analysis.
- **Monte Carlo uncertainty:** A simulated proportion `p` from `B` repetitions has approximate standard error `sqrt(p*(1-p)/B)`. Apparent small differences may be simulation noise. Empirical calibration is not a universal mathematical guarantee.

An additive offset to all predicted scores is absorbed by an intercept, and multiplying a nonconstant score by a nonzero constant does not change its linear span. Those transformations alone are therefore poor demonstrations of harmful distribution shift. A meaningful stress test changes prognostic relationships, noise, missingness, or population support.

## Limits and a neuroengineering development path

The paper identifies missing features, population underrepresentation, historical bias, and transportability across subgroups/endpoints as limitations. Our synthetic generator cannot establish real-world robustness to any of these. No clinical data, physiology, causal disease mechanism, model weights from the articles, or prospective clinical validation are included.

The [MIT announcement](https://picower.mit.edu/news/mit-based-team-releases-first-ai-foundation-model-alzheimers-prevention) motivates broader multimodal research. Its advertised benchmark improvements are not results or requirements for this project.

For a next neuroengineering study, select one clearly defined EEG task and an appropriately licensed public dataset. Define participant-level train/test separation, preprocessing and artifact rejection, a simple baseline, an external or held-out evaluation, and metrics before model fitting. A working BCI additionally needs acquisition hardware or a validated recording interface, online timing, decoder evaluation, and a feedback/control task. This repository establishes a reproducible software exercise, not those capabilities or a professional qualification.
