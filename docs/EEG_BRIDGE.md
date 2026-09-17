# EEG signal engineering exercise

This repository contains two independent learning exercises: a trial simulation
and this synthetic EEG pipeline. **No EEG feature is used to predict trial
outcomes, and neither exercise validates a clinical digital twin.**

The EEG module demonstrates signal-processing fundamentals relevant to a future
neuroengineering portfolio: sampling, units, epoch quality control, spectral
estimation, reproducible feature extraction, and explicit validation limits.
It does not implement a BCI decoder or demonstrate expertise with human recordings.

## What runs

`neurotwin.eeg.run_eeg_demo(seed=42)` creates eight channels at 256 Hz for ten
seconds. Each contains a 10 Hz alpha oscillation, a 6 Hz theta oscillation, a
20 Hz beta oscillation, and seeded Gaussian noise. Amplitudes are expressed in
microvolts. A 180 microvolt, 3 Hz contamination is deliberately added to CH08
between seconds 2 and 6. Channel labels are generic; no electrode montage or
spatial physiology is implied.

The pipeline divides signals into nonoverlapping two-second epochs. It rejects
epochs containing nonfinite values, flatlines, absolute amplitudes above 100 uV,
peak-to-peak excursions above 150 uV, or centered RMS above 40 uV. It rejects a
channel when more than 20% of its epochs fail. Bad epochs are excluded even
when their channel is retained. These thresholds are demonstration settings,
not a clinical artifact detector.

Welch power spectral density uses a Hann window on each retained two-second
epoch, yielding 0.5 Hz frequency spacing. The resulting PSDs are averaged.
Bandpower integrates the PSD over theta (4-8 Hz), alpha (8-13 Hz), and beta
(13-30 Hz), with interpolated boundaries. Power units are uV squared; PSD units
are uV squared per Hz. The theta/alpha ratio is dimensionless. Adjacent bands
share a mathematical boundary but no frequency interval.

The result includes channel and epoch QC, accepted-channel features, short
decimated display traces, and PSD arrays. Spectral calculations always use the
full-rate signals. These generated oscillations do not need a notch or a
band-pass filter to recover their known frequencies; a real acquisition
pipeline would require an explicit filtering and referencing design.

## What the tests establish

- A known 10 Hz sinusoid produces a 10 Hz PSD peak and integrated alpha power
  close to its analytic mean-square value, amplitude squared divided by two.
- The injected artifact rejects CH08 and the expected contaminated epochs.
- A single bad epoch is excluded while an otherwise usable channel is retained.
- Nonfinite signal samples are detected.
- Identical seeds yield identical results that serialize as strict JSON.

The tests establish properties of this synthetic fixture only. Theta/alpha
ratios are **not validated disease biomarkers here**. No patient signal,
diagnosis, motor intent, clinical utility, real-time latency, or hardware
performance has been evaluated.

## A credible next BCI project

Use a suitably licensed public EEG dataset and document its consent/access
conditions. Specify a concrete decoding task; separate participants or sessions
before fitting preprocessing; compare a simple prespecified baseline; and
report subject-level uncertainty and failure cases. A later hardware exercise
should separately measure acquisition timing and end-to-end latency. Those
steps are future work, not results claimed by this repository.
