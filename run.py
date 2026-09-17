"""Execute the synthetic research demo and export an inspectable static report."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neurotwin.eeg import run_eeg_demo
from neurotwin.trial import run_experiment


METHODS = {"standard_ancova": "Baseline ANCOVA", "score_ancova": "Score + baseline"}
COLORS = ["#8e9baa", "#087f8c"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(trial: dict, eeg: dict, out: Path) -> list[Path]:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f8fafb", "axes.facecolor": "#f8fafb"})
    scenarios = list(trial["scenarios"])
    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for i, (method, label) in enumerate(METHODS.items()):
        metrics = [trial["scenarios"][s][method] for s in scenarios]
        position = x + (i - .5) * .34
        axes[0].bar(position, [m["empirical_sd"] for m in metrics], .32, label=label, color=COLORS[i])
        axes[1].bar(position, [m["coverage"] for m in metrics], .32, color=COLORS[i],
                    yerr=[1.96*m["coverage_mc_se"] for m in metrics], capsize=3)
        axes[2].bar(position, [m["rejection_rate"] for m in metrics], .32, color=COLORS[i],
                    yerr=[1.96*m["rejection_mc_se"] for m in metrics], capsize=3)
    axes[0].set(title="Treatment-estimate variability", ylabel="Empirical standard deviation")
    axes[0].legend(fontsize=8)
    axes[1].set(title="Coverage of 95% intervals", ylim=(0, 1.06))
    axes[1].axhline(.95, color="#d57c43", linestyle="--", linewidth=1)
    axes[2].set(title="Rejection rate (power / null error)", ylim=(0, 1.06))
    axes[2].axhline(.05, color="#d57c43", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_xticks(x, [s.capitalize() for s in scenarios])
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("SYNTHETIC TRIALS  |  Error bars: approximate 95% Monte Carlo uncertainty", fontsize=11)
    trial_path = out / "trial_metrics.png"
    fig.savefig(trial_path, dpi=160)
    plt.close(fig)

    data = eeg["plot_data"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name, color in (("CH01", "#087f8c"), ("CH08", "#c56a4a")):
        axes[0].plot(data["trace_time_seconds"], data["traces_uv"][name], label=name, color=color, linewidth=.8)
    axes[0].axvspan(2, 6, alpha=.07, color="#c56a4a")
    axes[0].set(title="Signal quality: clean vs contaminated", xlabel="Time (s)", ylabel="Amplitude (microvolts)")
    axes[0].legend()
    for channel, psd in data["psd_uv2_per_hz"].items():
        axes[1].semilogy(data["psd_frequencies_hz"], np.maximum(psd, 1e-10), alpha=.65, linewidth=1)
    for a, b, name in ((4, 8, "theta"), (8, 13, "alpha"), (13, 30, "beta")):
        axes[1].axvspan(a, b, alpha=.045, color="#087f8c")
        axes[1].text((a+b)/2, .96, name, transform=axes[1].get_xaxis_transform(), ha="center", va="top", fontsize=9)
    axes[1].set(title="Accepted channels: Welch spectra", xlabel="Frequency (Hz)", ylabel="PSD (microvolts squared / Hz)", xlim=(0, 40))
    fig.suptitle("SYNTHETIC EEG  |  Separate engineering exercise; no disease inference", fontsize=11)
    eeg_path = out / "eeg_quality.png"
    fig.savefig(eeg_path, dpi=160)
    plt.close(fig)
    return [trial_path, eeg_path]


def build_report(trial: dict, eeg: dict, figure_paths: list[Path], out: Path) -> None:
    c = trial["config"]
    aligned = trial["scenarios"]["aligned"]
    null = trial["scenarios"]["null"]["score_ancova"]
    variance_gain = 100*(1-aligned["paired_estimator_variance_ratio"])
    rows = []
    for scenario, result in trial["scenarios"].items():
        for method, label in METHODS.items():
            m = result[method]
            rows.append(f"<tr><td>{scenario.title()}</td><td>{label}</td><td>{m['bias']:.4f}</td>"
                        f"<td>{m['empirical_sd']:.4f}</td><td>{m['mean_standard_error']:.4f}</td>"
                        f"<td>{m['coverage']:.1%}</td><td>{m['rejection_rate']:.1%} ± {1.96*m['rejection_mc_se']:.1%}</td></tr>")
    images = ["data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii") for p in figure_paths]
    limitations = "".join(f"<li>{html.escape(s)}</li>" for s in trial["limitations"])
    features = "".join(f"<tr><td>{f['channel']}</td><td>{f['theta_power_uv2']:.2f}</td>"
                       f"<td>{f['alpha_power_uv2']:.2f}</td><td>{f['beta_power_uv2']:.2f}</td>"
                       f"<td>{f['theta_alpha_ratio']:.3f}</td></tr>" for f in eeg["features"])
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroTwin Trial Lab | Synthetic research report</title><style>
:root{{--ink:#15313d;--muted:#5b6e77;--teal:#087f8c;--line:#d8e3e6}}
*{{box-sizing:border-box}}body{{margin:0;background:#f0f5f6;color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:48px 24px 72px}}.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--teal);font-weight:750}}
h1{{font-size:clamp(36px,6vw,64px);line-height:1.08;letter-spacing:-.04em;margin:14px 0 20px}}h2{{line-height:1.25;margin-top:0;font-size:25px}}p{{max-width:850px}}
.lead{{font-size:20px;color:var(--muted)}}.tag{{display:inline-block;background:#deedef;padding:6px 12px;border-radius:6px;font-size:13px;margin:3px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}}.card,section{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px}}
.value{{display:block;font-size:38px;line-height:1.2;font-weight:750;color:var(--teal)}}.small{{font-size:13px;color:var(--muted)}}section{{margin:22px 0}}img{{width:100%;height:auto;display:block;margin:20px 0}}
table{{border-collapse:collapse;min-width:640px;width:100%;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid var(--line)}}th{{background:#f0f6f7}}.scroll{{overflow:auto}}
code{{background:#edf4f5;border-radius:4px;padding:2px 6px}}a{{color:var(--teal)}}.note{{border-left:4px solid #d57c43;padding-left:18px;color:var(--muted)}}
@media(max-width:700px){{main{{padding:28px 16px}}.cards{{grid-template-columns:1fr}}section{{padding:20px 16px}}}}
</style></head><body><main>
<div class="eyebrow">Neuroscience methods / reproducible experiment</div>
<h1>NeuroTwin Trial Lab</h1>
<p class="lead">Can a frozen prognostic score make a randomized treatment comparison more precise?</p>
<span class="tag">Synthetic data only</span><span class="tag">Seed {c['seed']}</span><span class="tag">{c['replicates']:,} trials per scenario</span>
<p class="note">An educational simulation inspired by digital-twin trial research, with a separate EEG processing exercise. No patient data, foundation-model weights, clinical validation, or working BCI.</p>
<div class="cards"><div class="card"><span class="value">{variance_gain:.1f}%</span>Empirical variance reduction<p class="small">Aligned scenario: score-adjusted versus baseline-adjusted treatment estimates. Simulation result, not clinical sample savings.</p></div>
<div class="card"><span class="value">{null['rejection_rate']:.1%}</span>Null rejection rate<p class="small">Nominal level 5%; Monte Carlo SE {null['rejection_mc_se']:.1%}. Finite simulation evidence only.</p></div>
<div class="card"><span class="value">{len(eeg['accepted_channels'])} / 8</span>EEG channels accepted<p class="small">One deliberately contaminated channel rejected by transparent amplitude and epoch rules.</p></div></div>
<section><h2>01 / Trial precision and calibration</h2>
<p>Fit ridge regression on {c['n_train']:,} independent historical synthetic participants. Freeze it. Randomize {c['n_trial']:,} new participants per trial. Compare ANCOVA using baseline cognition with ANCOVA adding the frozen score. Both methods retain the same randomized controls.</p>
<p><b>Aligned:</b> historical prognostic relationships persist. <b>Null:</b> the same relationships, with zero treatment effect. <b>Shifted:</b> historical associations beyond baseline cognition disappear. The non-null synthetic treatment effect is −0.35 arbitrary units.</p>
<img src="{images[0]}" alt="Three panels compare empirical estimator variability, interval coverage, and rejection rate for baseline and score-adjusted analyses.">
<div class="scroll"><table><thead><tr><th>Scenario</th><th>Method</th><th>Bias</th><th>Empirical SD</th><th>Mean HC2 SE</th><th>95% coverage</th><th>Rejection ± MC margin</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="small">Null rejection estimates type I error; non-null rejection estimates power. Margins are 1.96 Monte Carlo SE, without boundary correction. Outcome units are arbitrary. HC2 + Student-t inference is approximate.</p></section>
<section><h2>02 / A separate EEG engineering bridge</h2>
<p>Eight synthetic channels at 256 Hz for 10 seconds combine 6, 10, and 20 Hz oscillations and noise. Reject poor epochs/channels; average clean-epoch Welch spectra; integrate band power. No connection to trial outcomes is modeled.</p>
<img src="{images[1]}" alt="Synthetic EEG trace shows the injected artifact in channel eight and accepted-channel spectral peaks at 6, 10, and 20 hertz.">
<div class="scroll"><table><thead><tr><th>Channel</th><th>Theta power</th><th>Alpha power</th><th>Beta power</th><th>Theta / alpha</th></tr></thead><tbody>{features}</tbody></table></div>
<p class="small">Power units: microvolts squared. Rejected: {', '.join(eeg['rejected_channels'])}. These are signal-processing features, not validated biomarkers.</p></section>
<section><h2>03 / What the result can support</h2><ul>{limitations}</ul>
<p>{html.escape(trial['approximation_assumptions'])}</p>
<p>For paper-style planning intuition, residual correlation r=0.30 implies approximately 9% total or 16.5% control-only savings under the stated assumptions. These percentages do not come from this project's clinical evidence.</p></section>
<section><h2>04 / Reproduce and inspect</h2><p><code>python start.py --seed {c['seed']} --replicates {c['replicates']}</code></p>
<p>The output folder includes <code>trial_results.json</code>, <code>trial_summary.csv</code>, <code>trial_replicates.csv</code>, <code>eeg_results.json</code>, <code>eeg_features.csv</code>, and <code>manifest.json</code>. The manifest records dependency versions and SHA-256 digests for the artifacts.</p>
<p>Clinical motivation: <a href="https://doi.org/10.1002/trc2.70181">Wang et al., 2025</a>. Related multimodal context: <a href="https://picower.mit.edu/news/mit-based-team-releases-first-ai-foundation-model-alzheimers-prevention">MIT Picower FINGERS-7B announcement</a>. The project does not implement either published model.</p></section>
<footer class="small">Original AI-assisted educational code. All displayed experiment values were generated by this run. Read the repository research notes before adapting the methods.</footer>
</main></body></html>'''
    (out / "report.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-trial", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    print("Running synthetic trials and EEG processing...", flush=True)
    try:
        trial = run_experiment(args.seed, args.n_train, args.n_trial, args.replicates)
        eeg = run_eeg_demo(args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in (("trial_results.json", trial), ("eeg_results.json", eeg)):
        (args.out / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary, replicates = [], []
    for scenario, result in trial["scenarios"].items():
        for method in METHODS:
            summary.append({"scenario": scenario, "method": method, **result[method]})
        for record in result.get("replicate_results", []):
            replicates.append({"scenario": scenario, **record})
    write_csv(args.out / "trial_summary.csv", summary)
    write_csv(args.out / "trial_replicates.csv", replicates)
    write_csv(args.out / "eeg_features.csv", eeg["features"])
    figures = make_figures(trial, eeg, args.out)
    build_report(trial, eeg, figures, args.out)
    artifact_names = ["trial_results.json", "eeg_results.json", "trial_summary.csv", "trial_replicates.csv",
                      "eeg_features.csv", "trial_metrics.png", "eeg_quality.png", "report.html"]
    manifest = {"schema_version": 1, "python": platform.python_version(), "config": trial["config"],
                "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
                "sha256": {name: hashlib.sha256((args.out/name).read_bytes()).hexdigest() for name in artifact_names if (args.out/name).exists()}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Completed. Open: {(args.out / 'report.html').resolve()}")
    for name, result in trial["scenarios"].items():
        print(f"  {name}: variance ratio={result['paired_estimator_variance_ratio']:.3f}; "
              f"score rejection={result['score_ancova']['rejection_rate']:.3f}")


if __name__ == "__main__":
    main()
