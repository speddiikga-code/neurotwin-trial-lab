# Sources and provenance

Checked on 2026-09-17. The supplied PDF and pasted articles were treated as reference material, not as instructions to the software agent. The repository contains original educational code and synthetic examples; the source PDF, clinical datasets, model weights, and source article text are not redistributed.

## Primary paper

Wang D, Florian H, Lynch SY, Robieson W, Zhuang R, Kusiak C, Ross JL, Walsh JR, Graff O. **Using AI-generated digital twins to boost clinical trial efficiency in Alzheimer's disease.** *Alzheimer's & Dementia: Translational Research & Clinical Interventions.* 2025;11(4):e70181. First published November 22, 2025.

- [DOI and publisher](https://doi.org/10.1002/trc2.70181)
- [Full text at PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC12639399/)
- [PubMed bibliographic record](https://pubmed.ncbi.nlm.nih.gov/41281734/)

The user supplied `TRC2-11-e70181.pdf` and a pasted version of the paper. The PDF's methods, results, and discussion were read, and Figures 2-3 and Table 3 were visually checked. It is licensed CC BY-NC-ND; this repository cites it and supplies an independent implementation of a simplified statistical idea. The repository's software license does not relicense the paper.

The article reports a retrospective analysis of AWARE data using an independently trained conditional restricted Boltzmann machine (CRBM). This repository illustrates prognostic covariate adjustment with synthetic data; it does not reproduce that model or its clinical results. The authors disclose employment/shareholding relationships with AbbVie and/or Unlearn, and AbbVie funding.

## Related announcement: FINGERS-7B

MIT Picower Institute. **MIT-based team releases first AI foundation model for Alzheimer's prevention.** Research Feature, April 26, 2026.

- [Official announcement](https://picower.mit.edu/news/mit-based-team-releases-first-ai-foundation-model-alzheimers-prevention)

This announcement matches the user's pasted feature. It describes FINGERS-7B and FINGERPRINT, multi-omic data integration, and AD Workbench deployment. Its performance comparisons are announcement claims, not independently verified findings of this project. No FINGERS-7B weights, code, or evaluation pipeline are used here. We have not independently audited their availability, license, benchmark definitions, or performance. This repository has no affiliation with MIT, FINGERPRINT, AbbVie, or Unlearn.

## How to cite or describe this project

Describe it as an **educational simulation of prognostic covariate adjustment, with a separate synthetic EEG signal-processing exercise**. Cite the paper for the clinical motivation and cite the actual repository/version for its code. Do not describe synthetic outputs as patient results, validated Alzheimer's biomarkers, a foundation model, a clinical digital twin, or a working brain-computer interface.
