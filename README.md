# MSc Thesis – AI Robustness in Digital Forensics

## Overview

The repository mirrors a structured and reproducible experimental pipeline designed to evaluate the robustness of AI-based systems under controlled yet realistic perturbation scenarios.

The study is conducted under realistic experimental conditions, where AI models are exposed to both adversarial and anti-forensic perturbations designed to degrade classification performance while preserving visual plausibility.

---

## Research Objective

The primary objective of this research is to assess the reliability and robustness of machine learning models when deployed in forensic workflows, particularly in scenarios where input data may be intentionally manipulated.

The work aims to answer the following research questions:

- How do AI-based image classification systems behave under adversarial perturbations?
- To what extent do anti-forensic transformations impact model predictions?
- Are current AI tools reliable enough for forensic and investigative use?
- What are the limitations of existing robustness evaluation practices in this domain?

---


## Contributions

This work provides the following contributions:

- A structured experimental pipeline for evaluating the robustness of AI-based image classification systems in digital forensic scenarios.
- A realistic dataset construction methodology combining heterogeneous sources (e.g., public datasets, OSINT, and web scraping).
- A comparative evaluation of multiple AI models under both adversarial and anti-forensic perturbations.
- An analysis of the impact of visually plausible manipulations on classification reliability.
- An explainability-driven investigation of model behavior under perturbed inputs.

---
  
## Pipeline Overview

The repository mirrors the methodological structure of the experimental pipeline and is organized into the following stages:

1. Raw data collection and source organization  
2. Dataset validation, normalization, and preparation  
3. Global deduplication and metadata generation  
4. Balanced dataset split construction  
5. Baseline model evaluation on clean data  
6. Adversarial and anti-forensic perturbation generation  
7. Robustness evaluation across multiple folds  
8. Explainability analysis (e.g., saliency maps, attribution methods)  
9. Aggregation of publication-ready results  

---

## Repository Structure

The repository is structured to reflect the full experimental workflow:

- `datasets/` – Dataset organization, preparation outputs, and reports  
- `src/` – Core pipeline scripts (dataset, models, attacks, evaluation)  
- `experiments/` – Experiment configurations and execution stages  
- `results/` – Evaluation outputs and aggregated results  
- `figures/` – Visual assets for analysis and publication  
- `tables/` – Structured experimental results  
- `docs/` – Methodological documentation  
- `progress/` – Research log and milestones  

---

## Dataset Availability

Due to legal, ethical, and operational constraints, the full dataset used in this study cannot be publicly released.

However, this repository provides:

- Dataset structure and schema
- Metadata examples
- Sample images where permissible
- Detailed documentation of the dataset construction process

This ensures methodological transparency while respecting data governance constraints typical of forensic contexts.

---


## Threat Model

The experimental setup assumes a threat model in which an adversary can manipulate input images with the goal of inducing misclassification, while maintaining high visual plausibility.

Two classes of perturbations are considered:

- Adversarial attacks: small, often imperceptible perturbations optimized to exploit model vulnerabilities.
- Anti-forensic transformations: realistic image manipulations (e.g., compression, resizing, filtering) that may occur in real-world forensic scenarios.

The attacker is assumed to have limited or no knowledge of the internal model parameters (black-box or gray-box setting).

--- 

## Reproducibility

This repository is designed to ensure methodological transparency and partial reproducibility of the experimental pipeline.

Certain components (e.g., data acquisition from restricted sources) have been abstracted or replaced with placeholders.

---

## Research Context

This work is developed within the MSc program in:

> Computer Engineering, Cybersecurity and Artificial Intelligence

and is aligned with current research topics in:

- Adversarial Machine Learning  
- AI Security  
- Digital Forensics  
- Robustness Evaluation  

---

## Status

🚧 Work in progress  

The repository is continuously updated as the research progresses.

---

## Citation

If you use or reference this work, please cite the thesis (details will be added upon completion).

---

## License

To be defined (recommended: MIT for code, CC BY 4.0 for documentation).
