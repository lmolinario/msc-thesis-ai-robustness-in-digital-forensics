# Dataset Card

## Dataset Name

**FAIR-Lab Forensic Image Robustness Dataset**

This dataset card describes the dataset artifacts used in the MSc thesis repository:

```text
msc-thesis-ai-robustness-in-digital-forensics
```

The dataset supports the experimental evaluation of AI-based image classification systems in a Digital/Computer Forensics setting. Its purpose is not to provide a generic computer-vision benchmark, but to support an operational robustness analysis of automated image triage under clean, out-of-distribution, adversarial, and anti-forensic conditions.

---

## Dataset Scope

The dataset is designed for a binary forensic image-classification task with a separate out-of-distribution evaluation layer.

Primary binary task:

```text
non_weapon vs weapon
```

Additional operational evaluation category:

```text
ood
```

The `ood` category is not treated as a third training class. It is used as a separate robustness and operational-risk condition to evaluate how classifiers behave when exposed to borderline, anomalous, synthetic, degraded, or semantically out-of-distribution images.

---

## Official Frozen Dataset

The official frozen dataset manifest is:

```text
datasets/final/manifests/manual_selection_final_1500.csv
```

Final class distribution:

| Class | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |
| `ood` | 500 |
| **Total** | **1500** |

The official binary subset used for clean split generation and perturbation generation is:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

Binary subset distribution:

| Class | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |
| **Total** | **1000** |

OOD samples are excluded from adversarial attack generation and are evaluated separately.

---

## Source Groups

The dataset was constructed from heterogeneous image sources to better approximate variability encountered in forensic triage scenarios.

| Source group | Role |
|---|---|
| `01_kaggle_weapon` | Public dataset source for weapon-related imagery. |
| `02_deepfirearm` | Firearm-oriented source used to increase intra-class weapon diversity. |
| `03_google_scraped` | Web-scraped images used to include broader visual variability. |
| `04_telegram_youtube` | Social/video-platform-derived imagery and thumbnails. |
| `05_deepweb` | Deep-web-oriented acquisition source used for methodological heterogeneity. |

The source groups are used as acquisition/provenance categories. They do not define the final labels by themselves. Final labels are assigned through a human-in-the-loop semantic review protocol.

---

## Label Definitions

### `weapon`

Images assigned to `weapon` contain a real firearm or firearm-like object sufficiently visible and semantically relevant to the binary task. The class is intended to represent images that an automated forensic triage system should flag as potentially relevant for weapon-related review.

Typical inclusion criteria:

- firearm clearly visible;
- visually interpretable image;
- object sufficiently central or recognizable;
- consistent with the intended forensic triage task.

Typical exclusion criteria:

- extremely small or unreadable object;
- ambiguous or visually indeterminate object;
- toy, replica, videogame, synthetic, or non-realistic weapon-like imagery when better treated as `ood`;
- images whose relevance depends on contextual assumptions not visible in the image.

### `non_weapon`

Images assigned to `non_weapon` are realistic negative samples that do not contain firearms and remain compatible with the visual domain of the binary task.

Typical inclusion criteria:

- no firearm visible;
- realistic image content;
- suitable negative class for binary classification;
- not intentionally ambiguous or out-of-distribution.

### `ood`

Images assigned to `ood` are out-of-distribution, borderline, ambiguous, synthetic, degraded, or semantically external to the binary `weapon`/`non_weapon` task.

Typical examples include:

- knives, swords, or non-firearm weapons;
- toys, replicas, or airsoft-like content;
- videogame, CGI, synthetic, or cartoon imagery;
- military scenes, vehicles, missiles, explosions, or war-related imagery not corresponding to the binary firearm task;
- extremely degraded, anomalous, or ambiguous images.

The `ood` class is used to evaluate operational robustness and false-positive behavior, not to train the binary proxy models.

---

## Dataset Construction Workflow

The dataset workflow follows a traceable human-in-the-loop methodology.

```text
datasets/scripts/acquisition/
    ↓
datasets/raw/
    ↓
datasets/scripts/prepared/08_build_prepared_dataset.py
    ↓
datasets/prepared/final_pool/
    ↓
datasets/scripts/prepared/09_generate_review_manifest_full.py
    ↓
datasets/prepared/manifests/review_manifest_full.csv
    ↓
datasets/scripts/final/10_manual_selection_protocol_reviewer.py
    ↓
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

The prepared dataset stage performs technical filtering, validation, hashing, and exact-duplicate removal. The final semantic selection is performed through a manual review protocol and exported into frozen manifests.

---

## Human-in-the-Loop Review

The final dataset is not the result of fully automatic labeling. It is produced through a controlled manual review process.

The review protocol is implemented by:

```text
datasets/scripts/final/10_manual_selection_protocol_reviewer.py
```

The review process records:

- final semantic labels;
- selected and removed samples;
- review state;
- review logs;
- class counts;
- source provenance;
- stable image identifiers.

Main review artifacts:

```text
datasets/final/manifests/manual_selection_protocol_db.csv
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/final/manifests/manual_selection_removed.csv
datasets/final/reports/manual_selection_log.csv
datasets/final/reports/manual_selection_state.json
datasets/final/reports/manual_selection_summary.json
```

Human review is considered a methodological feature of the forensic pipeline, not an implementation weakness. It reflects the need to document and control semantic relevance in forensic AI evaluation.

---

## Splits

Clean binary folds and the OOD evaluation set are generated from the frozen manifests.

Official split-generation script:

```text
datasets/scripts/splits/11_generate_clean_and_ood_splits.py
```

Clean split manifest:

```text
datasets/splits/manifests/clean_folds_manifest.csv
```

OOD evaluation manifest:

```text
datasets/splits/manifests/ood_eval_manifest.csv
```

Clean binary folds:

```text
datasets/splits/clean/fold_1/
datasets/splits/clean/fold_2/
datasets/splits/clean/fold_3/
datasets/splits/clean/fold_4/
datasets/splits/clean/fold_5/
```

Each fold contains 200 clean binary images:

| Class | Count per fold |
|---|---:|
| `weapon` | 100 |
| `non_weapon` | 100 |

The OOD evaluation set is stored separately and is not split into training folds.

---

## Perturbed and Forensic Evaluation Artifacts

The clean binary subset is used to generate adversarial and anti-forensic artifacts.

Adversarial attacks:

```text
fgsm
superdeepfool
sigma_zero
one_pixel
color_shift
```

Anti-forensic transformations:

```text
jpeg_recompression
resample_resize
gaussian_blur
histogram_modification
contrast_stretching
```

The forensic evaluation bundle is generated by:

```text
datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py
```

Bundle location:

```text
datasets/forensic_evaluation_bundle/
```

Bundle composition:

| Condition | Files |
|---|---:|
| Clean | 1000 |
| OOD | 500 |
| Adversarial | 5000 |
| Anti-forensic | 5000 |
| **Total** | **11500** |

The bundle includes a blind input folder for commercial forensic tools and separate metadata for traceability and post-export normalization.

Tool input directory:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

Internal metadata directory:

```text
datasets/forensic_evaluation_bundle/metadata/
```

The blind tool input directory must not contain labels, attack names, source identifiers, or fold information in the file paths.

---

## Intended Uses

This dataset is intended for:

- forensic AI robustness evaluation;
- automated image triage experiments;
- comparison of transparent proxy models and commercial forensic AI tools;
- study of operational degradation under adversarial and anti-forensic manipulations;
- analysis of false negatives, false positives, and OOD behavior;
- forensic traceability and auditability experiments.

---

## Out-of-Scope Uses

This dataset is not intended for:

- operational deployment as a law-enforcement detection system;
- autonomous evidentiary decision-making;
- biometric identification;
- face recognition;
- person identification;
- weapon detection in real-time surveillance systems;
- training a production-grade safety-critical detector;
- replacing human forensic review.

Outputs produced from this dataset must be interpreted as experimental and methodological results.

---

## Traceability and Integrity

The dataset pipeline uses file-level hashing and manifest-based traceability.

Primary integrity mechanism:

```text
SHA256
```

Additional forensic-tool compatibility hash:

```text
MD5
```

Traceability artifacts include:

- prepared metadata;
- review manifests;
- final frozen manifests;
- split manifests;
- attack manifests;
- forensic bundle manifest;
- bundle hash mapping;
- normalized forensic-tool predictions.

The main bundle metadata files are:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
datasets/forensic_evaluation_bundle/metadata/bundle_hashes_sha256.csv
datasets/forensic_evaluation_bundle/metadata/bundle_summary.json
```

---

## Ethical, Legal, and Distribution Considerations

The repository may contain scripts, manifests, metrics, documentation, selected generated artifacts, and methodological outputs. Raw images, generated image corpora, and forensic-tool exports may be subject to third-party terms, ethical restrictions, legal constraints, institutional handling requirements, or source-specific limitations.

The dataset should therefore be treated as a research artifact whose redistribution status depends on the source and on the specific file category.

General principle:

```text
Code and documentation may be reusable according to the repository license.
Images, raw datasets, forensic exports, generated perturbations, and model checkpoints require separate verification before redistribution.
```

---

## Known Limitations

- The final semantic labels depend on a documented human-in-the-loop review protocol.
- Source distributions are heterogeneous and not intended to represent a statistically complete real-world distribution.
- The dataset focuses on an operational forensic triage scenario, not on exhaustive weapon taxonomy.
- OOD samples are intentionally heterogeneous and should not be interpreted as a conventional closed-set class.
- Adversarial and anti-forensic artifacts are generated according to the thesis protocol and do not exhaust all possible manipulation strategies.
- Commercial forensic-tool evaluation depends on tool version, license, enabled modules, export format, and normalization quality.

---

## Citation

Citation details will be added upon thesis completion.

Until then, cite the repository and the corresponding MSc thesis when referring to the dataset, methodology, or results.
