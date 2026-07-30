# Dataset Pipeline Scripts

This directory contains the executable dataset pipeline, the adversarial and
anti-forensic generation components, and the internal modules that connect the
dataset artifacts to the proxy models and the forensic evaluation bundle.

The documentation below describes **every Python file currently present under
`datasets/scripts/`**. It distinguishes:

- official numbered pipeline entry points;
- supplementary public-audit utilities;
- internal modules imported by the entry points;
- empty `__init__.py` package markers.

The canonical description of the dataset itself remains
[`../DATASET_CARD.md`](../DATASET_CARD.md). The higher-level dataset layout,
artifact inventory, and controlled-data policy are documented in
[`../README.md`](../README.md).

## Operating boundary

The repository tracks manifests, reports, hashes, metadata, normalized outputs,
and documentation. Raw image corpora and generated image directories are
normally restored or regenerated locally and are intentionally excluded from
`main`.

Run commands from the repository root so that package imports and
repository-relative paths are resolved consistently:

```bash
python datasets/scripts/<stage>/<script>.py --help
```

Several entry points can overwrite generated directories when `--force` is
used. Review the selected paths before enabling that option. Acquisition scripts
that access external services also require the relevant credentials, network
access, and authorization.

## Pipeline map

| Step | File | Type | Main role |
|---:|---|---|---|
| 00 | `acquisition/00_download_raw_datasets_bundle.py` | Official entry point | Restore the controlled raw archive or the exact frozen evaluation bundle |
| 01 | `acquisition/01_download_kaggle.py` | Official entry point | Download and index the Kaggle weapon dataset |
| 02 | `acquisition/02_download_github.py` | Official entry point | Download and index the DeepFirearm GitHub repository |
| 03 | `acquisition/03_build_subset_deepfirearm.py` | Official entry point | Build a deterministic class-limited DeepFirearm subset |
| 04 | `acquisition/04_scrape_google.py` | Official entry point | Collect weakly labelled candidates from Google Images |
| 05 | `acquisition/05_scrape_telegram.py` | Official entry point | Collect public-channel Telegram photographs |
| 06 | `acquisition/06_scrape_youtube.py` | Official entry point | Collect YouTube thumbnails with a Google fallback |
| 07 | `acquisition/07_scrape_deepweb.py` | Official entry point | Collect candidate images from Ahmia-indexed onion pages through Tor |
| 08 | `prepared/08_build_prepared_dataset.py` | Official entry point | Validate, deduplicate, rename, and index the prepared image pool |
| 09 | `prepared/09_generate_review_manifest_full.py` | Official entry point | Create the full manifest used by the human review stage |
| 10 | `final/10_manual_selection_protocol_reviewer.py` | Official entry point | Run and resume the human-in-the-loop final selection protocol |
| 11 | `splits/11_generate_clean_and_ood_splits.py` | Official entry point | Build five clean binary folds and a separate OOD set |
| 13 | `attacks/13_generate_anti_forensic_attacks.py` | Official entry point | Generate five model-agnostic anti-forensic transformations |
| 14 | `attacks/14_generate_adversarial_attacks.py` | Official entry point | Generate fold-aware adversarial and adversarial-style images |
| 16 | `bundle/16_build_forensic_evaluation_bundle.py` | Official entry point | Build the blind, hash-traceable forensic evaluation bundle |
| 17 | `bundle/17_build_public_embedded_metadata_audit.py` | Supplementary utility | Minimize the detailed metadata audit for public release |

Pipeline numbers `12` and `15` are intentionally absent from this directory.
Step `12` belongs to the model-training pipeline under `models/`; step `15`
belongs to the forensic-tool evaluation pipeline under `evaluation/`.

## Artifact flow

```text
controlled/raw sources
        |
        v
00-07 acquisition and source indexing
        |
        v
08 technical validation + global exact deduplication
        |
        v
09 full review manifest
        |
        v
10 human-in-the-loop final selection
        |
        +----------------------------+
        |                            |
        v                            v
11 clean binary folds          11 separate OOD set
        |
        +---------------+
        |               |
        v               v
13 anti-forensic    14 adversarial generation
        \               /
         \             /
          v           v
        16 blind forensic evaluation bundle
                    |
                    v
        17 privacy-minimized public metadata audit
```

## Cross-cutting interpretation rules

### Acquisition labels are not ground truth

The category names used by Google, YouTube, Telegram, or deep-web collection
scripts are acquisition buckets or weak supervision signals. They do not become
final labels automatically. Semantic labels are assigned only during the
documented human review performed at step 10.

### OOD is a separate evaluation layer

`ood` is not trained or evaluated as a third supervised class in the binary
proxy task. Step 11 separates the 500 OOD images into their own evaluation set;
only the 500 `weapon` and 500 `non_weapon` images enter the five binary folds and
the perturbation-generation stages.

### Hash roles

SHA-256 is the primary integrity and traceability mechanism. MD5 is retained
where interoperability with forensic tools or historical acquisition logic
requires it. When MD5 is used for local deduplication or naming, it must not be
interpreted as the primary cryptographic integrity control.

### Probability terminology

Some historical manifests and internal dictionaries contain field names such as
`confidence`, `auto_confidence`, or `label_confidence`. For model outputs these
values are softmax-like predicted-class probabilities or maximum predicted-class
probabilities, not evidence of calibration unless a separate calibration
procedure is explicitly documented.

---

# Acquisition

## `acquisition/00_download_raw_datasets_bundle.py`

### Purpose

This is the controlled-restoration entry point. It supports two distinct
artifacts:

- `raw`: the source datasets used to rebuild the prepared image pool;
- `frozen`: the exact 11,500-file forensic evaluation bundle used in the
  completed experiments.

It is the preferred restoration path when a researcher has received an
authorized archive rather than regenerating all image corpora from external
sources.

### Inputs and access mechanisms

The artifact is selected with:

```bash
--artifact raw
--artifact frozen
```

The archive can be supplied directly with `--archive`, downloaded from an
authorized URL supplied with `--url`, or resolved from the environment variables
defined by the script. `--request-access` opens or prints the controlled-access
request page before restoration.

The expected archive digest is normally read from:

```text
docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256
```

An explicit value can be supplied with `--expected-sha256`.

### Processing

The script:

1. validates that the input is a readable ZIP archive;
2. tests the archive for corrupted members;
3. rejects symbolic links and unsafe traversal paths;
4. computes and verifies the complete archive SHA-256;
5. extracts only into the expected repository locations;
6. optionally replaces existing local outputs when the corresponding force
   option is supplied.

For a frozen restoration, it also verifies the expected 11,500-file profile and
checks every restored file against
`datasets/forensic_evaluation_bundle/metadata/bundle_hashes_sha256.csv`.

### Outputs

- raw restoration under `datasets/raw/`; or
- the local image views of `datasets/forensic_evaluation_bundle/`.

Committed canonical metadata are preserved. The script restores the controlled
image payloads and does not redefine the frozen labels, mappings, or metrics.

### Important options

- `--force-download`: replace a previously downloaded archive;
- `--force-extract`: replace the corresponding local extracted payload;
- `--skip-content-verification`: skip frozen per-file verification.

The last option weakens the restoration check and should be used only for a
specific diagnostic reason that is recorded separately.

### Examples

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact raw \
  --archive "/path/to/00_raw_datasets_bundle.zip"
```

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact frozen \
  --archive "/path/to/16_frozen_forensic_evaluation_bundle.zip"
```

---

## `acquisition/01_download_kaggle.py`

### Purpose

Downloads the configured Kaggle dataset, extracts it into the raw-data area, and
creates a file-level index. The default dataset identifier is:

```text
snehilsanyal/weapon-detection-test
```

### Inputs and prerequisites

The default destination is:

```text
datasets/raw/01_kaggle_weapon/
```

The Kaggle API must be installed and authenticated for automatic download.
Alternatively, an existing ZIP placed in the output directory can be reused and
extracted.

### Processing

The script avoids unnecessary downloads when images already exist, can reuse an
existing ZIP, and optionally preserves the archive. After acquisition it scans
all files recursively and records:

- relative path;
- filename and extension;
- file size;
- whether the extension is recognized as an image;
- SHA-256.

### Outputs

```text
datasets/raw/01_kaggle_weapon/
datasets/raw/01_kaggle_weapon/download_summary.csv
```

The summary indexes both images and auxiliary dataset files. `is_image` is based
on a supported-extension check; it is not a semantic or forensic validation.

### Main options

- `--dataset`;
- `--output-dir`;
- `--summary-csv`;
- `--keep-zip`;
- `--force-extract`;
- `--verbose`.

### Example

```bash
python datasets/scripts/acquisition/01_download_kaggle.py --verbose
```

---

## `acquisition/02_download_github.py`

### Purpose

Downloads a public GitHub repository as a ZIP archive and produces the same
file-level provenance index used by the Kaggle acquisition stage. The default
source is:

```text
jdhao/deep_firearm
```

### Inputs and branch resolution

The script accepts an `owner/repository` identifier. If `--branch` is omitted,
it attempts `main` and then `master`. The HTTP transfer is streamed with a
configurable timeout.

### Processing

The script:

1. checks whether a prepared repository copy already exists;
2. reuses a local ZIP when possible;
3. downloads the selected branch archive;
4. extracts the GitHub-generated root directory;
5. moves the content into a stable sanitized directory name;
6. regenerates a SHA-256 file index.

### Outputs

```text
datasets/raw/02_deepfirearm/
datasets/raw/02_deepfirearm/download_summary.csv
```

### Main options

- `--repo`;
- `--branch`;
- `--output-dir`;
- `--summary-csv`;
- `--keep-zip`;
- `--force`;
- `--timeout`;
- `--verbose`.

### Example

```bash
python datasets/scripts/acquisition/02_download_github.py \
  --repo jdhao/deep_firearm \
  --branch master
```

---

## `acquisition/03_build_subset_deepfirearm.py`

### Purpose

Builds a deterministic subset from a class-folder-organized DeepFirearm
collection. It limits the total number of images and the contribution of each
eligible class while preserving reproducibility through a fixed seed.

### Default input and output

```text
Input:
datasets/raw/02_deepfirearm/train/

Output:
datasets/raw/02_deepfirearm/subset_choosen/
datasets/raw/02_deepfirearm/subset_summary.csv
```

The historical directory name `subset_choosen` is retained by the implementation
for compatibility.

### Selection policy

For each sorted class directory:

- classes with fewer than `--min-per-class` source images are excluded;
- at most `--max-per-class` images are sampled;
- selection stops at `--max-total-images`;
- sampling uses `random.Random(--seed)`.

Defaults are 1,000 total images, 200 per class, a minimum source-class size of
20, and seed 42.

Images are copied with metadata preservation through `shutil.copy2`. The output
summary records SHA-256 values for the selected files.

### Main options

- `--source-dir`;
- `--dest-dir`;
- `--summary-csv`;
- `--max-total-images`;
- `--max-per-class`;
- `--min-per-class`;
- `--seed`;
- `--force`;
- `--verbose`.

### Methodological limitation

This operation controls sampling size and class contribution, but it does not
validate the semantic correctness of the source class folders. Final thesis
labels still depend on step 10.

---

## `acquisition/04_scrape_google.py`

### Purpose

Collects heterogeneous candidate images from Google Images using a fixed mapping
of weapon-related queries. The categories cover contexts such as concealed
weapons, firearms in bags or vehicles, toy and airsoft objects, holstered or
damaged firearms, and people holding or pointing firearms.

### Processing

For each category the script:

1. creates a category-specific directory;
2. invokes `GoogleImageCrawler`;
3. applies crawler-level minimum dimensions;
4. opens each downloaded file with Pillow;
5. removes unreadable or corrupted images;
6. stops when the configured total quota is reached;
7. regenerates a recursive SHA-256 summary.

### Defaults

```text
Output:
datasets/raw/03_google_scraped/dataset_scraping_google/

Summary:
datasets/raw/03_google_scraped/google_scraping_summary.csv

Maximum total valid images: 1000
Maximum requested per category: 125
Minimum size: 200 x 200
```

### Main options

- `--output-dir`;
- `--summary-csv`;
- `--max-total-images`;
- `--max-per-class`;
- `--min-width`;
- `--min-height`;
- `--force`;
- `--verbose`.

### Methodological limitation

A query/category match is only a candidate-source signal. Search results may
contain duplicates, irrelevant images, ambiguous objects, or semantic drift.
The directory name must never be interpreted as final ground truth.

---

## `acquisition/05_scrape_telegram.py`

### Purpose

Collects photo media from a predefined set of accessible public Telegram
channels and indexes the downloaded files. The current source list is embedded
in the script and represents source provenance, not class labels.

### Credentials

The script requires:

```text
TELEGRAM_API_ID
TELEGRAM_API_HASH
```

The optional environment variable `TELEGRAM_SESSION_NAME` controls the local
Telethon session name. Credentials and session files must remain local and must
not be committed.

### Processing

For each configured channel, the script inspects up to 200 messages, retains
only `MessageMediaPhoto` entries, stores each image under the channel directory
using the Telegram message identifier, and avoids downloading files that already
exist.

Unlike several other acquisition entry points, this script does not expose an
`argparse` CLI. Its output paths, source channels, and message limit are defined
in the module.

### Outputs

```text
datasets/raw/04_telegram_youtube/osint_telegram/
datasets/raw/04_telegram_youtube/telegram_download_summary.csv
```

### Methodological limitation

The channel name is a source-domain attribute. It is not a semantic label and
does not establish that an individual image contains a weapon.

---

## `acquisition/06_scrape_youtube.py`

### Purpose

Collects YouTube video thumbnails for predefined scenario queries. When a
category returns no valid thumbnail, it falls back to Google Images so that an
empty source query does not interrupt the broader acquisition pipeline.

### Processing

For each category the script:

1. submits a `yt_dlp` search;
2. obtains thumbnail URLs without downloading videos;
3. downloads candidate thumbnail files;
4. validates them with Pillow;
5. deletes malformed files;
6. invokes Google Images only when the number of valid YouTube thumbnails is
   zero.

The predefined categories include CCTV firearm incidents, guns in bags,
toy/airsoft objects, police shooting training, people with guns, and crime-scene
weapons.

### Defaults and outputs

```text
YouTube search results per category: 100
Google fallback maximum: 100
Fallback minimum size: 200 x 200

Output:
datasets/raw/04_telegram_youtube/osint_youtube/

Summary:
datasets/raw/04_telegram_youtube/youtube_download_summary.csv
```

This script also uses fixed module configuration rather than an `argparse` CLI.

### Traceability caveat

The current folder layout preserves the query category but does not encode in
each filename whether the retained file came from YouTube or the Google
fallback. Later manifests preserve the broader source dataset, while semantic
validity is established through human review.

---

## `acquisition/07_scrape_deepweb.py`

### Purpose

Collects candidate images from Ahmia-indexed `.onion` pages through a locally
available Tor SOCKS proxy.

### Network flow

Ahmia search pages are queried over the surface web. Candidate onion URLs are
then visited through a `requests.Session` configured with `socks5h`. The default
proxy is:

```text
127.0.0.1:9050
```

### Processing

The script:

1. runs a fixed set of firearm-related search queries;
2. extracts onion redirect targets from Ahmia;
3. avoids revisiting duplicate onion host/path combinations;
4. extracts candidate `<img>` URLs;
5. filters obvious logos, banners, icons, sprites, avatars, and captchas;
6. checks HTTP content type;
7. validates image bytes and minimum dimensions before writing;
8. uses an MD5 content fingerprint for local filename generation and exact
   duplicate avoidance;
9. records SHA-256 in the final dataset summary.

### Defaults and outputs

```text
Maximum onion links per query: 20
Maximum images per page: 15
Minimum size: 200 x 200
Delay between onion visits: 3 seconds

Output:
datasets/raw/05_deepweb/deepweb_dataset/

Summary:
datasets/raw/05_deepweb/deepweb_scraping_summary.csv
```

### Main options

- `--output-dir`;
- `--summary-csv`;
- `--max-links-per-query`;
- `--max-images-per-page`;
- `--min-width`;
- `--min-height`;
- `--sleep-seconds`;
- `--tor-host`;
- `--tor-port`;
- `--force`;
- `--verbose`.

### Methodological and operational limitation

The query directory is only a source bucket. The script performs no semantic
annotation and cannot guarantee relevance, legality, availability, or
repeatability of external onion content. It must be run only within the
researcher's authorization and institutional rules, and all retained candidates
remain subject to manual review.

---

# Prepared dataset

## `prepared/08_build_prepared_dataset.py`

### Purpose

Creates the technical prepared image pool from the raw source directories. This
is the first stage that consolidates all acquisition sources into a common,
stable image namespace.

It deliberately performs **technical curation only**. It does not assign
`weapon`, `non_weapon`, or `ood` labels.

### Input sources

By default it scans:

```text
datasets/raw/01_kaggle_weapon/
datasets/raw/02_deepfirearm/
datasets/raw/03_google_scraped/
datasets/raw/04_telegram_youtube/
datasets/raw/05_deepweb/
```

Missing source directories are reported and skipped.

### Processing

For every candidate image extension, the script:

1. verifies that Pillow can decode the file;
2. checks minimum dimensions, defaulting to 300 × 300;
3. computes SHA-256;
4. removes exact duplicates globally across all source datasets;
5. assigns a stable sequential identifier such as `img_00000001`;
6. copies the retained file to the prepared pool;
7. recomputes the copied SHA-256 and requires an exact match.

Global deduplication means that when identical bytes appear in multiple source
datasets, only the first deterministic occurrence is retained. The duplicate
report preserves both the retained and discarded provenance.

### Outputs

```text
datasets/prepared/final_pool/images/
datasets/prepared/final_pool/metadata.csv
datasets/prepared/final_pool/reports/invalid_images.csv
datasets/prepared/final_pool/reports/duplicates_discarded.csv
datasets/prepared/final_pool/reports/prepared_build_summary.json
```

`metadata.csv` records image identity, prepared path, source provenance,
dimensions, size, extension, validity, and SHA-256.

The public summary writes repository-relative paths. Deliberately external paths
are reduced to `<external>/<name>` so that committed reports do not leak a local
workstation path.

### Main options

- `--raw-root`;
- `--output-dir`;
- `--min-width`;
- `--min-height`;
- `--force`;
- `--verbose`.

The script refuses to reuse an existing output directory unless `--force` is
supplied.

---

## `prepared/09_generate_review_manifest_full.py`

### Purpose

Creates the official full review manifest from the technical metadata generated
at step 08. It is the bridge between technical image validation and semantic
human annotation.

### Input and output

```text
Input:
datasets/prepared/final_pool/metadata.csv

Output:
datasets/prepared/manifests/review_manifest_full.csv
```

The paths are fixed in the module and there is no command-line interface.

### Processing

The script validates the required technical columns, propagates image identity,
SHA-256, prepared filename, source provenance, dimensions, size, extension, and
validity, then introduces empty fields for later stages.

The added fields cover:

- optional automatic prelabelling metadata;
- manual final labels;
- review state and notes;
- reviewer identity and timestamp;
- exclusion reasons;
- OOD flags and notes.

Initial status values are:

```text
prelabel_status = pending
review_state    = pending
```

No prelabel model is executed by this script and no semantic class is assigned.
Historical columns containing `confidence` in their names are initialized only
for schema compatibility.

### Validation and summary

The script rejects missing required columns and empty inputs. After writing the
manifest it prints row counts, duplicate `image_id` counts, pending-state
counts, non-empty final-label counts, and the distribution by source dataset.

---

# Final human selection

## `final/10_manual_selection_protocol_reviewer.py`

### Purpose

Provides the graphical human-in-the-loop reviewer used to create and audit the
final frozen selection. It is the authoritative semantic decision stage of the
public pipeline.

### Input

```text
datasets/prepared/manifests/review_manifest_full.csv
```

On first execution the script initializes:

```text
datasets/final/manifests/manual_selection_protocol_db.csv
```

Subsequent sessions reopen that working database, allowing the review to resume.

### Operating modes

#### `review_pending`

Shows pending images from one selected source dataset. The reviewer can assign:

- `weapon`;
- `non_weapon`;
- `ood`;
- `exclude`.

Mouse and keyboard shortcuts support assignment, clearing, undo, navigation,
zoom, criteria display, and saving.

#### `review_selection`

Shows already assigned images from one selected class, optionally restricted to
one source. The reviewer can remove an assignment and return the image to the
pending pool. This mode supports explicit quality control of an existing
selection rather than only first-pass labelling.

### Class targets and criteria

The script enforces the global final targets:

```text
weapon      500
non_weapon  500
ood         500
```

It also displays soft source-balance targets of 100 images per source and class.
These source targets are advisory; they do not override semantic correctness.

The embedded criteria distinguish clean real firearms, realistic negatives,
boundary/OOD material such as replicas, CGI, knives, war scenes or degraded
images, and unusable exclusions.

### Persistence, traceability, and recovery

Reviewer actions are appended to:

```text
datasets/final/reports/manual_selection_log.csv
```

Session state is stored in:

```text
datasets/final/reports/manual_selection_state.json
```

CSV writes use a temporary file and atomic replacement. Existing databases are
backed up before replacement, and locked-file replacement is retried. The UI
supports undo within the current session and automatic export regeneration.

### Final outputs

```text
datasets/final/manifests/manual_selection_protocol_db.csv
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_removed.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/final/reports/manual_selection_log.csv
datasets/final/reports/manual_selection_state.json
datasets/final/reports/manual_selection_summary.json
```

The adversarial subset contains only the 1,000 final `weapon` and `non_weapon`
images. The 500 OOD images remain outside perturbation generation.

### Dependencies and execution

The reviewer requires pandas, Matplotlib, a functioning Tk GUI backend, and
local access to the prepared images.

```bash
python datasets/scripts/final/10_manual_selection_protocol_reviewer.py
```

It is an interactive application and exposes no `argparse` options.

---

# Split generation

## `splits/11_generate_clean_and_ood_splits.py`

### Purpose

Converts the final human-reviewed manifests into the exact clean binary folds
and separate OOD directory used by model evaluation and perturbation generation.

### Inputs

```text
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

Before copying, the script requires:

- exactly 1,500 final rows;
- exactly 500 `weapon`, 500 `non_weapon`, and 500 `ood`;
- exactly 1,000 binary-subset rows;
- exactly 500 rows for each binary label;
- unique `image_id` and SHA-256 values;
- existing source image files.

### Fold assignment

The 1,000 binary samples are assigned to five deterministic folds. With the
canonical configuration, every fold contains exactly:

```text
100 weapon + 100 non_weapon = 200 images
```

The hard class quota is enforced first. Source balance is a secondary objective:
within each class and source dataset, a deterministic source-specific shuffle is
used and the next sample is assigned to the currently least represented
eligible fold.

OOD images are not distributed across the five folds. All 500 are copied into a
single dedicated OOD evaluation set.

### Outputs

```text
datasets/splits/clean/fold_1/{weapon,non_weapon}/
...
datasets/splits/clean/fold_5/{weapon,non_weapon}/
datasets/splits/ood/ood_eval_set/ood/
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
datasets/splits/manifests/split_generation_summary.json
```

Every copied file receives SHA-256 and MD5 values. The copied SHA-256 must match
the value recorded in the final manifest.

### Main options

- `--final-manifest`;
- `--adversarial-subset`;
- `--n-folds`;
- `--seed`;
- `--force`;
- `--verbose`.

Existing split directories are protected unless `--force` is explicitly
supplied.

---

# Anti-forensic transformations

## `attacks/13_generate_anti_forensic_attacks.py`

### Purpose

Generates the five model-agnostic anti-forensic transformation families used in
the thesis and can optionally evaluate their effect on fold-aware
EfficientNet-B0 checkpoints.

### Input

Generation uses:

```text
datasets/splits/manifests/clean_folds_manifest.csv
```

The full canonical run applies every selected transformation to the same 1,000
binary clean images.

### Implemented transformations

| Name | Operation | Canonical/default parameter |
|---|---|---|
| `jpeg_recompression` | Save the RGB image as JPEG | quality 70 |
| `resample_resize` | Downscale, then restore the original dimensions with bicubic interpolation | scale 0.50 |
| `gaussian_blur` | Pillow Gaussian filter | radius 1.50 |
| `histogram_modification` | Per-channel histogram equalization | `ImageOps.equalize` |
| `contrast_stretching` | Automatic contrast stretching | cutoff 1.0% |

All outputs are JPEG. Non-recompression transformations are saved at quality 95.

### Generation outputs

```text
attacks/anti_forensic/<attack>/<fold>/<label>/<image_id>__<attack>.jpg
attacks/manifests/anti_forensic_attacks_manifest.csv
attacks/manifests/anti_forensic_generation_summary.json
```

The manifest records original and perturbed identifiers, fold, final label,
source provenance, paths, serialized transformation parameters, SHA-256, MD5,
size, extension, and creation time.

The generation summary checks expected counts, unique generated identifiers,
unique perturbed SHA-256 values, and per-attack/fold/label distributions. A full
five-transformation run over 1,000 clean images produces 5,000 derivatives.

### Optional proxy evaluation

The script supports:

- generation only;
- generation followed by evaluation;
- `--evaluate-only` using an existing anti-forensic manifest.

The built-in target-model option is `efficientnet_b0`. Fold-aware evaluation
requires `fold_1` through `fold_5` checkpoints and reports clean versus
transformed correctness, accuracy drop, manipulation-induced errors among
clean-correct images, weapon false negatives, non-weapon false positives, and
probability shifts.

These proxy measurements do not replace the later black-box forensic-tool
evaluation.

### Execution modes and options

Launching without arguments opens an interactive launcher. CLI mode includes:

- `--attack`;
- `--input-manifest`;
- transformation parameters;
- `--limit` for smoke tests;
- `--force`;
- `--evaluate` or `--evaluate-only`;
- `--checkpoint-path`;
- `--eval-batch-size`;
- `--device`;
- `--label-order`;
- `--verbose`.

Example:

```bash
python datasets/scripts/attacks/13_generate_anti_forensic_attacks.py \
  --attack jpeg_recompression resample_resize gaussian_blur \
  histogram_modification contrast_stretching \
  --force
```

---

# Adversarial generation

## `attacks/14_generate_adversarial_attacks.py`

### Purpose

This is the official fold-aware adversarial generation entry point. It keeps
dataset traversal, output construction, hashing, checkpoint traceability, and
manifest generation in one controlled workflow while delegating model and
reference-attack details to internal adapters.

### Input and checkpoint convention

```text
Input manifest:
datasets/splits/manifests/clean_folds_manifest.csv

Checkpoint layout:
models/checkpoints/<target_model>/fold_1.pt
...
models/checkpoints/<target_model>/fold_5.pt
```

The default target is `efficientnet_b0`. The code can construct adapters for
`efficientnet_b0`, `resnet18`, and the frozen-CLIP-encoder binary-head model, but
the frozen thesis attack corpus targets the configured fold-aware proxy stated
in the experiment records.

Before generation the script checks the input schema, duplicate identifiers,
source-file existence, and clean SHA-256/MD5 values. Checkpoint paths and
checkpoint SHA-256 values are written into model-dependent attack manifests.

### Implemented attacks

#### `fgsm`

A white-box untargeted Fast Gradient Sign Method perturbation. The default
epsilon is `8/255` in pixel space. The adapter computes the loss gradient with
respect to the normalized model input; the result is converted back to valid
pixel space and saved as lossless PNG.

#### `one_pixel`

A score-based, model-dependent one-pixel attack using SciPy Differential
Evolution over five variables: `x`, `y`, `R`, `G`, and `B`. It minimizes the
true-label predicted probability, uses a deterministic per-image seed derived
from the base seed, image identifier, and attack name, and stops early after a
successful misclassification.

#### `sigma_zero`

An untargeted sparse white-box attack connected to the reference
`adv_lib.attacks.sigma_zero` implementation through
`sigma_zero_reference_adapter.py`. The adapter keeps the optimizer in pixel
space `[0,1]`, supplies fold-specific logits, and records the pinned reference
commit and optimization parameters.

#### `superdeepfool`

A white-box, fold-dependent SDF(infinity,1) implementation supplied by
`superdeepfool_adapter.py`. It performs a DeepFool-style boundary search followed
by the SuperDeepFool projection step and records convergence and perturbation
metrics.

#### `color_shift`

A deterministic model-agnostic RGB/saturation/contrast transformation. It is
generated once per clean image, does not load a checkpoint, and is saved as
JPEG. It belongs to the adversarial-style branch of the experiment but must not
be described as a direct gradient attack.

### Output layout

Model-dependent outputs:

```text
attacks/adversarial/<attack>/<target_model>/<fold>/<label>/
```

Color Shift uses the model-agnostic target identifier in the same controlled
layout.

Model-dependent images are saved as PNG to avoid introducing additional lossy
encoding after perturbation. Color Shift is saved as JPEG.

Run-specific manifests prevent one attack run from overwriting another:

```text
attacks/manifests/adversarial_<attack>_<target_model>_manifest.csv
attacks/manifests/adversarial_<attack>_<target_model>_summary.json
```

The exact filename is constructed from the selected attack and target-model
combination. A Color Shift-only run uses the stable
`adversarial_color_shift_*` artifact names.

### Manifest contents

Each generated row records:

- original and generated identifiers;
- fold and final binary label;
- source provenance;
- clean and perturbed paths;
- attack family, attack name, target model, and dependency type;
- serialized parameters;
- checkpoint path and SHA-256 where applicable;
- original and adversarial predictions;
- historical predicted-class probability fields;
- attack success and correctness flags;
- SHA-256 and MD5 for clean and perturbed files;
- L0, L2, L-infinity, and mean absolute perturbation measurements;
- convergence/iteration information where available.

The probability fields are model scores and are not evidence of calibration.

### Execution modes

Launching the file without arguments starts an interactive menu with smoke-test
and full-run choices. CLI mode provides reproducible control over attacks,
models, checkpoint root, device, input size, attack parameters, output
replacement, and row limits.

Examples:

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack fgsm \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --force
```

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack color_shift \
  --force
```

### Interpretation limit

An attack generated against one proxy checkpoint is a direct robustness test of
that target checkpoint. Applying the resulting image to another proxy or a
commercial black-box tool measures transfer or operational sensitivity; it is
not a direct white-box attack against the non-target system.

---

# Internal adversarial modules

The following files are imported by step 14. They are not standalone pipeline
entry points and should not be executed as substitutes for
`14_generate_adversarial_attacks.py`.

## `attacks/adversarial_model_interface.py`

Defines the dependency-light contract shared by model-dependent attacks.

It contains:

- the official binary mapping `non_weapon -> 0`, `weapon -> 1`;
- supported target-model names;
- model-agnostic and model-dependent attack registries;
- immutable `TargetModelConfig`;
- the abstract `TargetModelAdapter`;
- validation and label-conversion helpers;
- expected-generation-count logic.

The abstract adapter requires model loading, preprocessing, prediction,
probability output, loss computation, and input-gradient computation. Heavy ML
libraries are intentionally not imported here, allowing the main pipeline to
validate configuration without immediately loading PyTorch or CLIP.

This module creates no images or manifests.

---

## `attacks/adversarial_torch_model_adapters.py`

Provides the concrete PyTorch implementations of the common model interface.

### Torchvision adapters

For ResNet18 and EfficientNet-B0 it reconstructs the binary architecture,
replaces the final classifier with two outputs, loads a strict checkpoint state
dictionary, applies ImageNet normalization, and exposes prediction, softmax
scores, cross-entropy loss, and input gradients.

Supported checkpoint forms include a raw state dictionary or dictionaries
containing `state_dict`, `model_state_dict`, or `model`.

### CLIP binary-head adapter

Implements the thesis proxy as:

```text
frozen CLIP visual encoder + trained two-class linear head
```

It is not zero-shot CLIP and it is not a fully fine-tuned CLIP model. The
default backbone is OpenCLIP ViT-B-32 with OpenAI weights. The visual encoder is
frozen, image features are normalized, and only the loaded binary head produces
the two-class logits.

### Dependency handling

PyTorch, torchvision, and OpenCLIP are imported lazily. Selecting a model
without its optional dependencies raises an explicit dependency error rather
than silently changing the model.

The factory `build_target_model_adapter()` returns an unloaded adapter; step 14
must call `load_model()` explicitly.

---

## `attacks/sigma_zero_reference_adapter.py`

Connects the FAIR-Lab proxy interface to the reference Sigma-Zero implementation
without reimplementing the optimizer.

The expected pinned dependency is the inspected commit:

```text
jeromerony/adversarial-library
b14f81a3e1c414a573b969b402c99e65bfe2ca33
```

The module imports the specific `sigma_zero.py` source directly to avoid
unnecessary optional visualization dependencies from the package-level attack
registry.

Its `PixelSpaceTargetModel` wraps a loaded FAIR-Lab adapter so that the reference
attack receives unnormalized tensors in `[0,1]` while the underlying proxy still
receives its correct model-specific normalization.

The adapter records:

- reference repository, function, and commit;
- optimization parameters;
- original and adversarial predictions and model scores;
- true-label probabilities;
- convergence;
- changed-pixel count;
- L0, L2, L-infinity, and mean absolute perturbation metrics.

The output is a PNG-ready RGB image plus JSON-serializable attack metadata. File
writing and manifest construction remain the responsibility of step 14.

---

## `attacks/superdeepfool_adapter.py`

Implements the paper-based internal SuperDeepFool SDF(infinity,1) procedure used
by the pipeline.

It does **not** vendor or claim to execute an official SuperDeepFool repository.
The implementation:

1. converts the proxy input between normalized model space and pixel space;
2. iteratively searches for a multiclass DeepFool boundary point;
3. applies the SuperDeepFool projection step;
4. verifies whether the projected point remains adversarial;
5. falls back to the boundary point when projection does not preserve
   adversariality;
6. records convergence and perturbation measurements.

The configurable limits include outer iterations, internal DeepFool iterations,
projection steps, and an optional number of candidate classes.

This module returns one adversarial image and its metadata to step 14. It does
not traverse datasets, select checkpoints, or write manifests independently.

---

# Forensic evaluation bundle

## `bundle/16_build_forensic_evaluation_bundle.py`

### Purpose

Builds the operational corpus imported into commercial black-box forensic tools.
It combines clean, OOD, adversarial, and anti-forensic records while separating
ground truth from tool-visible filenames and paths.

The script prepares the corpus only. It does not execute Magnet AXIOM/Magnet.AI,
Excire, Cellebrite Inseyets, Griffeye/T3K CORE, or any other forensic product.

### Inputs

By default it loads:

```text
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
attacks/manifests/adversarial_*_manifest.csv
attacks/manifests/anti_forensic_attacks_manifest.csv
```

Adversarial manifests are discovered by filename, excluding summaries and
evaluation files.

### Bias-control layouts

#### Blind tool input

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

This is the only directory intended for import into forensic tools. It is flat
and uses anonymous names such as:

```text
bundle_000001.jpg
bundle_000002.png
```

The script checks that blind filenames do not reveal labels, OOD status, attack
names, target models, or fold identifiers.

#### Structured audit view

```text
datasets/forensic_evaluation_bundle/structured_audit_view/
```

This optional internal view preserves semantic hierarchy for debugging and
traceability. It must not be imported into the evaluated tools.

#### Metadata

```text
datasets/forensic_evaluation_bundle/metadata/
```

This area preserves the mapping between anonymous bundle identifiers and
ground-truth labels, source images, perturbations, models, folds, and hashes. It
is used only after tool export for normalization and analysis.

### Outputs

```text
metadata/bundle_manifest.csv
metadata/bundle_hashes_sha256.csv
metadata/bundle_summary.json
metadata/embedded_metadata_audit.csv
metadata/embedded_metadata_audit_summary.json
blind_tool_input/files/
structured_audit_view/
```

For each item the script computes actual SHA-256 and MD5 values and compares the
actual SHA-256 with the source manifest when available.

### Embedded-metadata audit

Step 16 can inspect Pillow-readable image information and EXIF fields without
modifying files. It searches metadata keys and values for experiment-specific or
sensitive terms and writes a detailed local CSV. Because that detailed audit can
contain complete values or binary payload representations, step 17 is required
before publishing a privacy-minimized version.

The audit can also be rerun without rebuilding the bundle:

```bash
python datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py \
  --audit-metadata-only
```

### Main options

- `--clean-manifest`;
- `--ood-manifest`;
- `--attack-manifests-dir`;
- `--bundle-dir`;
- `--layout blind|structured|both`;
- `--copy-files` / `--no-copy-files`;
- `--limit`;
- `--force`;
- `--audit-metadata-only`;
- `--verbose`.

For the canonical black-box experiment, import only
`blind_tool_input/files/`.

---

## `bundle/17_build_public_embedded_metadata_audit.py`

### Purpose

Converts the detailed embedded-metadata audit produced by step 16 into a
privacy-minimized public artifact.

The detailed source may contain complete EXIF/XMP values, decoded strings, or
binary metadata representations that are useful locally but unnecessary for
public reproducibility. Step 17 retains only:

- anonymous `bundle_id`;
- suffix;
- metadata-presence flag;
- sensitive-term names and count;
- metadata key names.

### Default behavior

Without `--install`, reviewable outputs are written under:

```text
datasets/forensic_evaluation_bundle/metadata/.staging/
```

The source canonical audit is not replaced.

### Validation

The current frozen profile requires:

```text
11,500 audit rows
15 rows containing one or more sensitive-term hits
```

The script also requires unique, correctly formatted `bundle_######`
identifiers, the exact public schema, no NUL bytes, and no recognized local
absolute-path leakage.

These counts are frozen-artifact checks. A future experimental freeze with a
different bundle requires deliberate code and documentation updates rather than
silently bypassing the validation.

### Installation mode

With `--install`, the script:

1. requires the canonical detailed audit as input;
2. preserves it as the ignored local file
   `embedded_metadata_audit.private.csv`;
3. installs the minimized canonical audit;
4. writes a separate sensitive-hit index;
5. writes a public summary including source/output SHA-256 values.

Canonical public outputs are:

```text
datasets/forensic_evaluation_bundle/metadata/embedded_metadata_audit.csv
datasets/forensic_evaluation_bundle/metadata/embedded_metadata_sensitive_hits.csv
datasets/forensic_evaluation_bundle/metadata/embedded_metadata_public_summary.json
```

### Main options

- `--source`;
- `--staging-dir`;
- `--install`;
- `--force`.

Example:

```bash
python datasets/scripts/bundle/17_build_public_embedded_metadata_audit.py
```

After reviewing staged files:

```bash
python datasets/scripts/bundle/17_build_public_embedded_metadata_audit.py \
  --install
```

---

# Shared utilities

## `utils/paths.py`

Centralizes repository-root discovery and the canonical directory registry used
by dataset, attack, model, evaluation, results, forensic-tool, explainability,
and thesis-document scripts.

### Repository discovery

The module walks upward from its own location until it finds a directory
containing both:

```text
datasets/
datasets/scripts/
```

This avoids hardcoded workstation paths.

### Exposed paths

The constants include the repository root and the canonical locations for:

- raw, prepared, and final datasets;
- clean/OOD splits and split manifests;
- adversarial and anti-forensic outputs;
- evaluation and results;
- model, explainability, forensic-tool, and thesis-document directories.

`DEFAULT_PATHS` exposes the same locations through a named registry.

### Helpers

`repo_relative_path()` resolves a relative value against the repository root and
preserves an explicitly supplied absolute path.

`existing_path_validator()` builds reusable validators that raise a clear
`FileNotFoundError` when a required path does not satisfy the caller's
predicate.

This module writes no artifacts and should normally be imported rather than
executed.

---

# Package markers

The following files are intentionally empty:

```text
datasets/scripts/__init__.py
datasets/scripts/acquisition/__init__.py
datasets/scripts/prepared/__init__.py
datasets/scripts/final/__init__.py
datasets/scripts/splits/__init__.py
datasets/scripts/attacks/__init__.py
datasets/scripts/bundle/__init__.py
datasets/scripts/utils/__init__.py
```

They mark the directories as importable Python packages and support imports such
as:

```python
from datasets.scripts.utils.paths import REPO_ROOT
```

They contain no pipeline logic, accept no arguments, and produce no output.
They should not be treated as executable stages.

---

# Reproducibility checklist

Before running a numbered stage:

1. run it from the repository root;
2. confirm that its upstream manifest is the intended frozen or regenerated
   artifact;
3. inspect any use of `--force`;
4. record external dataset versions, URLs, credentials, and access dates outside
   committed secrets;
5. preserve the generated CSV/JSON summaries;
6. verify reported counts and hash checks;
7. keep blind forensic-tool inputs separate from labels and metadata;
8. do not interpret historical probability fields as calibrated confidence;
9. do not modify frozen manifests manually;
10. create a new documented experimental freeze when dataset composition,
    parameters, checkpoints, or transformation definitions change.

## Recommended execution sequence

A full local rebuild follows the numbered stages, subject to controlled-data
availability:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py --artifact raw ...
python datasets/scripts/prepared/08_build_prepared_dataset.py --force
python datasets/scripts/prepared/09_generate_review_manifest_full.py
python datasets/scripts/final/10_manual_selection_protocol_reviewer.py
python datasets/scripts/splits/11_generate_clean_and_ood_splits.py --force
python datasets/scripts/attacks/13_generate_anti_forensic_attacks.py ...
python datasets/scripts/attacks/14_generate_adversarial_attacks.py ...
python datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py --layout both --force
python datasets/scripts/bundle/17_build_public_embedded_metadata_audit.py
```

Steps 01-07 are alternative source-acquisition/reconstruction components used
when rebuilding raw sources rather than restoring the controlled raw archive.
