# Chapter 6 Finalization Report

## Applied changes

- Replaced the embedded-metadata sensitivity subsection.
- Replaced the comparative operational robustness analysis.
- Replaced the explainability and representative-failure-case section.
- Replaced the operational interpretation and limitations section.
- Updated both XAI scripts for Chapter 6 and Max-P terminology.

## Validation outcomes

- Repository audits: success
- Complete LaTeX compilation: success
- LaTeX log validation: success
- Validation source commit: 14246371e4551726e02a7e23c5fb76b78591148d

## Audit output (last 120 lines)

```text
OK  attacks/manifests/adversarial_color_shift_summary.json
OK  attacks/manifests/adversarial_fgsm_efficientnet_b0_summary.json
OK  attacks/manifests/adversarial_one_pixel_efficientnet_b0_summary.json
OK  attacks/manifests/adversarial_sigma_zero_efficientnet_b0_summary.json
OK  attacks/manifests/adversarial_superdeepfool_efficientnet_b0_summary.json
OK  attacks/manifests/anti_forensic_generation_summary.json
OK  datasets/final/reports/manual_selection_summary.json
OK  datasets/forensic_evaluation_bundle/metadata/bundle_summary.json
OK  datasets/forensic_evaluation_bundle/metadata/embedded_metadata_audit_summary.json
OK  datasets/prepared/final_pool/reports/prepared_build_summary.json
OK  datasets/splits/manifests/split_generation_summary.json
OK  evaluation/forensic_tools/normalization_summary.json
OK  evaluation/forensic_tools/normalized_predictions_public_summary.json
OK  explainability/manifests/chapter5/candidate_selection_summary.json
OK  explainability/manifests/chapter5/run_provenance.json
OK  explainability/manifests/chapter5/run_summary.json
OK  forensic_tools/public_extracts_summary.json
OK  forensic_tools/public_extracts_validation.json
OK  forensic_tools/run_registry.json
OK  models/model_registry.json
OK  results/figures/chapter_5/chapter5_figures_summary.json
OK  results/figures/chapter_5/embedded_metadata_sensitivity_summary.json
OK  results/metrics/proxy_model_evaluation_summary.json
JSON files checked: 23
Python syntax check passed.
Repository layout and text guards passed.
Results-chapter XAI public-artifact validation passed.
 - thesis cases: 5
 - thesis assets: 20
 - local path leakage: none
Results artifact validation passed.
 - canonical commercial decisions: 69000
 - commercial metric rows: 186
 - proxy prediction rows: 40500
 - OOD: 500 unique images x 5 folds = 2500 predictions per architecture
 - Chapter 5 manifest: 41 files, 24 unique asset IDs
Chapter 5 reporting-asset audit completed.
 - manifest rows: 41
 - unique asset IDs: 24
 - referenced in thesis: 11
 - unreferenced in thesis: 13
 - byte-identical thesis copy relations: 17
 - missing reporting outputs: 0
 - mismatched existing thesis copies: 0
 - unreferenced IDs:
   - fig_forensic_tools_attack_family_accuracy
   - fig_forensic_tools_attack_family_fnr
   - fig_forensic_tools_attack_family_fpr
   - fig_forensic_tools_attack_family_recall_weapon
   - fig_forensic_tools_clean_metrics_comparison
   - fig_max_accuracy_drop_by_model
   - tab_forensic_tools_attack_family_metrics
   - tab_forensic_tools_attack_name_metrics
   - tab_forensic_tools_clean_metrics
   - tab_forensic_tools_global_metrics
   - tab_forensic_tools_ood_metrics
   - tab_forensic_tools_sensitivity_summary
   - tab_forensic_tools_worst_cases

LaTeX image audit completed
  main.tex:                 docs/LatexThesis/main.tex
  tex files scanned:        13
  includegraphics found:    21
  resolved references:      21
  missing references:       0
  unique used images:       21
  inventory images:         93
  unused images:            72
  duplicate image groups:   0
  reports:                  results/latex_image_audit
```

## LaTeX diagnostics

```text
3300:Underfull \hbox (badness 4492) in paragraph at lines 3277--3277
3305:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3310:Underfull \hbox (badness 4416) in paragraph at lines 3670--3670
3315:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3320:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3325:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3330:Underfull \hbox (badness 3364) in paragraph at lines 3670--3670
3335:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3340:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3345:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3350:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3355:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3360:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3365:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3370:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3375:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3380:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3385:Underfull \hbox (badness 6708) in paragraph at lines 3670--3670
3390:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3395:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3400:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3405:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3410:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3415:Underfull \hbox (badness 6708) in paragraph at lines 3670--3670
3420:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3425:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3430:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3435:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3440:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3445:Underfull \hbox (badness 3803) in paragraph at lines 3670--3670
3450:Underfull \hbox (badness 10000) in paragraph at lines 3670--3670
3455:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3460:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3465:Underfull \hbox (badness 1097) in paragraph at lines 4031--4031
3470:Underfull \hbox (badness 4291) in paragraph at lines 4031--4031
3475:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3480:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3485:Underfull \hbox (badness 4291) in paragraph at lines 4031--4031
3490:Underfull \hbox (badness 1033) in paragraph at lines 4031--4031
3495:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3500:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3505:Underfull \hbox (badness 4291) in paragraph at lines 4031--4031
3510:Underfull \hbox (badness 2735) in paragraph at lines 4031--4031
3515:Underfull \hbox (badness 4132) in paragraph at lines 4031--4031
3520:Underfull \hbox (badness 1117) in paragraph at lines 4031--4031
3525:Underfull \hbox (badness 2932) in paragraph at lines 4031--4031
3530:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3535:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3540:Underfull \hbox (badness 1097) in paragraph at lines 4031--4031
3545:Underfull \hbox (badness 3965) in paragraph at lines 4031--4031
3550:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3555:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3560:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3565:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3570:Underfull \hbox (badness 3965) in paragraph at lines 4031--4031
3575:Underfull \hbox (badness 1275) in paragraph at lines 4031--4031
3580:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3585:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3590:Underfull \hbox (badness 7099) in paragraph at lines 4031--4031
3595:Underfull \hbox (badness 2189) in paragraph at lines 4031--4031
3600:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3605:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3610:Underfull \hbox (badness 10000) in paragraph at lines 4031--4031
3615:Underfull \hbox (badness 3965) in paragraph at lines 4031--4031
3629:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3634:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3639:Underfull \hbox (badness 6284) in paragraph at lines 4179--4179
3644:Underfull \hbox (badness 1946) in paragraph at lines 4179--4179
3649:Underfull \hbox (badness 1735) in paragraph at lines 4179--4179
3654:Underfull \hbox (badness 1337) in paragraph at lines 4179--4179
3659:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3664:Underfull \hbox (badness 4291) in paragraph at lines 4179--4179
3669:Underfull \hbox (badness 2393) in paragraph at lines 4179--4179
3674:Underfull \hbox (badness 1776) in paragraph at lines 4179--4179
3679:Underfull \hbox (badness 2310) in paragraph at lines 4179--4179
3684:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3689:Underfull \hbox (badness 4291) in paragraph at lines 4179--4179
3694:Underfull \hbox (badness 6412) in paragraph at lines 4179--4179
3699:Underfull \hbox (badness 3229) in paragraph at lines 4179--4179
3704:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3709:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3714:Underfull \hbox (badness 3965) in paragraph at lines 4179--4179
3719:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3724:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3729:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3734:Underfull \hbox (badness 3965) in paragraph at lines 4179--4179
3739:Underfull \hbox (badness 1715) in paragraph at lines 4179--4179
3744:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3749:Underfull \hbox (badness 1845) in paragraph at lines 4179--4179
3754:Underfull \hbox (badness 2486) in paragraph at lines 4179--4179
3759:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3764:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3769:Underfull \hbox (badness 3965) in paragraph at lines 4179--4179
3774:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3779:Underfull \hbox (badness 10000) in paragraph at lines 4179--4179
3784:Underfull \hbox (badness 1931) in paragraph at lines 4179--4179
3789:Underfull \hbox (badness 2359) in paragraph at lines 4179--4179
3803:Underfull \hbox (badness 1028) in paragraph at lines 4274--4279
3817:Underfull \hbox (badness 2512) in paragraph at lines 4514--4514
3822:Underfull \hbox (badness 2042) in paragraph at lines 4514--4514
3827:Underfull \hbox (badness 2134) in paragraph at lines 4514--4514
3832:Underfull \hbox (badness 2253) in paragraph at lines 4514--4514
3837:Underfull \hbox (badness 5119) in paragraph at lines 4514--4514
4010:Underfull \hbox (badness 1097) in paragraph at lines 4893--4900
4016:Underfull \hbox (badness 7326) in paragraph at lines 4942--4947
4026:Underfull \hbox (badness 2608) in paragraph at lines 277--277
4032:Underfull \hbox (badness 3058) in paragraph at lines 277--277
4037:Underfull \hbox (badness 2744) in paragraph at lines 277--277
4042:Underfull \hbox (badness 1014) in paragraph at lines 277--277
4047:Underfull \hbox (badness 1221) in paragraph at lines 277--277
4052:Underfull \hbox (badness 1248) in paragraph at lines 277--277
4057:Underfull \hbox (badness 1264) in paragraph at lines 277--277
4062:Underfull \hbox (badness 4391) in paragraph at lines 277--277
4067:Underfull \hbox (badness 1194) in paragraph at lines 502--508
4081:Underfull \hbox (badness 1565) in paragraph at lines 457--457
4087:Underfull \hbox (badness 2951) in paragraph at lines 457--457
4096:Underfull \hbox (badness 10000) in paragraph at lines 140--140
4102:Underfull \hbox (badness 1194) in paragraph at lines 140--140
4108:Underfull \hbox (badness 1221) in paragraph at lines 140--140
4114:Underfull \hbox (badness 1521) in paragraph at lines 140--140
```
