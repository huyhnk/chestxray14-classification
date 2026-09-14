# Review of the supplied code files

This repository was refactored from six supplied files. The goal was to keep the reproducible research logic while removing exploratory or redundant code.

| Original file | Decision | Reason |
|---|---|---|
| `TestDannyNet.ipynb` | **Keep logic / refactor** | Best starting point for the training pipeline: patient-level split, DenseNet121, focal loss, AUROC/F1 evaluation, early stopping. Refactored into `src/` + `scripts/train.py`. |
| `Model.ipynb` | **Keep selected logic only** | Contains useful dataset/model experiments (DenseNet/ResNet variants), but has duplicate class definitions, notebook-order dependencies, hard-coded Drive paths, and experimental prediction cells. Model variants were consolidated into `src/models.py`. |
| `PreProcessingLungs.ipynb` | **Keep logic / refactor** | Contains the reusable lung-detection/crop procedure. Refactored into `src/preprocess_lungs.py` and `scripts/preprocess_lungs.py`. The hard-coded API key was removed. |
| `chest-xray14.py` | **Keep selected logic only** | Useful standalone dataset loading and 14-label conversion, but overlaps with the training notebooks and has no validation/test pipeline. The useful parts were consolidated into `src/data.py`. |
| `segmeentLungs.ipynb` | **Exclude from main repository** | Mostly commented experiments, visualization, and one-off data manipulation. It duplicates the lung-cropping logic from `PreProcessingLungs.ipynb`. |
| `TestResizeImg.ipynb` | **Exclude from main repository** | Mainly file counting, archive extraction, missing-file checks, and one-off CSV edits. These are debugging/data-maintenance operations, not part of the reproducible model pipeline. |

## Important corrections made during refactoring

1. **Removed embedded Roboflow credentials.** Public repositories must not contain API keys. `ROBOFLOW_API_KEY` is now read from the environment.
2. **Validation thresholds are no longer fitted on the test set.** The supplied `TestDannyNet.ipynb` optimized thresholds inside every evaluation call; doing that on the test set leaks test-label information. Thresholds are now fitted on validation predictions and frozen for final test evaluation.
3. **Patient-level splitting is retained.** This is preferable to image-level splitting because multiple images may belong to the same patient.
4. **Image discovery now scans all subfolders recursively.** The supplied test notebook only indexed one image folder in the active code path.
5. **Model outputs are logits.** Sigmoid is applied only for prediction/metrics; training uses `BCEWithLogitsLoss` or focal loss based on logits.
6. **Reduced notebook-order dependence and duplicated classes.** Training now runs from command line with explicit inputs.
7. **Learning-rate scheduling uses validation loss.** This is more consistent with `ReduceLROnPlateau` than stepping it with training loss.
8. **AUROC is robust to a class missing positives/negatives in a split.** Such class AUROC values are reported as missing rather than crashing evaluation.

## What is intentionally not included

- NIH ChestX-ray14 image data or metadata files.
- Trained weights/checkpoints.
- User-specific Google Drive paths.
- Roboflow API keys.
- Temporary CSVs such as `Dropdata.csv` and ad-hoc file-counting notebooks.
