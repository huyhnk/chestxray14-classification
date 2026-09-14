# ChestX-ray14 Multi-label Classification

A cleaned PyTorch pipeline for multi-label chest X-ray classification on NIH ChestX-ray14. The code was refactored from exploratory notebooks into a reproducible repository with patient-level data splitting, DenseNet/ResNet baselines, focal/BCE loss, validation-based threshold selection, and optional lung-region cropping.

## Repository structure

```text
chestxray14-classification/
├── README.md
├── requirements.txt
├── requirements-preprocess.txt
├── .env.example
├── .gitignore
├── docs/
│   └── CODE_REVIEW.md
├── scripts/
│   ├── train.py
│   └── preprocess_lungs.py
├── src/
│   ├── constants.py
│   ├── data.py
│   ├── engine.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   └── preprocess_lungs.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── outputs/
```

## Task

The classifier predicts 14 ChestX-ray14 findings:

`Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Effusion`, `Emphysema`, `Fibrosis`, `Hernia`, `Infiltration`, `Mass`, `Nodule`, `Pleural_Thickening`, `Pneumonia`, and `Pneumothorax`.

`No Finding` is encoded as an all-zero target vector.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

The repository does not redistribute NIH ChestX-ray14 data. Place the metadata CSV and images locally, for example:

```text
data/raw/
├── Data_Entry_2017.csv
└── images/
    ├── images_001/
    │   └── images/
    ├── images_002/
    │   └── images/
    └── ...
```

The image loader scans the image root recursively, so equivalent nested layouts also work.

The metadata must contain at least:

- `Image Index`
- `Finding Labels`
- `Patient ID`

## Train

DenseNet121 with focal loss:

```bash
python scripts/train.py \
  --csv data/raw/Data_Entry_2017.csv \
  --images data/raw/images \
  --model densenet121 \
  --loss focal \
  --epochs 10 \
  --batch-size 16 \
  --output outputs/densenet121
```

Other included backbones:

```text
resnet18
resnet101
resnet152
```

The final test thresholds are **not** tuned on the test set. Per-class thresholds are selected from the validation split and saved with the best checkpoint.

Outputs include:

```text
outputs/<run>/
├── best_model.pt
├── history.json
└── test_metrics.json
```

## Optional lung-region preprocessing

The supplied preprocessing notebook used a Roboflow-hosted detector to crop the union of detected lung boxes and resize it to 224×224. This step is optional.

Install the preprocessing dependency:

```bash
pip install -r requirements-preprocess.txt
```

Set the API key outside source code:

```bash
# Linux/macOS
export ROBOFLOW_API_KEY="your-key"

# PowerShell
$env:ROBOFLOW_API_KEY="your-key"
```

Then run:

```bash
python scripts/preprocess_lungs.py \
  --input data/raw/images \
  --output data/processed/lung_crops
```

Train on the cropped images by passing `--images data/processed/lung_crops`.

## Reproducibility and evaluation notes

- Data are split by `Patient ID`, not by image row.
- Default split: 80% train, 10% validation, 10% test at patient level.
- The training loss can be focal loss or `BCEWithLogitsLoss`.
- Model selection uses validation macro AUROC.
- Optimal per-class classification thresholds are estimated on validation data only.
- Final metrics are macro AUROC and macro F1, with per-class AUROC/F1 saved in JSON.

See `docs/CODE_REVIEW.md` for the mapping from the supplied notebooks to this cleaned repository and the corrections made during refactoring.

## Security note

Do not commit API keys, private dataset credentials, patient data, or trained checkpoints to a public repository. The supplied preprocessing notebook contained a hard-coded Roboflow API key; it has intentionally been removed from this repository.
