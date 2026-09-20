# Concrete Crack Segmentation & Severity Estimation

A U-Net segmentation model that finds cracks in concrete at the pixel level, then measures crack width and coverage to classify each one as Minor, Moderate, or Severe.

Built as a solo deep learning project. Full methodology and results are in [REPORT.md](REPORT.md).

## Key result

Two encoders were trained and compared: one initialized from scratch, one pretrained on ImageNet (MobileNetV2). The from-scratch model matched or slightly outperformed the pretrained one on every metric.

| Metric | From Scratch | Pretrained (MobileNetV2) |
|---|---|---|
| Dice Coefficient | 0.759 | 0.742 |
| IoU | 0.637 | 0.632 |
| Binary Accuracy | 0.9909 | 0.9904 |

The pretrained model had a genuine early lead (val IoU ~0.48 after epoch 1 vs. ~0.07 for the from-scratch model), but the from-scratch model overtook it within two epochs and held a small, consistent lead through epoch 20. REPORT.md walks through the likely cause: fine-tuning the pretrained encoder jointly with an untrained decoder from step one, with no staged freeze/unfreeze schedule, which likely let early noisy gradients disturb the pretrained features. Not a case of transfer learning failing, a fixable methodology choice.

## Severity calibration

Tested on 200 validation images with detected cracks. Severity cutoffs: Minor/Moderate at 5.66px max width, Moderate/Severe at 7.21px. Resulting distribution: 78 Severe, 70 Moderate, 52 Minor.

## What's in this repo

- `concrete_crack_segmentation_severity.ipynb` — full pipeline: data loading, both models, training, evaluation, severity classification, end-to-end demo
- `REPORT.md` — full write-up: methodology, results, generalization test, limitations
- `crack_segmentation_project_report.docx` — same report, original formatted version
- `media/` — figures referenced in REPORT.md

No trained weights are included in this repo. The notebook downloads the Crack Segmentation dataset itself and trains both models from a cold start.

## Reproducing this

Open the notebook in Google Colab with a GPU runtime and run all cells. No manual dataset download or pretrained checkpoint needed. Training both models takes roughly 80 minutes total on a free T4 GPU. A rerun won't match these exact numbers, GPU training isn't fully deterministic, but the same pattern holds: the pretrained model's early advantage, and the from-scratch model's overtake.

## Built with

TensorFlow, Keras, Python

---

Mehtab Ahmed, FCCA, ACA, CISA — [LinkedIn](https://linkedin.com/in/mehtabahmedfcca)
