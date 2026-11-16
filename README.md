# Explainable Deep Learning for Superficial Peritoneal Endometriosis (SPE) Detection

Code for an explainable deep learning pipeline that classifies endometriosis in laparoscopic images using the [GLENDA v1.5](https://ftp.itec.aau.at/datasets/GLENDA/) dataset. The work follows and extends the methodology of Zhu & Elbattah (2025) by adding a peritoneal-specialised classifier, a GradCAM++ vs. RISE explainability comparison, and a small external validation set.

## Overview

- **General classifier:** three architectures (EdgeNeXt_Small, ResNet50, EfficientFormerV2-S0) compared under patient-wise five-fold cross-validation.
- **SPE classifier:** the best general model (EdgeNeXt_Small) fine-tuned for peritoneal endometriosis vs. normal tissue by three-stage progressive unfreezing.
- **Explainability:** GradCAM++ and RISE heatmaps scored against expert lesion masks (IoU, Dice, pixel-level Recall) on all 373 annotated test samples.
- **Demo:** Gradio interface showing predictions, confidence scores and GradCAM++ overlays for both models.

## Repository Structure

```
.
├── Code/
│   ├── deeplearningendo (1).ipynb   # Data pipeline, training, SPE fine-tuning, XAI evaluation
│   ├── externalvalidation.ipynb     # Evaluation on external laparoscopic images
│   └── interfaceformodels.ipynb     # Gradio demo
└── Dataset/
    └── Glenda_v1.5_classes/         # GLENDA v1.5 frames and annotation masks
```

## Requirements

The notebooks were developed and run in Google Colab (GPU/TPU runtime) with Python 3.

```
torch  torchvision  timm  torchmetrics  scikit-learn
pandas  numpy  opencv-python  matplotlib
grad-cam  gradio
```

## Usage

1. Open the notebooks in Google Colab, or locally with Jupyter.
2. Set the dataset and output paths at the top of `deeplearningendo (1).ipynb`. The notebooks were written against Google Drive, so these paths need editing for your setup.
3. Run the notebook in order:
   1. **Metadata construction:** builds the image/mask index and extracts patient IDs (video ID for pathological frames, `v_XXXX` folder token for normal frames).
   2. **Splitting:** patient-wise five-fold split with a programmatic leakage check, saved as `splits.pkl`.
   3. **Training:** EdgeNeXt_Small, ResNet50 and EfficientFormerV2-S0 per fold. Best checkpoint per fold is saved as `fold{n}_best.pt`.
   4. **SPE fine-tuning:** progressive unfreezing from the best general checkpoint. Best checkpoint per fold is saved as `fold{n}_peritoneal_best.pt`.
   5. **Explainability:** GradCAM++ and RISE evaluation against annotation masks.
4. Run `externalvalidation.ipynb` for external images and `interfaceformodels.ipynb` to launch the Gradio demo.

## Methods Summary

| Component | Setting |
|---|---|
| Split | Patient-wise 5-fold; 1:1 normal:pathological test set, 2:1 training pool, 80/20 train/validation |
| Input | 320×320 (224×224 for EfficientFormerV2-S0); flip, ±10° rotation, brightness/contrast jitter 0.1 |
| Training | AdamW (lr 1e-4, wd 1e-4), cosine annealing, 30 epochs, batch 32, early stopping (patience 7, val AUC) |
| SPE fine-tuning | Stage 1 head only (lr 1e-3, 10 ep); Stage 2 + final encoder stage (lr 1e-4, 10 ep); Stage 3 all layers (lr 5e-5, 15 ep) |
| GradCAM++ | Target layer `model.stages[-1]`, fixed threshold 0.3 |
| RISE | 2,000 masks (8×8 Bernoulli, p=0.5, bilinearly upsampled); threshold 0.5 (general model), Otsu (SPE model) |

## Results

### Classification (mean over 5 folds)

| Model | AUC | Accuracy | F1 |
|---|---|---|---|
| EdgeNeXt_Small | 0.9952 ± 0.0039 | 0.9496 ± 0.0280 | 0.9462 ± 0.0324 |
| ResNet50 | 0.9921 ± 0.0072 | 0.9365 ± 0.0293 | 0.9319 ± 0.0340 |
| EfficientFormerV2-S0 | 0.9876 ± 0.0124 | n/a | 0.9441 ± 0.0260 |
| **EdgeNeXt_Small, SPE fine-tuned** | **0.9999 ± 0.0002** | n/a | n/a |

### Explainability alignment (mean)

| Model | Method | IoU | Dice | Recall |
|---|---|---|---|---|
| General | GradCAM++ | 0.1469 | 0.2333 | 0.6369 |
| General | RISE | 0.1450 | 0.2279 | 0.6759 |
| SPE | GradCAM++ | 0.0906 | 0.1534 | 0.6022 |
| SPE | RISE | 0.1205 | 0.1952 | 0.7430 |

High classification AUC did not translate into better spatial alignment: the SPE model scored lower than the general model on IoU and Dice.

## Data

- **GLENDA v1.5** is a public, de-identified dataset. Please cite the original authors and follow their terms of use.
- **External validation images:** [DESCRIBE SOURCE AND COUNT, e.g. images shared by a practising surgeon on gynsurgery.org with consent for research use, plus any synthetic images].

## Model Weights

(huggingface links to be added)

## Citation

```bibtex
@misc{[KEY],
  title  = {Developing an Explainable Deep Learning Model for the Detection of Superficial Peritoneal Endometriosis},
  author = {[Anusha Asim]},
  year   = {2026},
  note   = {[JOURNAL tbd / DOI once available]}
}
```

## References

- Zhu, Y. & Elbattah, M. (2025). Explainable Deep Learning for Endometriosis Classification in Laparoscopic Images. *BioMedInformatics*, 5(4), 63. https://doi.org/10.3390/biomedinformatics5040063
- Leibetseder, A. et al. (2020). GLENDA: Gynecologic Laparoscopy Endometriosis Dataset. *MultiMedia Modeling (MMM)*.

## License

[MIT for code. GLENDA remains under its own licence.]
