# Medical Representation Learning on Gastrointestinal Endoscopy (Kvasir v2)

Structured experimental comparison of supervised transfer learning and contrastive self-supervised pretraining (SimCLR) on gastrointestinal endoscopy imaging, with downstream classification and segmentation analysis.

This repository emphasizes quantitative evaluation, controlled experimental design, and honest analysis of representation learning trade-offs in medical computer vision.

---

## Overview

This project investigates how different representation learning strategies perform on the Kvasir v2 gastrointestinal endoscopy dataset.

We compare:

- Supervised transfer learning (ResNet50, EfficientNet-B0, ViT-B/16)
- Contrastive self-supervised pretraining (SimCLR)
- Feature-injected segmentation prototype (U-Net variant)

All experiments were conducted under controlled preprocessing and training conditions to ensure fair comparison.

---

## Dataset

**Kvasir v2** — A multi-class gastrointestinal endoscopy image dataset.

- Dataset source: https://www.kaggle.com/datasets/plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset
- Total classes: 8 GI findings
- Images resized to 224×224
- ImageNet normalization applied

### Data Split Strategy

A controlled stratified split was used:

- 64% Training
- 16% Validation
- 20% Test

This ensures stable evaluation and prevents data leakage across experiments.

---

## Experimental Design

### Supervised Transfer Learning

Pretrained ImageNet backbones were fine-tuned under identical conditions:

- Input resolution: 224×224
- Optimizer: Adam
- Learning rate: 1e-4
- Loss: CrossEntropyLoss
- Batch size: 32

Backbones evaluated:
- ResNet50
- EfficientNet-B0
- ViT-B/16

Metrics:
- Accuracy
- Macro F1 Score

---

### Self-Supervised Learning (SimCLR)

To evaluate representation quality without labels, SimCLR contrastive pretraining was implemented.

Configuration:
- Encoder: ResNet18
- Projection head: 128-dimensional MLP
- Loss: NT-Xent
- Temperature: 0.5
- Pretraining epochs: 5
- Fine-tuning epochs: 5

After contrastive pretraining, a linear classifier was trained on frozen features and evaluated on the same test set.

---

## Experimental Results

### Supervised vs Contrastive Learning

![Supervised vs SimCLR Comparison](results/supervised_vs_simclr_comparison.png)

### Test Set Performance

| Model            | Accuracy | F1 Score |
|------------------|----------|----------|
| EfficientNet-B0  | 0.8919   | 0.8917   |
| ResNet50         | 0.8888   | 0.8879   |
| ViT-B/16         | 0.7956   | 0.7680   |
| SimCLR (Fine-tuned) | 0.5672 | 0.5615 |

---

## Key Findings

- EfficientNet-B0 achieved the strongest classification performance.
- ResNet50 performed competitively, confirming strong convolutional inductive bias.
- ViT underperformed in this moderate-scale medical dataset regime.
- Short-horizon SimCLR pretraining did not surpass supervised transfer learning.
- Strong ImageNet initialization provided richer inductive bias than limited self-supervised pretraining in this setting.

These results highlight the importance of dataset scale, pretraining duration, and inductive bias when applying representation learning to medical imaging.

---

## Segmentation Prototype (Exploratory)

A feature-injected U-Net variant using ResNet50 encodings was prototyped on Kvasir-SEG.

- Mean IoU: 39.23%

This experiment explores cross-task feature transfer and representation reuse between classification and segmentation.

---

## Repository Structure

```
medical-representation-learning-kvasir/
│
├── notebooks/
│   └── kvasir_supervised_vs_simclr_pipeline.ipynb
│
├── results/
│   └── supervised_vs_simclr_comparison.png
│
├── configs/
│
├── assets/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Mohd-Abdul-Rafay/medical-representation-learning-kvasir.git
cd medical-representation-learning-kvasir
pip install -r requirements.txt
```

---

## Reproducibility Notes

•	All experiments use fixed preprocessing and identical splits.
•	Results reported are from held-out test data.
•	SimCLR pretraining duration was intentionally limited to evaluate short-horizon SSL behavior.

---

## Author

**Abdul Rafay Mohd**  
Artificial Intelligence | Medical AI | Computer Vision 

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).

---

## Citation

If this work is useful in your research, please cite:

```bibtex
@software{rafay2025smallobject,
  author  = {Abdul Rafay Mohd},
  title   = {Small Object Detection in Dense UAV Imagery: Structured Ablation of YOLOv8-L},
  year    = {2025},
  url     = {https://github.com/Mohd-Abdul-Rafay/medical-representation-learning-kvasir}
}
