# Breast-cancer-histopathology-classification
Deep Learning-based Breast Cancer Histopathology Image Classification using MobileNetV2 with Threshold Optimization and Grad-CAM.
#  Breast Cancer Histopathology Image Classification using Deep Learning

##  Project Overview

This project focuses on building a deep learning model to classify breast cancer histopathology images into:

- **0 → Benign**
- **1 → Malignant**

The objective is to develop a reliable medical image classification system using transfer learning while optimizing evaluation metrics critical for clinical diagnosis.

---

##  Dataset Information

**Source:** Kaggle – Breast Histopathology Images Dataset  
**Total Images:** 555,048  

### Original Class Distribution
- Benign (0): 397,476
- Malignant (1): 157,572

The dataset was highly imbalanced.

### Data Balancing Strategy

To prevent model bias:

- 20,000 benign images were sampled
- 20,000 malignant images were sampled

**Final Dataset Size:** 40,000 images (Balanced)

---

##  Data Preprocessing

### Steps Performed

- Extracted image paths from folder structure
- Assigned binary labels (0 = benign, 1 = malignant)
- Created structured DataFrame
- Stratified 80-20 train-test split

### Dataset Split

- Training Set: 32,000 images
- Testing Set: 8,000 images

### Data Augmentation (Training Only)

Applied using `ImageDataGenerator`:

- Rescaling (1/255 normalization)
- Rotation (±15°)
- Zoom (10%)
- Horizontal flip

Purpose:
- Improve generalization
- Reduce overfitting

---

##  Model Architecture

### Base Model

- **MobileNetV2**
- Pretrained on ImageNet
- `include_top=False`
- Initially frozen for feature extraction

### Custom Classification Head

- GlobalAveragePooling2D
- Dense(64, activation='relu')
- Dropout(0.3)
- Dense(1, activation='sigmoid')

### Compilation

- Loss: Binary Crossentropy
- Optimizer: Adam
- Metric: Accuracy

---

##  Training Strategy

### Phase 1 – Feature Extraction

- Base model frozen
- Trained classification head
- 3 epochs

**Validation Accuracy:** ~81%

---

### Phase 2 – Fine-Tuning

- Unfroze upper layers of MobileNetV2
- Reduced learning rate
- Trained 2 additional epochs

**Validation Accuracy:** ~81–82%

The model converged stably.

---

##  Initial Evaluation (Threshold = 0.5)

### Confusion Matrix

- TN = 3587
- FP = 413
- FN = 1109
- TP = 2891

### Performance

- Accuracy ≈ 81%
- Malignant Recall ≈ 0.72

⚠ Observation:
High false negatives (1109 malignant cases misclassified).

In medical diagnosis, minimizing false negatives is critical.

---

##  ROC Curve & AUC

ROC analysis was performed.

**AUC Score ≈ 0.90**

Interpretation:
The model shows strong class separability.
The issue lies in decision threshold selection.

---

##  Threshold Optimization

Instead of using default threshold = 0.5,
optimal threshold was computed using Youden’s J statistic.

**Optimal Threshold ≈ 0.326**

---

##  Final Evaluation (Optimized Threshold)

### Updated Confusion Matrix

- TN = 3291
- FP = 709
- FN = 634
- TP = 3366

### Updated Performance Metrics

| Metric | Value |
|--------|--------|
| Accuracy | 83% |
| Sensitivity (Malignant Recall) | 0.8415 |
| Specificity | 0.8227 |
| F1 Score | 0.83 |
| AUC | 0.90 |

---

##  Interpretation of Results

After lowering the threshold:

- False negatives reduced significantly (1109 → 634)
- Malignant recall improved (0.72 → 0.84)
- Slight increase in false positives (acceptable trade-off)

In medical AI systems, maximizing sensitivity is more important than maximizing raw accuracy.

The optimized model is safer for diagnostic screening.

---

##  Hardware Used

- CPU-only training
- Transfer learning used for computational efficiency

---

##  Key Contributions

- Balanced large-scale medical dataset
- Implemented transfer learning using MobileNetV2
- Two-phase training strategy
- ROC-based threshold optimization
- Medical-metric-focused evaluation
- Significant reduction in false negatives

---

##  Conclusion

A robust deep learning pipeline was successfully developed for breast cancer histopathology image classification.

The final optimized model achieves:

- 83% accuracy
- 84% sensitivity
- 82% specificity
- 0.90 AUC

Threshold optimization significantly improved diagnostic reliability.

This project demonstrates practical application of deep learning in medical image analysis while maintaining computational efficiency using CPU-only training.

---
