# 🔬 Plant Pathology Research & ML Engineering

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=white)

This repository contains the **R&D (Research and Development)** workflow for training, validating, and optimizing Deep Learning models for plant disease classification. It documents the transition from raw data to a production-ready **EfficientNetB0** model.

## 🎯 Engineering Goals

* **Maximize Accuracy** on a small, noisy dataset (~1600 images).
* **Solve Class Imbalance** and Label Errors using Data-Centric AI techniques.
* **Optimize for Edge:** Create a model lightweight enough for Drone deployment.

## 🧠 Methodology & Experiments

We followed a rigorous scientific approach to model development:

### 1. Data Auditing (Cleanlab)
* Used **Confident Learning** algorithms (`cleanlab`) to detect label errors in the dataset.
* **Result:** Identified and corrected over 1000 mislabeled images (e.g., Healthy leaves labeled as Fusarium), significantly reducing noise.

### 2. Architecture Search
We evaluated multiple backbones for the specific task of texture recognition in leaves:
* **VGG16:** Discarded (Too heavy, high overfitting).
* **MobileNetV2:** Good baseline, but struggled with fine-grained texture differentiation (~60% Val Acc).
* **EfficientNetB0 (Winner):** Selected for its superior feature extraction capabilities and internal scaling. Achieved consistent convergence.

### 3. Training Strategy (Warmup + Fine-Tuning)
To prevent **Catastrophic Forgetting** during Transfer Learning:
1.  **Phase 1 (Warmup):** Frozen backbone, training only the classification head with High LR (`1e-3`).
2.  **Phase 2 (Fine-Tuning):** Unfreezing top 50 layers, training with Low LR (`1e-5`) and customized optimization parameters (based on **EDA algorithm research**: $\alpha=7.1e^{-5}$, $\beta_1=0.75$).

## 📊 Project Structure

```text
PlantPathology_Research/
├── data/               # Raw training and validation datasets
├── models/             # Saved .keras and .tflite models
├── src/
│   ├── audit_labels.py # Cleanlab implementation for error detection
│   ├── train_thor.py   # Final EfficientNet training pipeline
│   ├── convert_thor.py # Keras to TFLite conversion script
│   └── visualize_data.py
├── logs/               # TensorBoard logs
└── requirements.txt
```

## 🚀 How to Train

1. Setup Environment:
```bash
pip install -r requirements.txt
```

2. Run Training Pipeline:
```bash
python src/train_thor.py
```

3. Monitor with TensorBoard:
```bash
tensorboard --logdir logs/fit
```

## 📈 Results

* **Final Model:** EfficientNetB0 ("Thor")
* **Precision (Fusarium):** 85%
* **Precision (Healthy):** 91%
* **Deployment Format:** INT8/Float32 Quantized TFLite

## 
*Authored by Carlos Luis Noriega - Lead AI Engineer*