# Deep Learning Plant Disease Classification

## Overview
This project applies deep learning techniques to classify plant diseases using the **PlantVillage** dataset. The models implemented include:
- **DenseNet**
- **Xception**
- **ResNet**

These models leverage **transfer learning** and **training from scratch** approaches to compare their effectiveness in identifying plant diseases.

## Dataset
We use the **PlantVillage Dataset**, which contains **54,303** images of healthy and diseased leaves across **39 classes**. The dataset undergoes preprocessing steps such as:
- **Resizing** all images to 244x244 pixels
- **Normalization** using ImageNet mean and standard deviation values
- **Stratified splitting** into training (70%), validation (15%), and test (15%) sets

## Model Architectures
### 1. DenseNet (Densely Connected Convolutional Networks)
- Implements short connections between layers to improve gradient flow
- Uses **DenseNet-201** for transfer learning with a frozen base model
- Final classifier: Global Average Pooling + Fully Connected Layer

### 2. Xception (Extreme Inception)
- Uses **Depthwise Separable Convolutions** to reduce computation cost
- Incorporates **residual connections** for better gradient flow
- Three main components: **Entry Flow, Middle Flow, Exit Flow**

### 3. ResNet (Residual Neural Networks)
- Addresses vanishing gradients using **residual learning**
- Implements **bottleneck design** for deeper networks
- Evaluated by training from scratch without pre-trained weights

## Implementation Details
- **Programming Language:** Python
- **Frameworks:** TensorFlow / Keras
- **Optimizer:** Adam (initial learning rate: 0.00001 for DenseNet & Xception, 0.001 for ResNet)
- **Loss Function:** Categorical Cross-Entropy
- **Training:**
  - DenseNet: 20 epochs
  - Xception: 5 epochs
  - ResNet: 20 epochs

## Results & Comparisons
| Model  | Training Approach | Trainable Params | Test Accuracy | AUC Score |
|--------|------------------|-----------------|---------------|-----------|
| DenseNet | Transfer Learning | 74,919 | 95.0% | 0.99 |
| DenseNet | Fine-tuning | 18M+ | 98.0% | 0.99 |
| Xception | Transfer Learning | 534,567 | 94.6% | 0.996 |
| ResNet | From Scratch | 11M+ | 91.0% | 0.949 |

**Key Observations:**
- **DenseNet (Fine-Tuned)** achieved the highest accuracy (98.0%) and AUC score (0.99)
- **Xception** performed well with fewer parameters and less training time (AUC: 0.996)
- **ResNet (Trained from Scratch)** had the lowest accuracy due to its reliance on large datasets

## How to Use
### 1. Clone Repository
```bash
 git clone https://github.com/your-repo-name.git
 cd your-repo-name
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train & Evaluate Models
Run the training script for each model:
```bash
python train_densenet.py
python train_xception.py
python train_resnet.py
```

### 4. Predict on New Images
```bash
python predict.py --image path/to/image.jpg
```

## References
- **PlantVillage Dataset:** [Original Paper](https://arxiv.org/abs/1511.08060)
- **DenseNet:** [Huang et al., 2017](https://arxiv.org/abs/1608.06993)
- **Xception:** [Chollet, 2017](https://arxiv.org/abs/1610.02357)
- **ResNet:** [He et al., 2015](https://arxiv.org/abs/1512.03385)

---
