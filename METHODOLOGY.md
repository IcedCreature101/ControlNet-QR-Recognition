# Methodology

## Overview

This project extends the work of Kormaník et al. (2024) on using ControlNet for ML education by reframing the QR code generation problem as a supervised machine learning classification task.

---

## 1. Problem Formulation

### 1.1 Original Paper's Approach
The original paper focuses on:
- Using ControlNet + Stable Diffusion to generate artistic QR codes
- Teaching ML concepts through visual feedback
- Manual parameter tuning (trial-and-error approach)
- Generation time: 2.5 hours average per student

### 1.2 Our Extension
We convert this into a predictive ML problem:
- **Input**: Generation parameters (strength, CCS, GS, prompt characteristics)
- **Output**: Binary classification (scannable vs. not scannable)
- **Goal**: Predict scannability without expensive image generation

### 1.3 Research Questions
1. Can we predict QR code scannability from generation parameters?
2. Which parameters most influence scannability?
3. How do different ML algorithms perform on this task?
4. Can this approach reduce computation time while maintaining educational value?

---

## 2. Data Collection & Generation

### 2.1 Synthetic Data Generation

Since the original paper doesn't provide their full dataset, we generated synthetic data based on their documented findings:

**Key Findings from Paper:**
- Strength parameter (0-1): Optimal range 0.5-0.7 for scannability
- CCS (ControlNet Conditioning Scale): Optimal ~1.5
- GS (Guidance Scale): Optimal range 10-15
- Prompt length: Shorter prompts (1-3 words) = 92% scannable
- Prompt length: Longer prompts (>10 words) = 45% scannable

**Data Generation Process:**
```python
# Core parameters (continuous)
strength: uniform(0.0, 1.0)
ccs_value: uniform(0.5, 2.0)
gs_value: uniform(5.0, 20.0)

# Prompt characteristics (discrete)
prompt_length: randint(1, 30)
negative_prompt_length: randint(5, 20)

# QR properties (categorical/discrete)
error_correction_level: choice(['L', 'M', 'Q', 'H'])
qr_version: randint(1, 41)
image_resolution: choice([256, 512, 768, 1024])
```

**Label Generation Logic:**
```
base_probability = 0.5

# Strength effect (weight: 0.4)
if 0.5 <= strength <= 0.7:
    probability += 0.3
else:
    probability -= 0.2

# CCS effect (weight: 0.25)
if 1.3 <= ccs_value <= 1.7:
    probability += 0.2
else:
    probability -= 0.15

# Prompt length effect (weight: 0.15)
if prompt_length <= 3:
    probability += 0.2
elif prompt_length > 10:
    probability -= 0.25

# Error correction effect (weight: 0.05)
probability += error_correction_bonus
```

### 2.2 Dataset Statistics
- **Total samples**: 5,000
- **Training set**: 3,500 (70%)
- **Validation set**: 750 (15%)
- **Test set**: 750 (15%)
- **Class distribution**: 64% scannable, 36% not scannable
- **Noise added**: 5% label noise for realism

---

## 3. Feature Engineering

### 3.1 Base Features (8)
1. `strength`: Control image influence [0, 1]
2. `ccs_value`: ControlNet conditioning scale [0.5, 2.0]
3. `gs_value`: Guidance scale [5, 20]
4. `prompt_length`: Number of words in prompt
5. `negative_prompt_length`: Number of words in negative prompt
6. `qr_version`: QR code version [1, 40]
7. `image_resolution`: Output resolution (pixels)
8. `num_iterations`: Diffusion steps [20, 100]

### 3.2 Categorical Features (1)
9. `error_correction_level`: L, M, Q, or H (encoded as ordinal)

### 3.3 Derived Features (3)
10. `strength_ccs_interaction`: strength × ccs_value
11. `strength_gs_interaction`: strength × gs_value
12. `prompt_ratio`: prompt_length / (negative_prompt_length + 1)

### 3.4 Feature Scaling
- Applied StandardScaler to all numeric features
- Z-score normalization: (x - μ) / σ

---

## 4. Model Selection & Training

### 4.1 Models Evaluated

#### 4.1.1 Logistic Regression (Baseline)
- **Purpose**: Simple linear baseline
- **Hyperparameters**:
  - Solver: lbfgs
  - Max iterations: 1000
  - Regularization: L2 (default)

#### 4.1.2 Random Forest
- **Purpose**: Capture non-linear relationships
- **Hyperparameters**:
  - n_estimators: 200
  - max_depth: 15
  - min_samples_split: 5
  - min_samples_leaf: 2

#### 4.1.3 XGBoost
- **Purpose**: Gradient boosting for complex patterns
- **Hyperparameters**:
  - n_estimators: 150
  - max_depth: 8
  - learning_rate: 0.1
  - subsample: 0.8
  - colsample_bytree: 0.8

#### 4.1.4 Support Vector Machine (SVM)
- **Purpose**: Maximum margin classification
- **Hyperparameters**:
  - Kernel: RBF
  - C: 1.0
  - Gamma: scale
  - Probability: True (for probability estimates)

#### 4.1.5 Neural Network (MLP)
- **Purpose**: Deep learning approach
- **Architecture**:
  - Hidden layers: (128, 64, 32)
  - Activation: ReLU
  - Solver: Adam
  - Early stopping: Enabled

### 4.2 Training Procedure
1. Split data (70-15-15)
2. Preprocess features (encoding, scaling)
3. Train each model on training set
4. Validate on validation set
5. Final evaluation on test set
6. 5-fold cross-validation for robustness

### 4.3 Hyperparameter Selection
- Based on literature best practices
- Grid search was considered but omitted due to time constraints
- Future work: Bayesian optimization

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

### 5.2 Confusion Matrix Analysis
```
                Predicted
                0      1
Actual  0      TN     FP
        1      FN     TP
```

### 5.3 Feature Importance Analysis
- Random Forest: Gini importance
- XGBoost: Gain-based importance
- Logistic Regression: Coefficient magnitudes

---

## 6. Comparison with Original Paper

### 6.1 Original Methodology
- **Approach**: Generate images and manually test scannability
- **Time**: 2.5 hours average to first successful image
- **Cost**: High GPU usage (CUDA-enabled)
- **Validation**: Three libraries (PyZBar, OpenCV, ZXing)
- **Iterations**: 1000+ generations per parameter set
- **Educational value**: Visual feedback, immediate results

### 6.2 Our ML Methodology
- **Approach**: Predict scannability from parameters
- **Time**: <0.1 seconds per prediction
- **Cost**: CPU-only, minimal resources
- **Validation**: Probabilistic confidence scores
- **Iterations**: Test millions of combinations instantly
- **Educational value**: Teaches ML classification + original concepts

### 6.3 Complementary Nature
Our approach **doesn't replace** the original but **augments** it:
1. Students can quickly validate parameter ranges
2. Reduces trial-and-error time
3. Enables parameter space exploration
4. Still generates images for high-confidence predictions
5. Teaches both generative AI and predictive ML

---

## 7. Limitations & Assumptions

### 7.1 Data Limitations
- Synthetic data based on paper's findings
- May not capture all real-world complexities
- Limited to parameters mentioned in paper
- 5% noise may not reflect actual generation variance

### 7.2 Model Limitations
- Binary classification (scannable/not scannable)
- Doesn't predict scan success rate
- Doesn't account for device-specific scanning
- Assumes paper's findings generalize

### 7.3 Scope Limitations
- Focused only on QR codes
- Single ControlNet model (DionTimmer's qrcode-control)
- Single base model (Stable Diffusion 2.1)
- No consideration of aesthetic quality

---

## 8. Reproducibility

### 8.1 Random Seeds
- Data generation: seed=42
- Train-test split: random_state=42
- All models: random_state=42

### 8.2 Software Versions
- Python: 3.8+
- scikit-learn: 1.3.0
- XGBoost: 1.7.6
- NumPy: 1.24.3
- Pandas: 2.0.3

### 8.3 Hardware
- CPU: Any modern processor
- RAM: 8GB minimum
- GPU: Not required
- Storage: 500MB for data and models

---

## 9. Ethical Considerations

### 9.1 Educational Context
- Designed for learning, not production use
- Students should understand limitations
- Emphasizes critical thinking about ML predictions

### 9.2 Data Privacy
- No personal data collected
- All data is synthetic
- Open-source approach

### 9.3 Academic Integrity
- Clear attribution to original paper
- Methodology fully documented
- Code and data publicly available

---

## 10. Future Work

### 10.1 Model Improvements
- Hyperparameter tuning (grid search, Bayesian optimization)
- Ensemble methods (stacking, blending)
- Deep learning architectures (attention mechanisms)
- Multi-task learning (scannability + aesthetic quality)

### 10.2 Data Enhancements
- Real image generation and labeling
- Crowdsourced scanning success rates
- Multi-device testing
- Cross-platform validation

### 10.3 Feature Expansion
- Image analysis features (edge detection, contrast)
- Prompt semantic analysis (NLP)
- Historical generation success rates
- User feedback integration

### 10.4 Deployment
- Web API for real-time predictions
- Integration with ControlNet UI
- Mobile app for on-device prediction
- Batch processing system

---

## References

1. Kormaník, T., Gabonai, M., & Porubän, J. (2024). Using Machine Learning Concepts with ControlNet for Educational Advancements. ICETA 2024.

2. Zhang, L., Rao, A., & Agrawala, M. (2023). Adding Conditional Control to Text-to-Image Diffusion Models. ICCV 2023.

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.

4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
