# Glaucoma Detection Model Card

## Model Overview

**Model Name**: Glaucoma Detection Model  
**Version**: 3.6.2  
**Type**: Binary Classification (Glaucoma / No Glaucoma)  
**Date**: April 2025  
**License**: MIT  

## Model Description

This model is designed to detect glaucoma from retinal images. It uses a two-stage approach:

1. **Feature Extraction**: A pre-trained ResNet50 CNN is used to extract deep features from retinal images.
2. **Classification**: A Multi-Layer Perceptron (MLP) neural network classifies the extracted features to determine the presence of glaucoma.

This model was developed to assist healthcare providers in the early detection of glaucoma, potentially enabling timely intervention and treatment.

## Intended Use

- **Primary Use Case**: Assist ophthalmologists and optometrists in screening for glaucoma in retinal images.
- **Intended Users**: Healthcare professionals with expertise in eye conditions.
- **Out-of-Scope Uses**: This model should not be used as the sole basis for clinical diagnosis. It is intended as a screening tool only and all predictions should be reviewed by qualified medical professionals.

## Model Architecture

### Feature Extraction

- **Backbone**: ResNet50
- **Weights**: Pre-trained on ImageNet
- **Image Input Size**: 224 × 224 pixels
- **Pooling Method**: Global Average Pooling
- **Feature Output**: 2048-dimensional feature vector

### Classification

- **Model Type**: Multi-Layer Perceptron (MLP)
- **Architecture**: 3 hidden layers with 100 neurons each
- **Activation Function**: ReLU
- **Optimizer**: Adam
- **Early Stopping**: Enabled with 10 iterations patience
- **Regularization**: L2 regularization (alpha=0.0001)

## Training Methodology

### Data

- **Source**: Proprietary dataset of retinal images with glaucoma and non-glaucoma cases
- **Data Split**: 80% training, 20% testing
- **Class Balance**: Class imbalance addressed using SMOTE (Synthetic Minority Over-sampling Technique)

### Preprocessing

- **Image Preprocessing**:
  - Resizing to 224 × 224 pixels
  - Normalization using ImageNet mean and standard deviation
  - Data augmentation (rotation, shifting, flipping, zoom, shear)

- **Feature Preprocessing**:
  - Standardization using StandardScaler
  - Feature selection using Random Forest importance (optional)

### Training Process

- **Hyperparameter Optimization**: Grid search with 5-fold cross-validation
- **Feature Selection**: Optional Random Forest-based feature selection
- **Class Balancing**: SMOTE for handling class imbalance
- **Training Parameters**:
  - Maximum iterations: 200
  - Batch size: 16
  - Learning rate: Adaptive (starting at 0.0005)
  - Random seed: 42

## Performance

The model was evaluated on a test set of 484 retinal images. The confusion matrix analysis shows:

<img src="/images/confusion_matrix.png" alt="Confusion Matrix" width="400" />

### Exact Performance Metrics:

- **Total Test Samples**: 484
- **Accuracy**: 90.7% (439/484)
- **Glaucoma Class**:
  - True Positives: 252
  - False Negatives: 15
  - Sensitivity (Recall): 94.4% (252/267)
  - Precision: 89.4% (252/282)
- **No Glaucoma Class**:
  - True Negatives: 187
  - False Positives: 30
  - Specificity: 86.2% (187/217)
  - Precision: 92.6% (187/202)
- **F1-Score**: 91.8% (for Glaucoma class)

The model demonstrates high sensitivity (94.4%) in detecting Glaucoma cases, which is particularly important for a screening tool where missing positive cases (false negatives) would be more concerning than false positives.

### Confusion Matrix Explanation:

- **Top-Left (252)**: Correct Glaucoma predictions
- **Top-Right (15)**: Glaucoma cases incorrectly predicted as No Glaucoma
- **Bottom-Left (30)**: No Glaucoma cases incorrectly predicted as Glaucoma
- **Bottom-Right (187)**: Correct No Glaucoma predictions

## Limitations

- **Demographic Representation**: The model may not perform equally well across all demographic groups if the training data lacks diversity.
- **Image Quality**: Performance may degrade with poor quality images, incorrect framing, or presence of artifacts.
- **Co-occurring Conditions**: The model may not correctly classify images with multiple conditions or unusual presentations of glaucoma.
- **False Positives/Negatives**: Like all screening tools, this model will produce some false positives and false negatives.

## Ethical Considerations

- **Human Oversight**: This model should be used under human supervision with final decisions made by qualified healthcare professionals.
- **Privacy**: No patient data should be stored or processed without appropriate consent and data protection measures.
- **Bias**: Regular monitoring should be conducted to ensure the model does not exhibit bias across demographic groups.
- **Transparency**: Patients should be informed when AI tools are used as part of their screening or diagnosis.

## How to Use

### API Usage

The model is deployed as a REST API. To use it:

1. Send retinal images to the `/upload/` endpoint
2. The API will return:
   - Prediction (Glaucoma/No Glaucoma)
   - Confidence score
   - Visualization showing the model's focus areas (Grad-CAM)

Example:
```python
import requests

url = "http://localhost:8236/upload/"
files = {"file": ("image.jpg", open("path/to/image.jpg", "rb"), "image/jpeg")}

response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
```

### Local Model Usage

The model can also be used locally:

```python
from model_utils import load_resnet_feature_extractor, load_mlp_model, extract_features_from_image, predict_glaucoma

# Load models
feature_extractor = load_resnet_feature_extractor()
mlp_model = load_mlp_model()

# Extract features
features = extract_features_from_image("path/to/image.jpg", feature_extractor)

# Make prediction
prediction, probability = predict_glaucoma(features, mlp_model)
print(f"Prediction: {prediction}, Probability: {probability}")
```

## Maintenance

The model should be periodically retrained as new data becomes available. Performance metrics should be monitored to detect any degradation over time. The model's behavior should be regularly audited, especially after significant updates to the underlying libraries or deployment environment.

## Model Provenance

Developed by the Medical Imaging AI Team. The feature extraction backbone (ResNet50) was developed by Microsoft Research and pre-trained on ImageNet.

## Contact Information

For questions or support, please contact the project maintainers at:
- Email: [glaucoscan.ai@gmail.com](mailto:glaucoscan.ai@gmail.com)
- GitHub Issues: [https://github.com/KrishNachnani/Glaucoma/issues](https://github.com/KrishNachnani/Glaucoma/issues)

---

*This model card follows the recommendations outlined in "Model Cards for Model Reporting" (Mitchell et al., 2019).*