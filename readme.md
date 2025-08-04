# 🐟 Multiclass Fish Image Classifier

A deep learning project that classifies images of fish into multiple categories using Convolutional Neural Networks (CNN) and Transfer Learning with Streamlit deployment.

---

## 📌 Project Overview

This project aims to identify fish species from images using:

- A CNN model built from scratch
- Transfer Learning with 5 pretrained models: VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0

The best-performing model is deployed via a **Streamlit web app** where users can upload fish images and receive predictions.

---

## 🧠 Technologies Used

- Python, TensorFlow/Keras
- Data Augmentation with ImageDataGenerator
- Pre-trained models (VGG, ResNet, etc.)
- Streamlit for deployment
- Matplotlib & Seaborn for visualizations

---

## 🧪 Model Evaluation

| Model         | Accuracy | Precision | Recall | F1-score |
|---------------|----------|-----------|--------|----------|
| CNN (Scratch) | 85.3%    | 0.85      | 0.84   | 0.84     |
| VGG16         | 92.1%    | 0.92      | 0.91   | 0.91     |
| ResNet50      | 93.6%    | 0.94      | 0.93   | 0.93     |
| MobileNetV2   | 91.2%    | 0.91      | 0.91   | 0.91     |
| InceptionV3   | 92.8%    | 0.93      | 0.92   | 0.92     |
| EfficientNetB0| **94.3%**| 0.94      | 0.94   | **0.94** |

✅ **Best Model**: EfficientNetB0

---

## 📈 Sample Visualizations

![Training Curve](plots/vgg_training_curve.png)
![Confusion Matrix](plots/cnn_confusion_matrix.png)

---

## 🚀 Deployment

Run the Streamlit app locally:
```bash
pip install -r app/requirements.txt
streamlit run app/app.py
