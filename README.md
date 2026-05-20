# Halal Food Classifier

A Machine Learning and Deep Learning approach to classify food ingredients as **Halal** or **Haram**.

## Overview
This project focuses on classifying food ingredients based on their composition using two main approaches:
1.  **Classical Machine Learning**: TF-IDF Vectorizer + Logistic Regression.
2.  **Deep Learning**: Fine-tuning the **DistilBERT** (Transformer) model for Sequence Classification.

## Dataset
The project uses `cleaned_dataset.csv` which contains ingredient lists and their corresponding labels (Halal/Haram).

## Technologies Used
- Python
- Scikit-learn (Logistic Regression, TF-IDF)
- Hugging Face Transformers (DistilBERT)
- PyTorch
- Pandas, NumPy, Matplotlib, Seaborn

## Getting Started
To run the notebook, you can use Google Colab or a local Jupyter environment with GPU support (recommended for Deep Learning).

### Prerequisites
```bash
pip install transformers datasets accelerate scikit-learn pandas numpy matplotlib seaborn
```

## Results
The model compares the performance of Logistic Regression and DistilBERT in terms of Accuracy, F1-Score, and Training Time.
