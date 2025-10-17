# 🌸 Iris Flower Classification using SVM

## Project Overview

This project classifies Iris flowers into three species: **Setosa**, **Versicolor**, and **Virginica** based on their physical measurements: Sepal Length, Sepal Width, Petal Length, and Petal Width. The classification is performed using **Support Vector Machines (SVM)** with Linear, Polynomial, and RBF kernels.

An interactive **Streamlit app** is provided to input flower measurements and predict the species with fun visualizations and animations.

## 🔗 App Link

You can access the interactive Iris Flower Classification app here:  

[🌸 Iris Flower Classifier App] https://irisflowerclassification-svm.streamlit.app/ 

<img src="https://github.com/user-attachments/assets/1d659f18-2c58-4158-95e4-10a5bcea93b3" width="500" alt="Iris Flower Classifier Screenshot">

<img src="https://github.com/user-attachments/assets/8d7e982b-9a6b-400b-959f-4edeee87c990" width="300" alt="Iris Flower Classifier Screenshot">

## 📂 Dataset

The dataset used is the classic Iris dataset from Kaggle: https://www.kaggle.com/datasets/uciml/iris

### Features:

SepalLengthCm

SepalWidthCm

PetalLengthCm

PetalWidthCm

### Target:

Species (Setosa, Versicolor, Virginica)

## ⚙ Installation

### Clone the repository:

git clone https://github.com/your-username/IrisFlowerClassification.git
cd IrisFlowerClassification

### Create and activate a virtual environment (optional but recommended):

python -m venv myvenv
source myvenv/bin/activate      # Linux/Mac
myvenv\Scripts\activate         # Windows

### Install required libraries:

pip install -r requirements.txt

### Run the app:

streamlit run app.py

## 📝 Steps Performed

### Data Preprocessing

Dropped irrelevant columns like Id.

Checked for missing values.

Visualized feature distributions, correlations, and outliers.

Encoded target labels using LabelEncoder.

Split data into training and testing sets.

Standardized features using StandardScaler.

### Model Training

Trained SVM classifiers with:

Linear kernel

Polynomial kernel (degree=3)

RBF kernel

Evaluated using Accuracy, Macro F1-score, Confusion Matrix, and Classification Report.

### Model Saving

Saved the trained Linear SVM model, scaler, and label encoder using pickle for deployment.

### Visualization

Plotted feature distributions, correlations, boxplots for outliers.

Scatter plots for Sepal and Petal measurements.

### Deployment

Created an interactive Streamlit app with:

Colorful slider inputs

Animated predictions using emojis

User-friendly interface

Footer credit

## 📊 Results

| Model          | Accuracy | Macro F1-score |
| -------------- | -------- | -------------- |
| Linear SVM     | 1.0      | 1.0            |
| Polynomial SVM | 0.9      | 0.8977         |
| RBF SVM        | 0.9667   | 0.9666         |

Confusion Matrices:

Linear SVM

[[10  0  0]
 [ 0 10  0]
 [ 0  0 10]]

Polynomial SVM

[[10  0  0]
 [ 0 10  0]
 [ 0  3  7]]

RBF SVM

[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]

## 🛠 Technology and Libraries

Python 3

pandas

numpy

scikit-learn

seaborn

matplotlib

streamlit

pickle
