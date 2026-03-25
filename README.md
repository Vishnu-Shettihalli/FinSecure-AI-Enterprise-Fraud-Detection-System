# 💳 FinSecure AI – Enterprise Fraud Detection System

![Streamlit](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Live-success)
![License](https://img.shields.io/badge/License-Portfolio-lightgrey)

🚀 **Live Application:**  
👉 https://finsecure-ai-enterprise-fraud-detection.streamlit.app/

---

## 🧠 Project Overview

FinSecure AI is an enterprise-grade fraud detection system designed to identify high-risk credit card transactions in real time.  

The system leverages advanced machine learning techniques including **SMOTE balancing**, **XGBoost classification**, and **dynamic threshold tuning** to deliver highly optimized fraud detection performance.

This project demonstrates end-to-end ML engineering:
- Data preprocessing
- Handling class imbalance
- Model comparison
- Evaluation metrics
- Feature importance analysis
- Production-safe deployment

---

## 🎯 Business Problem

Credit card fraud detection presents a highly imbalanced classification problem where fraudulent transactions represent a very small percentage of total transactions.

Key challenges:
- Extreme class imbalance
- High cost of false negatives
- Need for real-time risk scoring
- Model interpretability requirements

FinSecure AI addresses these challenges using advanced ML optimization techniques.

---

## 📊 Model Performance

- **Algorithm:** XGBoost Classifier
- **Data Handling:** SMOTE (Synthetic Minority Oversampling Technique)
- **ROC-AUC Score:** 0.97+
- **Threshold:** Dynamically adjustable
- **Evaluation Metrics:** Precision, Recall, F1-score, Confusion Matrix, ROC Curve

The system prioritizes fraud recall while maintaining controlled false positive rates.

---

## 🏗 System Architecture
```
User Input (Demo / CSV Upload)
↓
Data Preprocessing & Scaling
↓
Feature Alignment & Validation
↓
XGBoost Model Prediction
↓
Fraud Probability Score
↓
Dynamic Threshold Engine
↓
Risk Classification Output
↓
Downloadable Risk Report
```


---

## 📸 Application Preview

### 🖥 Dashboard Overview
![Dashboard](assets/dashboard_overview.png)

### 📈 Risk Assessment Output
![Risk Analysis](assets/risk_analysis.png)

### 🧠 Model Explainability
![Explainability](assets/model_explainability.png)

---

## ⚙️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **Scikit-Learn**
- **XGBoost**
- **SMOTE (Imbalanced-Learn)**
- **Matplotlib & Seaborn**
- **Streamlit (Deployment)**
- **Joblib (Model Serialization)**

---

## 🧪 Key Features

✔ Real-time fraud probability scoring  
✔ Adjustable decision threshold  
✔ Risk visualization with progress indicator  
✔ CSV transaction upload support  
✔ Automated feature schema validation  
✔ Downloadable fraud risk reports  
✔ Deployment-ready Streamlit application  
✔ Production-safe environment handling  

---

## 📂 Project Structure
```
Fraud-Detection-ML/
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── models/
│ ├── xgboost_model.pkl
│ └── scaler.pkl
│
├── src/
│ ├── data_preprocessing.py
│ ├── train_models.py
│ └── evaluate.py
│
├── assets/
│ ├── dashboard_overview.png
│ └── risk_analysis.png
│
└── outputs/
```


---

## 🧠 Why XGBoost?

XGBoost was selected due to:
- Superior performance on tabular data
- Strong handling of non-linear relationships
- Built-in regularization
- High performance in imbalanced datasets
- Industry adoption in fintech risk modeling

---

## 📈 Handling Class Imbalance

Fraud detection datasets are highly skewed.

This project applies:

- **SMOTE oversampling**
- Threshold tuning
- ROC-AUC optimization
- Precision-Recall tradeoff analysis

This ensures improved fraud detection sensitivity.

---

## 🚀 Deployment

The system is deployed using **Streamlit Cloud**, enabling:

- Public access
- Live inference
- Interactive threshold control
- Real-time transaction evaluation

---

## 🧑‍💻 How to Run Locally

```bash
git clone https://github.com/Vishnu-Shettihalli/FinSecure-AI---Enterprise-Fraud-Detection-System.git
cd FinSecure-AI---Enterprise-Fraud-Detection-System
pip install -r requirements.txt
streamlit run app.py
