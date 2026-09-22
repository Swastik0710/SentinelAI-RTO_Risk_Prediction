# 🚚 SentinelAI — Return-to-Origin (RTO) Risk Prediction

> An end-to-end Machine Learning application that predicts the likelihood of an e-commerce order being returned to the seller (RTO) using behavioral, customer, seller, and logistics features.

## 📌 Overview

**SentinelAI** is a predictive analytics project built to address one of the most common operational challenges in e-commerce: **Return-to-Origin (RTO)**.

The system analyzes historical order patterns and generates an RTO probability using a **Random Forest Classifier**. It also provides an interactive **Streamlit dashboard** where users can enter order details and receive real-time risk predictions.

### Why this project?

RTO orders increase shipping costs, delay inventory turnover, and reduce overall profitability. Instead of reacting after a failed delivery, SentinelAI helps identify potentially risky orders **before dispatch**.

---

## ✨ Key Features

* 🔍 Real-time RTO prediction through a Streamlit web app
* 🤖 Machine Learning model trained using Random Forest
* 📊 Interactive probability and risk visualization
* 📈 Feature importance analysis
* 🧹 Data preprocessing and label encoding pipeline
* 💾 Model serialization with Joblib
* 📱 Clean and responsive user interface

---

## 🧠 Machine Learning Pipeline

The complete workflow follows a standard ML lifecycle:

```text
CSV Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Categorical Encoding
      │
      ▼
Train / Test Split
      │
      ▼
Random Forest Training
      │
      ▼
Model Evaluation
      │
      ▼
Feature Importance
      │
      ▼
Streamlit Web Application
```

---

## 📂 Dataset Information

The dataset contains **35 predictive features** and **1 target variable**.

### Feature Categories

| Category           | Examples                           |
| ------------------ | ---------------------------------- |
| Order Details      | Order value, quantity, discount    |
| Customer History   | Previous orders, previous RTO rate |
| Seller Information | Seller rating, seller RTO rate     |
| Logistics          | Distance, delivery days            |
| Behavioral Signals | Night order, weekend order         |
| Payment            | Payment method, COD amount         |

### Target Variable

| Value | Meaning       |
| ----- | ------------- |
| `0`   | Low RTO Risk  |
| `1`   | High RTO Risk |

---

## 🏗️ Project Structure

```text
SentinelAI-RTO-Risk-Prediction/
│
├── app.py
├── rto_prediction_model.pkl
├── SentinelAI_RTO_Risk_Dataset.csv
├── requirements.txt
├── Output.ipynb
└── README.md
```

---

## ⚙️ Technology Stack

### Programming & ML

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Joblib

### Visualization

* Matplotlib
* Seaborn

### Web Framework

* Streamlit

### Version Control

* Git & GitHub

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/SentinelAI-RTO-Risk-Prediction.git

cd SentinelAI-RTO-Risk-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python -m streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## 📊 Model Evaluation

The model is evaluated using multiple classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* Feature Importance

These metrics help measure how effectively the model distinguishes between low-risk and high-risk orders.

---

## 🖥️ Application Preview

The Streamlit interface allows users to enter values such as:

* Order Value
* Quantity
* Discount Percentage
* Payment Method
* Customer Order Count
* Previous RTO Rate
* Seller RTO Rate
* Distance
* Delivery Days
* COD Amount

After clicking **Predict**, the application displays:

* ✅ Risk Classification
* 📈 RTO Probability
* 📊 Top Important Features

---

## 💡 Example Workflow

```text
User enters order details
          │
          ▼
Input features are processed
          │
          ▼
Random Forest model predicts
          │
          ▼
RTO probability generated
          │
          ▼
Risk level displayed in Streamlit
```

---

## 🔮 Future Improvements

Potential enhancements for future versions include:

* SHAP-based model explainability
* Live database integration
* Batch CSV prediction
* Customer fraud detection module
* Hyperparameter optimization
* Cloud deployment with authentication
* Historical prediction dashboard

---

## 🎓 Learning Outcomes

This project demonstrates practical knowledge of:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Supervised Machine Learning
* Model evaluation
* Model deployment
* Streamlit application development
* GitHub project management

---

## 👨‍💻 Author

**Swastik Bhattacharya**

B.Tech — Electronics & Communication Engineering

Machine Learning • IoT • Software Development

---

## ⭐ If you found this project useful

Consider giving the repository a **Star ⭐** on GitHub to support the project.
