# 💳 Credit Card Fraud Detection using Imbalanced Data Techniques

This project builds a machine learning pipeline to detect fraudulent credit card transactions, where fraud instances are extremely rare compared to genuine transactions.  
It handles **highly imbalanced data**, supports both **supervised classification** and **unsupervised anomaly detection**, and includes model saving and real-time prediction.

---

## 🧠 Features

- ✅ Handles severe class imbalance with **SMOTE** or **Random UnderSampling**
- ✅ Supports **supervised** models: Logistic Regression, Random Forest
- ✅ Includes **unsupervised** anomaly detectors: Isolation Forest, Local Outlier Factor
- ✅ Auto-saves the trained model
- ✅ Allows single-transaction predictions using JSON input

---

## 📁 Project Structure


---

## 📊 Dataset Used

I recommend starting with the **[Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**  
- 284,807 transactions  
- 492 frauds (≈ 0.17%)  
- 30 numerical features (`V1` to `V28`, `Amount`, `Time`)  
- Target column: `Class` (0 = legit, 1 = fraud)

---

## 🛠️ Installation

```bash
git clone https://github.com/yourname/credit_card_fraud.git
cd credit_card_fraud

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn joblib
