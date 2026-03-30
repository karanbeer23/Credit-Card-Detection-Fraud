# Credit-Card-Detection-Fraud
# Credit Card Fraud Detection using TPOT 🛡️💳

An automated machine learning (AutoML) approach to detecting fraudulent financial transactions using genetic programming.

## 📌 Project Overview
Credit card fraud detection is a classic case of a highly imbalanced classification problem. This project utilizes **TPOT** (Tree-based Pipeline Optimization Tool) to automatically search through thousands of machine learning pipelines (including various algorithms and preprocessing steps) to find the most effective model for this specific dataset.

## 📊 Dataset
The project uses the **Credit Card Fraud Detection** dataset from Kaggle.
- **Transactions:** 284,807
- **Fraudulent cases:** 492 (0.172% of all transactions)
- **Features:** PCA-transformed features (V1-V28), `Time`, and `Amount`.
- **Target:** `Class` (1 for fraud, 0 otherwise).

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **AutoML:** [TPOT](https://epistasislab.github.io/tpot/)
* **Data Handling:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
* **Environment:** Google Colab / Jupyter Notebook

## 🚀 Key Features
- **AutoML Integration:** Uses Genetic Programming to optimize the ML pipeline.
- **Exploratory Data Analysis (EDA):** Visualizing transaction distributions and fraud correlation.
- **Handling Imbalance:** Optimized for metrics like Recall and F1-Score rather than just accuracy.
- **Automated Code Generation:** TPOT exports the best performing Python code for the final model.



Bash
pip install tpot pandas matplotlib seaborn scikit-learn
Run the Notebook:
Open Credit_Card_Fraud_Detection_using_TPOT.ipynb in Google Colab or Jupyter.

📈 Results
The final model identified by TPOT focuses on minimizing False Negatives, which is crucial in fraud detection to ensure as many fraudulent transactions as possible are caught.
