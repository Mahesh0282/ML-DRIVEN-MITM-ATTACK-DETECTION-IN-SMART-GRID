# ML-DRIVEN-MITM-ATTACK-DETECTION-IN-SMART-GRID
🔐 ML-Driven MITM Attack Detection in Smart Grid
This repository contains the implementation of a Machine Learning (ML)-based Intrusion Detection System (IDS) for detecting Man-in-the-Middle (MITM) attacks in Smart Grid systems, specifically in Modbus TCP/IP environments. The project leverages various ML models, with performance boosted using Genetic Algorithm (GA) optimization.

🧠 Project Objective
Detect MITM (Man-in-the-Middle) attacks using various machine learning models.

Enhance model performance using Evolutionary Algorithms (Genetic Algorithm).

Evaluate and identify the best-performing model for deployment.

🗂 Dataset
The dataset used in this project (modbus_dataset.xlsx) is based on Modbus TCP/IP traffic data. It contains:

21,384 rows

12 features including function codes, byte count, protocol types, and a binary label indicating attack or normal traffic.

Key Features:
Packet Timing and Length

Function Code, Transaction ID, and Unit ID

Derived numerical features extracted from packet summaries (e.g., from the Info field)

Binary label: 0 = Normal, 1 = Attack

🔍 Methodology
Models Implemented:
Logistic Regression

K-Nearest Neighbors (kNN)

Decision Tree

Support Vector Machine (SVM)

Gradient Boosting

XGBoost

Random Forest

Workflow:
Data Preprocessing

Handling missing and noisy values

Feature selection via Recursive Feature Elimination (RFE)

Data balancing using SMOTE

Standardization using StandardScaler

Model Training

Baseline training with default hyperparameters

Hyperparameter tuning using Genetic Algorithm (GA)

Evaluation Metrics

Accuracy, Precision, Recall, F1-score, ROC-AUC

⚙️ Technologies Used
Python 3.9

Google Colab (for cloud-based execution)

Libraries: pandas, numpy, sklearn, xgboost, deap, imbalanced-learn, matplotlib, seaborn

📊 Results Summary
| Model                 | Accuracy (Default) | Accuracy (Optimized) | Best Metric (GA)      |
| --------------------- | ------------------ | -------------------- | --------------------- |
| Gradient Boosting     | 89%                | 90%                  | Recall: 92%, AUC: 98% |
| XGBoost               | 89%                | 90%                  | High Precision        |
| Random Forest         | 89%                | 90%                  | Robustness            |
| Logistic Regression   | 88%                | 89%                  | Highest Recall (93%)  |
| Others (SVM, DT, kNN) | \~88%              | \~89-90%             | Good ROC-AUC          |


✅ Optimized Gradient Boosting gave the best trade-off among all models.

📌 Conclusion
The project demonstrates that:
1. Machine Learning is highly effective for detecting MITM attacks in Modbus-based Smart Grids.
2. Genetic Algorithm significantly improves model performance.
3. Gradient Boosting is the most reliable model in terms of accuracy, recall, and generalization.
