# ML-DRIVEN-MITM-ATTACK-DETECTION-IN-SMART-GRID
Project Overview

This project focuses on detecting Man-in-the-Middle (MITM) attacks in Smart Grid systems using Machine Learning (ML). It addresses vulnerabilities in the Modbus protocol, particularly to ARP spoofing, by evaluating seven ML models—XGBoost, Random Forest, Logistic Regression, SVM, Decision Tree, kNN, and Gradient Boosting—for anomaly detection in Modbus TCP/IP traffic.

The models are tested in both default and Genetic Algorithm (GA)-optimized configurations using a dataset of 21,384 network packets with 12 features. Key preprocessing steps include handling noisy values, feature selection with Recursive Feature Elimination (RFE), class balancing with SMOTE, and feature scaling. The GA-optimized Gradient Boosting model achieved the best performance: Accuracy 90%, Precision 88%, Recall 92%, F1-Score 90%, and ROC AUC 98%.

Installation
Clone the Repository:
git clone https://github.com/your-username/ml-mitm-smart-grid.git
cd ml-mitm-smart-grid


📌 Conclusion
The project demonstrates that:
1. Machine Learning is highly effective for detecting MITM attacks in Modbus-based Smart Grids.
2. Genetic Algorithm significantly improves model performance.
3. Gradient Boosting is the most reliable model in terms of accuracy, recall, and generalization.
