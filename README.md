# ML-DRIVEN-MITM-ATTACK-DETECTION-IN-SMART-GRID
#  ML-Driven MITM Attack Detection in Smart Grid

> A machine learning-powered intrusion detection system for identifying Man-in-the-Middle (MITM) attacks in Modbus TCP/IP-based smart grid networks.

![Smart Grid Security](https://img.shields.io/badge/smart--grid-cybersecurity-blue)
![MITM Detection](https://img.shields.io/badge/MITM-Detection-red)
![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Overview

The increasing reliance on digital communication in Smart Grids—particularly through protocols like **Modbus TCP/IP**—has introduced new cybersecurity risks. This project presents a **software-based Intrusion Detection System (IDS)** using **Machine Learning (ML)** to detect **Man-in-the-Middle (MITM)** attacks, specifically ARP spoofing, in real-time data from Advanced Metering Infrastructure (AMI).

>  Our best model achieved **90% accuracy**, **92% recall**, and **98% ROC AUC**, making it highly reliable for real-world deployment.

---

##  Objectives

-  Detect MITM attacks using multiple ML classifiers.
-  Optimize model performance using **Genetic Algorithm (GA)**.
-  Identify the most effective model for deployment in smart grid networks.

---

##  Dataset

- **Source**: Custom Modbus TCP/IP packet captures (PCAPs)
- **Rows**: 21,384
- **Features**: 12 (including protocol, function codes, transaction ID, packet size, timing)
- **Label**: Binary (0: Normal, 1: Attack)

>  Preprocessing: Handled noisy values (e.g. -1), applied RFE for feature selection, balanced with SMOTE, and scaled using StandardScaler.

---

##  Models & Techniques

| Model               | Description                                      |
|--------------------|--------------------------------------------------|
| Logistic Regression| Linear classifier for binary classification     |
| kNN                | Lazy learner based on nearest neighbors          |
| Decision Tree      | Hierarchical, rule-based classifier              |
| SVM                | Margin-based classifier with RBF kernel          |
| Random Forest      | Ensemble of decision trees using bagging         |
| Gradient Boosting  | Sequential boosting of weak learners             |
| XGBoost            | Optimized gradient boosting with regularization  |

###  Hyperparameter Optimization
- Implemented using **Genetic Algorithm (GA)** from the `deap` library.
- Tuned over 10 generations × 20 individuals = 200 configurations/model.
- Evaluation metric: **Accuracy** (with focus on Recall).

---

##  Results Summary

| Model       | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------------|----------|-----------|--------|----------|---------|
| Gradient Boosting (GA) | **90%**  | 88%      | **92%** | 90%      | **98%** |
| Random Forest (GA)     | 90%      | 89%      | 91%    | 90%      | 97%     |
| SVM (GA)               | 90%      | 88%      | 90%    | 89%      | 97%     |

 **Gradient Boosting** with GA optimization outperformed others across all key metrics.

---


## Conclusion
The project demonstrates that:
1. Machine Learning is highly effective for detecting MITM attacks in Modbus-based Smart Grids.
2. Genetic Algorithm significantly improves model performance.
3. Gradient Boosting is the most reliable model in terms of accuracy, recall, and generalization.

---

## Future Work
-  Explore faster optimization techniques (e.g., Bayesian optimization).
-  Test models on larger, real-world Modbus datasets.
-  Deploy the Gradient Boosting model in a live Smart Grid environment for real-time validation.

---

## Clone the repository:
-  git clone https://github.com/your-username/MITM-Detection-SmartGrid.git
-  cd MITM-Detection-SmartGrid
