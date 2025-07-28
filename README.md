# ML-DRIVEN-MITM-ATTACK-DETECTION-IN-SMART-GRID
Project Overview

This project focuses on detecting Man-in-the-Middle (MITM) attacks in Smart Grid systems using Machine Learning (ML). It addresses vulnerabilities in the Modbus protocol, particularly to ARP spoofing, by evaluating seven ML models—XGBoost, Random Forest, Logistic Regression, SVM, Decision Tree, kNN, and Gradient Boosting—for anomaly detection in Modbus TCP/IP traffic.

The models are tested in both default and Genetic Algorithm (GA)-optimized configurations using a dataset of 21,384 network packets with 12 features. Key preprocessing steps include handling noisy values, feature selection with Recursive Feature Elimination (RFE), class balancing with SMOTE, and feature scaling. The GA-optimized Gradient Boosting model achieved the best performance: Accuracy 90%, Precision 88%, Recall 92%, F1-Score 90%, and ROC AUC 98%.

Repository Contents





final_report.pdf: Comprehensive project documentation covering introduction, methodology, implementation, results, and references.



code/: Directory containing the Python script (mitm_detection.py) for data preprocessing, model training, GA optimization, and evaluation.



results/: Directory with output files:





model_performance_results.xlsx: Performance metrics for all models.



classification_reports.txt: Detailed classification reports.



confusion_matrices.png: Visualizations of confusion matrices.



model_accuracy_barplot.png: Bar plot comparing model accuracies.



dataset/: Placeholder for modbus_dataset_11.xlsx (not included due to size; available upon request).

Installation





Clone the Repository:

git clone https://github.com/your-username/ml-mitm-smart-grid.git
cd ml-mitm-smart-grid



Set Up Environment:





Use Google Colab (recommended) or a local Python environment (Python 3.9+).



Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost imblearn deap joblib



Dataset:





Place modbus_dataset_11.xlsx in the dataset/ directory or update the script's file path.



Ensure the dataset is in Excel format with expected features (protocol_encoded, Function Code, etc.).

Usage





Run the Script:





Open code/mitm_detection.py in Google Colab or a local IDE.



Update the dataset path in the script if necessary.



Execute the script to preprocess data, train models, optimize with GA, and generate results.

python code/mitm_detection.py



Output:





Results are saved in the results/ directory as Excel, text, and PNG files.



Review model_performance_results.xlsx for metrics and confusion_matrices.png for visualizations.

Methodology





Data Preprocessing:





Handle noisy values (e.g., -1 replaced with median).



Select top 8 features using RFE with Random Forest.



Balance classes with SMOTE.



Split data (80% train, 20% test) with stratification.



Apply Standard searing for feature scaling.



Models:





Evaluated: XGBoost, Random Forest, Logistic Regression, SVM, Decision Tree, kNN, Gradient Boosting.



Metrics: Accuracy, Precision, Recall, F1-Score, ROC AUC, Cross-Validation F1-Score.



Optimization:





Genetic Algorithm with 20 individuals, 10 generations, and tournament selection.



Hyperparameter spaces defined for each model (e.g., n_estimators: 50-200 for XGBoost).

Results





Default Models: Accuracy ranged from 87% (kNN) to 89% (Gradient Boosting).



GA-Optimized Models: Gradient Boosting achieved the highest performance:





Accuracy: 90%



Precision: 88%



Recall: 92%



F1-Score: 90%



ROC AUC: 98%



Gradient Boosting is recommended for its high Recall, minimizing missed attacks.

Future Work





Explore faster optimization techniques (e.g., Bayesian optimization).



Test models on larger, real-world Modbus datasets.



Deploy the Gradient Boosting model in a live Smart Grid environment for real-time validation.


Conclusion
The project demonstrates that:
1. Machine Learning is highly effective for detecting MITM attacks in Modbus-based Smart Grids.
2. Genetic Algorithm significantly improves model performance.
3. Gradient Boosting is the most reliable model in terms of accuracy, recall, and generalization.

Future Work
  Explore faster optimization techniques (e.g., Bayesian optimization).
  Test models on larger, real-world Modbus datasets.
  Deploy the Gradient Boosting model in a live Smart Grid environment for real-time validation.
