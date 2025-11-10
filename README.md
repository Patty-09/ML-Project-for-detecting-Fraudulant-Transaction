# ML-Project-for-detecting-Fraudulant-Transaction
Machine Learning–based Fraud Detection System using seven classification algorithms like Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Naïve Bayes, and XGBoost-to identify and prevent fraudulent financial transactions with high accuracy.

#  Fraud Detection System Using Machine Learning

##  Project Overview
The **Fraud Detection System** is a comprehensive machine learning project designed to identify and prevent fraudulent financial transactions. With the rapid growth of digital payments, detecting fraud in real time has become a significant challenge for financial institutions.  
This project applies **supervised machine learning algorithms** to analyze transaction patterns and predict fraudulent activities with high accuracy and reliability.

## Objectives
- Build a predictive model to detect fraudulent transactions.  
- Compare the performance of multiple machine learning algorithms.  
- Handle class imbalance and data noise effectively.  
- Achieve high precision and recall to minimize false predictions.  

##  Dataset Description
- Contains anonymized financial transaction records.  
- **Categorical features:** `Transaction_Type`, `Device_Type`, `Location`, `Merchant_Category`, `Card_Type`, `Authentication_Method`.  
- **Numerical features:** `Transaction_Amount`, `Account_Balance`, `Daily_Transaction_Count`, `Avg_Transaction_Amount_7d`, `Failed_Transaction_Count_7d`, `Risk_Score`, `Card_Age`.  
- Removed irrelevant columns such as `Transaction_ID` and `User_ID`.  
- Outliers were detected and handled using the **IQR method**.  
- Missing values were treated, and data was normalized for uniform scaling.  

---

##  Methodology

1. **Data Preprocessing**
   - Handled missing values and duplicates.  
   - Encoded categorical variables and scaled numerical features.  
   - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance.  

2. **Exploratory Data Analysis (EDA)**
   - Studied feature distributions, correlations, and outliers.  
   - Used **Matplotlib** and **Seaborn** for data visualization.  

3. **Model Training and Evaluation**
   - Trained **7 different machine learning algorithms** and compared their performance:  

     | Algorithm | Description | Accuracy |
     |------------|--------------|-----------|
     | Logistic Regression | Baseline linear classifier to predict fraud probability. | **88.5%** |
     | Decision Tree Classifier | Simple, interpretable tree-based model. | **90.2%** |
     | Random Forest Classifier | Ensemble of decision trees for better accuracy and generalization. | **95.3%** |
     | K-Nearest Neighbors (KNN) | Instance-based learner classifying based on nearest data points. | **87.8%** |
     | Support Vector Machine (SVM) | Separates fraud and non-fraud classes with optimal hyperplane. | **91.6%** |
     | Naïve Bayes Classifier | Probabilistic model suitable for high-dimensional data. | **86.9%** |
     | XGBoost (Extreme Gradient Boosting) | Boosted ensemble model offering high accuracy and efficiency. | **96.1%** |

4. **Performance Metrics**
   - Evaluated using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.  
   - Generated confusion matrices and ROC curves for each model.  

##  Key Findings
- **XGBoost** achieved the highest accuracy (**96.1%**) with strong precision and recall.  
- **Random Forest** performed nearly as well (**95.3%**) and offered more interpretability.  
- Models like **Naïve Bayes** and **KNN** performed decently but were less robust to data imbalance.  
- After tuning hyperparameters and applying cross-validation, performance stabilized across all models.  

##  Technologies Used
- **Python**  
- **Pandas**, **NumPy**  
- **Matplotlib**, **Seaborn**  
- **Scikit-learn**, **XGBoost**  
- **Jupyter Notebook**

##  Results Summary

| Metric | Best Model (XGBoost) |
|:--------|:---------------------|
| Accuracy | **96.1%** |
| Precision | **93%** |
| Recall | **91%** |
| F1-Score | **92%** |
| ROC-AUC | **0.97** |


## Future Enhancements
- Integrate **Deep Learning models** (e.g., LSTM, Autoencoders) for advanced anomaly detection.  
- Deploy the model as a **real-time fraud monitoring system**.  
- Use **Explainable AI (XAI)** to interpret model predictions and build trust.  
- Connect to a live database or API for dynamic fraud detection.  

